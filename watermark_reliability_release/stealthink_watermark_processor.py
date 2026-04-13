# Adapted from StealthInk codebase
# https://github.com/yajiang4215/StealthInk_A-Multi-bit-and-Stealthy-Watermark-for-Large-Language-Models

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import torch
from torch import Tensor
from transformers import LogitsProcessor

from mb_watermark_processor import WatermarkBase
from normalizers import normalization_strategy_lookup

try:
    from tokenizers import Tokenizer
except Exception:  # type: ignore
    Tokenizer = object  # fallback type for static checkers


##########################################################################
# Generation-side components
##########################################################################


class ReweightProcessor(WatermarkBase):
    """
    Thin wrapper around WatermarkBase that exposes a standalone reweight()
    function (CPU-only reference implementation). The generation-time
    logits processor below uses a fully vectorized variant of the same
    mapping on the active device, so this class primarily provides
    seeding, message handling and vocabulary size information.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reweight(
        self,
        seed: Tensor,
        original_token_probs: Dict[int, float],
        pos_embedded_message: int,
        base: int,
    ) -> Tensor:
        """
        Reference reweight operation (Fig.2-style piecewise mapping).
        Operates on CPU and returns log-probabilities over the vocab.
        """
        # Seed RNG from context
        if self.rng is None:
            self.rng = torch.Generator(device="cpu")
        self._seed_rng(seed)

        vocab_perm = (
            torch.randperm(self.vocab_size, device="cpu", generator=self.rng)
            .detach()
            .cpu()
            .tolist()
        )
        colorlist = torch.chunk(torch.tensor(vocab_perm), base)
        original_probs_tensor = torch.tensor(
            [original_token_probs[tok] for tok in vocab_perm],
            dtype=torch.float64,
        )

        red_tokens_alpha = 0
        red_tokens_beta = 0
        for i in range(base):
            if i < pos_embedded_message:
                red_tokens_alpha += len(colorlist[i])
            if i == pos_embedded_message:
                red_tokens_beta = red_tokens_alpha + len(colorlist[i])

        if red_tokens_alpha == 0:
            alpha = torch.tensor(0.0, dtype=torch.float64)
        else:
            alpha = original_probs_tensor.cumsum(dim=0)[red_tokens_alpha - 1]
        beta = original_probs_tensor.cumsum(dim=0)[red_tokens_beta - 1]

        acc = original_probs_tensor.cumsum(dim=0)
        acc = torch.cat((torch.tensor([0.0], dtype=torch.float64), acc))

        if alpha >= 0.5 or beta <= 0.5:
            if alpha >= 0.5:  # 2p p 0
                a, b, c, d = 1 - beta, 1 - alpha, alpha, beta
                mapped = torch.where(
                    acc <= a,
                    acc - d,
                    torch.where(
                        acc <= b,
                        2 * acc - 1,
                        torch.where(
                            acc <= c,
                            acc - c,
                            torch.where(
                                acc <= d,
                                torch.zeros(1, dtype=torch.float64),
                                acc - d,
                            ),
                        ),
                    ),
                )
            else:  # beta <= 0.5, 0 p 2p
                a, b, c, d = alpha, beta, 1 - beta, 1 - alpha
                mapped = torch.where(
                    acc <= a,
                    acc - a,
                    torch.where(
                        acc <= b,
                        torch.zeros(1, dtype=torch.float64),
                        torch.where(
                            acc <= c,
                            acc - b,
                            torch.where(
                                acc <= d,
                                2 * acc - 1,
                                acc - a,
                            ),
                        ),
                    ),
                )
        else:
            if alpha <= 1 - beta <= beta <= 1 - alpha:  # alpha+beta<1 -> 0 p 2p
                a, b, c, d = alpha, 1 - beta, beta, 1 - alpha
                mapped = torch.where(
                    acc <= a,
                    acc - a,
                    torch.where(
                        acc <= b,
                        torch.zeros(1, dtype=torch.float64),
                        torch.where(
                            acc <= c,
                            acc - b,
                            torch.where(
                                acc <= d,
                                2 * acc - 1,
                                acc - a,
                            ),
                        ),
                    ),
                )
            else:  # alpha+beta>1 -> 2p p 0
                a, b, c, d = 1 - beta, alpha, 1 - alpha, beta
                mapped = torch.where(
                    acc <= a,
                    acc - d,
                    torch.where(
                        acc <= b,
                        2 * acc - 1,
                        torch.where(
                            acc <= c,
                            acc - c,
                            torch.where(
                                acc <= d,
                                torch.zeros(1, dtype=torch.float64),
                                acc - d,
                            ),
                        ),
                    ),
                )

        reweighted_probs = mapped[1:] - mapped[:-1]
        combined = {k: v for k, v in zip(vocab_perm, reweighted_probs)}
        sorted_vals = torch.tensor(
            [combined[k] for k in sorted(combined.keys())],
            dtype=torch.float64,
        )
        v_non_zero = torch.where(
            sorted_vals > 0,
            sorted_vals,
            torch.tensor(1e-50, dtype=torch.float64),
        )
        logits = torch.log(v_non_zero).to(dtype=torch.float32)
        return logits


class ReweightLogitsProcessor(LogitsProcessor):
    """
    StealthInk-style logits processor implementing the reweighting rule.

    This is wired to be drop-in compatible with utils/generation.generate():
    - exposes set_message() and converted_message
    - exposes flush_position() and _get_and_clear_stored_spike_ents()
    - tracks is_r/output_logits for debugging/analysis (unused by pipeline).
    """

    def __init__(
        self,
        reweight_processor: ReweightProcessor,
        R: float,
        seen_seeds: Optional[set] = None,
        cache_max: int = 50000,
    ):
        super().__init__()
        self.reweight_processor = reweight_processor
        # R is the red-list mass fraction; base is its reciprocal (integer)
        self.R = float(R)
        self.base = int(1.0 / self.R) if self.R > 0 else 2
        self.converted_msg_length = self.reweight_processor.converted_msg_length
        self.n_gram_len = self.reweight_processor.context_width

        self.embedded_message: List[int] = []
        self.converted_message: str = ""
        self.output_logits: Optional[Tensor] = None

        self.seen_seeds = seen_seeds if seen_seeds is not None else set()
        self.is_r = False

        # cache: seed_tuple -> (vocab_perm (cpu tensor), colorlist_indices)
        self._perm_cache: Dict[Tuple[int, ...], Tuple[Tensor, Tuple[Tensor, ...]]] = {}
        self._cache_max = cache_max

        # Compatibility fields for utils/generation.py
        self.position_increment = 0
        self.spike_entropies = None

    # ------------------------------------------------------------------
    # Message handling / compatibility helpers
    # ------------------------------------------------------------------

    def set_message(self, binary_msg: str):
        """
        Convert binary message to base-`self.base` digits and store both
        in this processor and the underlying ReweightProcessor.
        """
        # StealthInk uses the same conversion logic as WatermarkBase
        self.reweight_processor.set_message(binary_msg)
        self.converted_message = self.reweight_processor.converted_message
        self.converted_msg_length = self.reweight_processor.converted_msg_length
        self.embedded_message = [int(c) for c in self.converted_message]

    def flush_position(self) -> List[str]:
        """
        Return sampled bit positions for analysis.
        Delegates to the underlying WatermarkBase implementation.
        """
        return self.reweight_processor.flush_position()

    def _get_and_clear_stored_spike_ents(self):
        """
        For compatibility with mb_watermark_processor.WatermarkLogitsProcessor.
        StealthInk does not currently track spike entropies.
        """
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_perm_and_chunks(
        self, seed: Tensor, base: int, vocab_size: int
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        seed_tuple = tuple(seed.view(-1).tolist())
        hit = self._perm_cache.get(seed_tuple)
        if hit is not None:
            return hit

        # Use CPU RNG for permutation to avoid device mismatches
        if self.reweight_processor.rng is None:
            self.reweight_processor.rng = torch.Generator(device="cpu")
        self.reweight_processor._seed_rng(seed)
        vocab_perm = torch.randperm(
            vocab_size, device="cpu", generator=self.reweight_processor.rng
        )
        colorlist = torch.chunk(vocab_perm, base)

        if len(self._perm_cache) >= self._cache_max:
            self._perm_cache.clear()
        self._perm_cache[seed_tuple] = (vocab_perm, colorlist)
        return self._perm_cache[seed_tuple]

    # ------------------------------------------------------------------
    # Core logits processing
    # ------------------------------------------------------------------

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        """
        Apply StealthInk reweighting to the logits.

        Args:
            input_ids: [B, T] integer ids for current sequences.
            scores: [B, V] float logits for next token.
        """
        device = scores.device
        batch_size, vocab_size = scores.shape

        # Ensure RNG exists (CPU-based for reproducible permutations)
        if self.reweight_processor.rng is None:
            self.reweight_processor.rng = torch.Generator(device="cpu")

        logits_out = scores.clone()

        for b in range(batch_size):
            # If prefix shorter than seeding width, skip watermarking
            if input_ids.shape[1] < self.n_gram_len or not self.embedded_message:
                continue

            # Seed is last n_gram_len tokens for this sequence
            seed = input_ids[b : b + 1, -self.n_gram_len :]
            seed_tuple = tuple(seed.view(-1).tolist())

            # Skip repeated seed (purely in-memory; no disk I/O)
            if seed_tuple in self.seen_seeds:
                self.is_r = True
                continue
            self.seen_seeds.add(seed_tuple)
            self.is_r = False

            # Bit position from RNG
            self.reweight_processor._seed_rng(seed)
            bit_pos = torch.randint(
                low=0,
                high=self.converted_msg_length,
                size=(1,),
                generator=self.reweight_processor.rng,
            ).item()
            pos_embedded_message = self.embedded_message[bit_pos]

            # Probs for this sequence on the same device
            probs = torch.softmax(scores[b], dim=-1)  # [V] on device

            # Permutation and chunking on CPU, then index on device
            vocab_perm_cpu, colorlist = self._get_perm_and_chunks(
                seed, self.base, vocab_size
            )
            vocab_perm = vocab_perm_cpu.to(device)

            # Reorder probs by permutation
            original_probs_tensor = probs.index_select(0, vocab_perm).to(
                torch.float64
            )

            # Compute alpha/beta via cumsum (vectorized)
            cdf = original_probs_tensor.cumsum(dim=0)
            chunk_sizes = [len(t) for t in colorlist]
            red_alpha = sum(chunk_sizes[:pos_embedded_message])
            red_beta = red_alpha + chunk_sizes[pos_embedded_message]

            alpha = (
                cdf[red_alpha - 1]
                if red_alpha > 0
                else torch.tensor(
                    0.0, dtype=torch.float64, device=device
                )
            )
            beta = cdf[red_beta - 1]

            # build acc = [0, cdf]
            acc = torch.cat(
                [
                    torch.zeros(1, dtype=torch.float64, device=device),
                    cdf,
                ],
                dim=0,
            )

            # piecewise mapping (same logic as reference, tensorized)
            if alpha >= 0.5 or beta <= 0.5:
                if alpha >= 0.5:  # 2p p 0
                    a, b2, c2, d = 1 - beta, 1 - alpha, alpha, beta
                    z = torch.where(
                        acc <= a,
                        acc - d,
                        torch.where(
                            acc <= b2,
                            2 * acc - 1,
                            torch.where(
                                acc <= c2,
                                acc - c2,
                                torch.where(
                                    acc <= d,
                                    torch.zeros_like(acc),
                                    acc - d,
                                ),
                            ),
                        ),
                    )
                else:  # beta <= 0.5, 0 p 2p
                    a, b2, c2, d = alpha, beta, 1 - beta, 1 - alpha
                    z = torch.where(
                        acc <= a,
                        acc - a,
                        torch.where(
                            acc <= b2,
                            torch.zeros_like(acc),
                            torch.where(
                                acc <= c2,
                                acc - b2,
                                torch.where(
                                    acc <= d,
                                    2 * acc - 1,
                                    acc - a,
                                ),
                            ),
                        ),
                    )
            else:
                if alpha <= 1 - beta <= beta <= 1 - alpha:  # alpha+beta<1 -> 0 p 2p
                    a, b2, c2, d = alpha, 1 - beta, beta, 1 - alpha
                    z = torch.where(
                        acc <= a,
                        acc - a,
                        torch.where(
                            acc <= b2,
                            torch.zeros_like(acc),
                            torch.where(
                                acc <= c2,
                                acc - b2,
                                torch.where(
                                    acc <= d,
                                    2 * acc - 1,
                                    acc - a,
                                ),
                            ),
                        ),
                    )
                else:  # alpha+beta>1 -> 2p p 0
                    a, b2, c2, d = 1 - beta, alpha, 1 - alpha, beta
                    z = torch.where(
                        acc <= a,
                        acc - d,
                        torch.where(
                            acc <= b2,
                            2 * acc - 1,
                            torch.where(
                                acc <= c2,
                                acc - c2,
                                torch.where(
                                    acc <= d,
                                    torch.zeros_like(acc),
                                    acc - d,
                                ),
                            ),
                        ),
                    )

            reweighted_probs = (z[1:] - z[:-1]).clamp_min(1e-50)

            # Map back to original vocab order
            logits_b = torch.full_like(
                probs, fill_value=-1e9, dtype=torch.float32
            )
            logits_b.index_copy_(
                0, vocab_perm, reweighted_probs.log().to(torch.float32)
            )
            logits_out[b] = logits_b

        self.output_logits = logits_out
        return logits_out


##########################################################################
# Detection-side components
##########################################################################


class DetectorProcessor(WatermarkBase):
    """
    Low-level helper for StealthInk detection.
    Provides consistent PRF-based permutations (colorlists) per context.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use a CPU RNG for deterministic permutations
        self.rng = torch.Generator(device="cpu")

    def _get_colorlist_ids(
        self, input_ids: torch.LongTensor, base: int
    ) -> Tuple[Tensor, ...]:
        self._seed_rng(input_ids)
        vocab_perm = torch.randperm(
            self.vocab_size, device="cpu", generator=self.rng
        )
        colorlist = torch.chunk(vocab_perm, base)
        return colorlist


def _compute_norm_p_val(
    cl_total: Dict[int, List[int]], R: float
) -> Tuple[float, List[List[int]], float]:
    """
    Compute StealthInk-style normal approximation p-value and
    per-position candidate digits from colorlist hit counts.
    """
    from scipy.stats import norm  # local import to avoid hard dependency where unused

    T_total = 0
    t_total = 0
    min_p_value = 10.0
    msg: List[List[int]] = []

    for _, value in cl_total.items():
        T = sum(value)
        if T:
            t = min(value)
            cur_msg = [i for i, v in enumerate(value) if v == t]
            msg.append(cur_msg)
            z = (t - R * T) / (math.sqrt(R * (1 - R) * T))
            cur_p_value = 1 - pow((1 - norm.cdf(z)), len(value))
            if cur_p_value < min_p_value:
                min_p_value = cur_p_value
            T_total += T
            t_total += t
        else:
            cur_msg = [int(random.choice(np.arange(len(value))))]
            msg.append(cur_msg)

    if T_total > 0:
        z_total = (t_total - R * T_total) / (
            math.sqrt(R * (1 - R) * T_total)
        )
        p_value = norm.cdf(z_total)
    else:
        z_total = 0.0
        p_value = 0.5

    return p_value, msg, z_total


class StealthInkDetector(DetectorProcessor):
    """
    Detector for StealthInk-style watermarks.

    This class exposes a detect() and dummy_detect() API compatible with
    utils/evaluation.py:load_detector and compute_z_score.
    """

    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: List[str] = None,
        ignore_repeated_ngrams: bool = False,
        R: float = 0.25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert device is not None, "Must pass device"
        assert tokenizer is not None, "Need tokenizer used for generation"

        self.device = device
        self.tokenizer = tokenizer
        self.z_threshold = float(z_threshold)
        self.R = float(R)

        self.normalizers = []
        if normalizers is None:
            normalizers = ["unicode"]
        for strategy in normalizers:
            self.normalizers.append(
                normalization_strategy_lookup(strategy)
            )
        self.ignore_repeated_ngrams = bool(ignore_repeated_ngrams)

    # --------------------------------------------------------------
    # Dummy detector output (mirrors WatermarkDetector.dummy_detect)
    # --------------------------------------------------------------

    def dummy_detect(
        self,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_all_window_scores: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
        return_bit_match: bool = True,
        return_z_score_max: bool = True,
    ):
        score_dict: Dict[str, object] = {}
        score_dict.update(dict(pred_message=""))
        score_dict.update(dict(sampled_positions=""))
        score_dict.update(dict(position_acc=float("nan")))
        score_dict.update(dict(bit_match=None))
        score_dict.update(dict(bit_acc=float("nan")))
        score_dict.update(dict(decoding_time=float("nan")))

        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=float("nan")))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=float("nan")))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=float("nan")))
        if return_z_score:
            score_dict.update(dict(z_score=float("nan")))
        if return_p_value:
            score_dict.update(dict(p_value=float("nan")))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=[]))
        if return_all_window_scores:
            score_dict.update(dict(window_list=[]))
        if return_z_at_T:
            score_dict.update(dict(z_score_at_T=torch.tensor([])))
            # Keep an explicit empty bit_acc_at_T sequence so that callers
            # relying on prefix-aware metrics see a consistent schema even
            # when prefix statistics are not computed.
            score_dict.update(dict(bit_acc_at_T=torch.tensor([])))

        output_dict: Dict[str, object] = {}
        if return_scores:
            output_dict.update(score_dict)
        if return_prediction:
            z_threshold = z_threshold if z_threshold is not None else self.z_threshold
            assert (
                z_threshold is not None
            ), "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = False
        return output_dict

    # --------------------------------------------------------------
    # Core scoring
    # --------------------------------------------------------------

    def _score_sequence(
        self,
        input_ids: Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
        return_bit_match: bool = True,
        message: str = "",
        **kwargs,
    ) -> Dict[str, object]:
        """
        StealthInk detection: accumulate colorlist counts per message digit,
        run normal-approximation test, and optionally compute bit accuracy.
        """
        # Sliding over all possible n-grams (context_width prefix + target)
        ids = input_ids.detach()
        if ids.dim() == 2:
            ids = ids.view(-1)
        ids = ids.to(device="cpu")
        n = int(ids.numel())
        ctx_w = self.context_width
        base = self.base

        if n <= ctx_w:
            # Not enough tokens for any scoring
            return self.dummy_detect(
                return_prediction=False,
                return_num_tokens_scored=return_num_tokens_scored,
                return_num_green_tokens=return_num_green_tokens,
                return_green_fraction=return_green_fraction,
                return_green_token_mask=return_green_token_mask,
                return_all_window_scores=False,
                return_z_score=return_z_score,
                return_z_at_T=return_z_at_T,
                return_p_value=return_p_value,
                return_bit_match=return_bit_match,
            )

        cl_total: Dict[int, List[int]] = {
            pos: [0 for _ in range(base)]
            for pos in range(self.converted_msg_length)
        }
        token_count = 0
        green_token_mask: List[bool] = []
        bit_positions: List[int] = []
        hit_indices: List[int] = []

        for idx in range(ctx_w, n):
            prefix = ids[idx - ctx_w : idx]
            target = int(ids[idx].item())

            # Determine which message digit index was used at this step
            self._seed_rng(prefix)
            bit_pos = torch.randint(
                low=0,
                high=self.converted_msg_length,
                size=(1,),
                generator=self.rng,
            ).item()

            colorlist = self._get_colorlist_ids(prefix, base)

            # Each token belongs to exactly one chunk
            hit_idx = -1
            for m, cl in enumerate(colorlist[:base]):
                if (cl == target).any().item():
                    cl_total[bit_pos][m] += 1
                    hit_idx = m
                    break

            token_count += 1
            if return_green_token_mask:
                # For now treat "green" as belonging to predicted chunk 0; this is a
                # placeholder to keep API shape; not used for StealthInk metrics.
                green_token_mask.append(hit_idx == 0)
            if return_z_at_T:
                bit_positions.append(int(bit_pos))
                hit_indices.append(int(hit_idx))

        # Aggregate stats
        R = self.R if self.R > 0 else (1.0 / float(base))
        p_value, msg_candidates, z_total = _compute_norm_p_val(cl_total, R)

        # Choose best digit per position (first candidate) to build prediction
        pred_digits: List[int] = []
        for cand_list in msg_candidates:
            if not cand_list:
                pred_digits.append(0)
            else:
                pred_digits.append(int(cand_list[0]))

        # Convert base-b digits to binary string
        decimal = 0
        for d in pred_digits:
            decimal = decimal * base + int(d)
        decimal = min(decimal, 2 ** self.message_length - 1)
        pred_binary = format(decimal, f"0{self.message_length}b")

        bit_acc = float("nan")
        bit_match = None
        if message:
            gold = message
            if len(pred_binary) == len(gold):
                match = sum(1 for g, p in zip(gold, pred_binary) if g == p)
                bit_acc = match / len(gold) if len(gold) > 0 else float(
                    "nan"
                )
                bit_match = match == len(gold)

        score_dict: Dict[str, object] = {}
        score_dict.update(dict(pred_message=pred_binary))
        score_dict.update(dict(sampled_positions=""))  # not tracked
        if return_bit_match:
            score_dict.update(dict(bit_acc=bit_acc))
            score_dict.update(dict(bit_match=bit_match))

        # Use t_total approximation: sum of minima across positions
        t_total = 0
        for value in cl_total.values():
            if sum(value) > 0:
                t_total += min(value)

        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=token_count))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=t_total))
        if return_green_fraction:
            denom = token_count if token_count > 0 else 1
            score_dict.update(
                dict(green_fraction=float(t_total) / float(denom))
            )
        if return_z_score:
            # For StealthInk, smaller z_total (fewer hits in the true red list)
            # means \"more watermarked\". The evaluation pipeline assumes that
            # larger scores correspond to the positive (watermarked) class, so
            # we store the *negated* z-score here to align with that convention.
            z_score = float(-z_total)
            score_dict.update(dict(z_score=z_score))
        if return_p_value:
            score_dict.update(dict(p_value=float(p_value)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask))
        if return_z_at_T:
            # Compute prefix-wise detection statistics when requested. For
            # StealthInk we expose:
            #   - z_score_at_T: prefix-averaged detection statistic
            #   - bit_acc_at_T: prefix-wise bit accuracy curve (if gold message).
            z_seq: List[float] = []
            bit_acc_seq: List[float] = []

            # Initialize a prefix colorlist count table.
            cl_prefix: Dict[int, List[int]] = {
                pos: [0 for _ in range(base)]
                for pos in range(self.converted_msg_length)
            }
            R = self.R if self.R > 0 else (1.0 / float(base))

            gold = message if isinstance(message, str) and message != "" else ""

            for idx2, (bit_pos2, hit_idx2) in enumerate(
                zip(bit_positions, hit_indices), start=ctx_w
            ):
                if 0 <= hit_idx2 < base:
                    cl_prefix[bit_pos2][hit_idx2] += 1

                # Preserve the original PRF-seeded Python RNG side effects so
                # that msg candidates (for positions with T=0) are generated
                # consistently with the unoptimized implementation.
                self._seed_rng(ids[idx2 - ctx_w : idx2])

                # Detection statistic at this prefix
                try:
                    p_T, msg_cands_T, z_total_T = _compute_norm_p_val(
                        cl_prefix, R
                    )
                    z_T = float(-z_total_T)
                except Exception:
                    msg_cands_T = None
                    z_T = float("nan")
                z_seq.append(z_T)

                # Prefix-wise bit accuracy when gold message is available
                if gold and msg_cands_T is not None:
                    digits_T: List[int] = []
                    for cand_list in msg_cands_T:
                        if not cand_list:
                            digits_T.append(0)
                        else:
                            digits_T.append(int(cand_list[0]))

                    # Convert base-b digits to binary string
                    decimal_T = 0
                    for d in digits_T:
                        decimal_T = decimal_T * base + int(d)
                    decimal_T = min(decimal_T, 2 ** self.message_length - 1)
                    pred_binary_T = format(decimal_T, f"0{self.message_length}b")

                    if len(pred_binary_T) == len(gold) and len(gold) > 0:
                        match_T = sum(
                            1 for g, p in zip(gold, pred_binary_T) if g == p
                        )
                        bit_acc_T = match_T / len(gold)
                    else:
                        bit_acc_T = float("nan")
                else:
                    bit_acc_T = float("nan")
                bit_acc_seq.append(float(bit_acc_T))

            # Anchor the final prefix values to the global metrics when they
            # are finite, so that curves end at the scalar statistics.
            try:
                final_z = float(score_dict.get("z_score", float("nan")))
            except Exception:
                final_z = float("nan")
            if z_seq and np.isfinite(final_z):
                z_seq[-1] = final_z

            try:
                final_bit = float(score_dict.get("bit_acc", float("nan")))
            except Exception:
                final_bit = float("nan")
            if bit_acc_seq and np.isfinite(final_bit):
                bit_acc_seq[-1] = final_bit

            score_dict.update(
                dict(
                    z_score_at_T=torch.as_tensor(
                        z_seq, dtype=torch.float32
                    ),
                    bit_acc_at_T=torch.as_tensor(
                        bit_acc_seq, dtype=torch.float32
                    ),
                )
            )

        return score_dict

    # --------------------------------------------------------------
    # Public detect() API
    # --------------------------------------------------------------

    def detect(
        self,
        text: str = None,
        tokenized_text: Tensor = None,
        window_size: str = None,
        window_stride: int = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        convert_to_float: bool = False,
        **kwargs,
    ) -> Dict[str, object]:
        """
        Scores a given string of text and returns a dictionary of results.
        Windowed variants are currently not implemented for StealthInk.
        """
        assert (text is not None) ^ (
            tokenized_text is not None
        ), "Must pass either the raw or tokenized string"

        if return_prediction:
            kwargs["return_p_value"] = True

        # Apply optional normalizers
        if text is not None:
            for normalizer in self.normalizers:
                text = normalizer(text)

        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "StealthInk detection on raw string "
                "requires an instance of the tokenizer "
                "used at generation time."
            )
            tokenized_text = self.tokenizer(
                text, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0].to(self.device)
            if (
                hasattr(self.tokenizer, "bos_token_id")
                and tokenized_text[0] == self.tokenizer.bos_token_id
            ):
                tokenized_text = tokenized_text[1:]
        else:
            if (
                self.tokenizer is not None
                and hasattr(self.tokenizer, "bos_token_id")
                and tokenized_text[0] == self.tokenizer.bos_token_id
            ):
                tokenized_text = tokenized_text[1:]

        output_dict: Dict[str, object] = {}

        if window_size is not None:
            # Windowed detection not currently implemented
            score_dict = self.dummy_detect(
                return_prediction=False,
                return_z_score=True,
                return_p_value=True,
            )
        else:
            score_dict = self._score_sequence(tokenized_text, **kwargs)

        if return_scores:
            output_dict.update(score_dict)

        if return_prediction:
            z_threshold = (
                z_threshold if z_threshold is not None else self.z_threshold
            )
            assert (
                z_threshold is not None
            ), "Need a threshold in order to decide outcome of detection test"
            z = score_dict.get("z_score", 0.0)
            try:
                z_val = float(z)
            except Exception:
                z_val = 0.0
            output_dict["prediction"] = z_val > float(z_threshold)
            if output_dict["prediction"] and "p_value" in score_dict:
                output_dict["confidence"] = 1 - float(
                    score_dict.get("p_value", 0.5)
                )

        if convert_to_float:
            for key, value in list(output_dict.items()):
                if isinstance(value, int):
                    output_dict[key] = float(value)

        return output_dict
