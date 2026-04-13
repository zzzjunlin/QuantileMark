# coding=utf-8
# zero-bit unbiased watermark, including gamma/delta reweighting strategies

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import LogitsProcessor

from alternative_prf_schemes import seeding_scheme_lookup
from normalizers import normalization_strategy_lookup

try:
    from tokenizers import Tokenizer
except Exception:  # type: ignore
    Tokenizer = object  # fallback type for static checkers


##########################################################################
# Generation-side core reweighting (Delta / Gamma strategies)
##########################################################################


class WatermarkStrategy:
    """Base class for Unbiased watermark strategies."""

    def from_random(
        self,
        rng: torch.Generator | Sequence[torch.Generator],
        vocab_size: int,
    ) -> Tensor:
        raise NotImplementedError

    def reweight_logits(self, code: Tensor, p_logits: Tensor) -> Tensor:
        raise NotImplementedError


class DeltaStrategy(WatermarkStrategy):
    """
    Delta-style strategy: draws a single quantile u in [0, 1] and
    collapses the distribution to a delta at the CDF-crossing index.
    """

    def from_random(
        self,
        rng: torch.Generator | Sequence[torch.Generator],
        vocab_size: int,
    ) -> Tensor:
        if isinstance(rng, list):
            batch_size = len(rng)
            u = torch.stack(
                [
                    torch.rand((), generator=rng[i], device=rng[i].device)
                    for i in range(batch_size)
                ]
            )
        else:
            u = torch.rand((), generator=rng, device=rng.device)
        return u

    def reweight_logits(self, u: Tensor, p_logits: Tensor) -> Tensor:
        """
        Reweight logits using quantile u.
        p_logits: [B, V] or [V]
        u:       [] or [B]
        """
        if p_logits.dim() == 1:
            p_logits = p_logits.unsqueeze(0)
        cumsum = torch.cumsum(F.softmax(p_logits, dim=-1), dim=-1)
        index = torch.searchsorted(cumsum, u[..., None], right=True)
        index = torch.clamp(index, 0, p_logits.shape[-1] - 1)
        vocab_range = torch.arange(p_logits.shape[-1], device=p_logits.device)
        modified_logits = torch.where(
            vocab_range == index,
            torch.zeros_like(p_logits),
            torch.full_like(p_logits, float("-inf")),
        )
        return modified_logits


class GammaStrategy(WatermarkStrategy):
    """
    Gamma-style strategy: samples a vocab permutation and applies a
    piecewise-linear transform to the CDF in that permutation order.
    """

    def from_random(
        self,
        rng: torch.Generator | Sequence[torch.Generator],
        vocab_size: int,
    ) -> Tensor:
        if isinstance(rng, list):
            batch_size = len(rng)
            shuffle = torch.stack(
                [
                    torch.randperm(
                        vocab_size, generator=rng[i], device=rng[i].device
                    )
                    for i in range(batch_size)
                ]
            )
        else:
            shuffle = torch.randperm(vocab_size, generator=rng, device=rng.device)
        return shuffle

    def reweight_logits(
        self,
        shuffle: Tensor,
        p_logits: Tensor,
        alpha: float = 0.5,
    ) -> Tensor:
        """
        Reweight logits using the shuffle and alpha.
        p_logits: [B, V] or [V]
        shuffle: [B, V] or [V]
        """
        if p_logits.dim() == 1:
            p_logits = p_logits.unsqueeze(0)
        if shuffle.dim() == 1:
            shuffle = shuffle.unsqueeze(0)

        unshuffle = torch.argsort(shuffle, dim=-1)

        s_p_logits = torch.gather(p_logits, -1, shuffle)
        s_log_cumsum = torch.logcumsumexp(s_p_logits, dim=-1)

        # normalize the log_cumsum to force the last element to be 0
        s_log_cumsum = s_log_cumsum - s_log_cumsum[..., -1:]
        s_cumsum = torch.exp(s_log_cumsum)
        s_p = F.softmax(s_p_logits, dim=-1)

        boundary_1 = torch.argmax((s_cumsum > alpha).to(torch.int), dim=-1, keepdim=True)
        p_boundary_1 = torch.gather(s_p, -1, boundary_1)
        portion_in_right_1 = (torch.gather(s_cumsum, -1, boundary_1) - alpha) / p_boundary_1
        portion_in_right_1 = torch.clamp(portion_in_right_1, 0, 1)
        s_all_portion_in_right_1 = (s_cumsum > alpha).type_as(p_logits)
        s_all_portion_in_right_1.scatter_(-1, boundary_1, portion_in_right_1)

        boundary_2 = torch.argmax(
            (s_cumsum > (1 - alpha)).to(torch.int), dim=-1, keepdim=True
        )
        p_boundary_2 = torch.gather(s_p, -1, boundary_2)
        portion_in_right_2 = (
            torch.gather(s_cumsum, -1, boundary_2) - (1 - alpha)
        ) / p_boundary_2
        portion_in_right_2 = torch.clamp(portion_in_right_2, 0, 1)
        s_all_portion_in_right_2 = (s_cumsum > (1 - alpha)).type_as(p_logits)
        s_all_portion_in_right_2.scatter_(-1, boundary_2, portion_in_right_2)

        s_all_portion_in_right = s_all_portion_in_right_2 / 2 + s_all_portion_in_right_1 / 2
        s_shift_logits = torch.log(s_all_portion_in_right)
        shift_logits = torch.gather(s_shift_logits, -1, unshuffle)

        return p_logits + shift_logits


class UnbiasedCore:
    """
    Shared generation/detection core for Unbiased watermarking.

    Handles:
    - context code extraction
    - deterministic seed generation from context + private key
    - mapping seeds to strategy codes and reweighting logits
    """

    def __init__(
        self,
        seeding_scheme: str = "simple_1",
        wm_type: str = "gamma",
        prefix_length: int = 0,
        ignore_history_generation: bool = False,
        ignore_history_detection: bool = False,
    ):
        (
            _prf_type,
            _context_width,
            _self_salt,
            hash_key_int,
        ) = seeding_scheme_lookup(seeding_scheme)
        # Use the hash_key from the seeding scheme as a deterministic private key.
        self._hash_key_bytes = str(int(hash_key_int)).encode("utf-8")

        self.prefix_length = max(int(prefix_length), 0)
        self.ignore_history_generation = bool(ignore_history_generation)
        self.ignore_history_detection = bool(ignore_history_detection)

        self.strategy: WatermarkStrategy
        if wm_type == "delta":
            self.strategy = DeltaStrategy()
        else:
            self.strategy = GammaStrategy()

        self.cc_history: set[bytes] = set()

    # ------------------------------------------------------------------
    # History and context hashing
    # ------------------------------------------------------------------

    def reset_history(self) -> None:
        self.cc_history.clear()

    def _extract_context_code(self, context: Tensor) -> bytes:
        """
        Extract context code from the given context tokens.
        If prefix_length == 0, use full context; otherwise use last prefix_length tokens.
        """
        if self.prefix_length == 0:
            return context.detach().cpu().numpy().tobytes()
        else:
            return context[..., -self.prefix_length :].detach().cpu().numpy().tobytes()

    def _get_rng_seed(self, context_code: bytes, mode: str) -> int:
        """
        Get the random seed from the given context code and private key.
        mode: 'generation' | 'detection'
        """
        if mode == "generation":
            ignore_history = self.ignore_history_generation
        else:
            ignore_history = self.ignore_history_detection

        if not ignore_history:
            self.cc_history.add(context_code)

        m = hashlib.sha256()
        m.update(context_code)
        m.update(self._hash_key_bytes)
        full_hash = m.digest()
        seed = int.from_bytes(full_hash, "big") % (2**32 - 1)
        return seed

    def _get_mask_and_seeds(self, input_ids: Tensor, mode: str) -> Tuple[Tensor, List[int]]:
        """
        input_ids: [B, T]
        Returns:
            mask:  [B] bool, True if context has been seen before
            seeds: list[int] size B
        """
        batch_size = input_ids.size(0)
        context_codes: List[bytes] = [
            self._extract_context_code(input_ids[i]) for i in range(batch_size)
        ]
        mask_list: List[bool] = []
        seeds: List[int] = []
        for c in context_codes:
            mask_list.append(c in self.cc_history)
            seeds.append(self._get_rng_seed(c, mode))
        mask = torch.tensor(mask_list, device=input_ids.device, dtype=torch.bool)
        return mask, seeds

    def apply_watermark(
        self,
        input_ids: Tensor,
        scores: Tensor,
        mode: str = "generation",
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply the Unbiased watermark to scores.

        Args:
            input_ids: [B, T] prefix tokens
            scores:    [B, V] logits for next token
            mode:      'generation' or 'detection'
        Returns:
            mask:  [B] bool tensor indicating repeated contexts
            logits: [B, V] reweighted logits
        """
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)

        mask, seeds = self._get_mask_and_seeds(input_ids, mode)
        rngs: List[torch.Generator] = [
            torch.Generator(device=scores.device).manual_seed(seed) for seed in seeds
        ]

        code = self.strategy.from_random(rngs, scores.size(-1))
        if isinstance(self.strategy, GammaStrategy):
            reweighted_scores = self.strategy.reweight_logits(code, scores)
        else:
            reweighted_scores = self.strategy.reweight_logits(code, scores)

        return mask, reweighted_scores


##########################################################################
# Generation-time LogitsProcessor
##########################################################################


class UnbiasedWatermarkLogitsProcessor(LogitsProcessor):
    """
    Generation-time logits processor implementing the Unbiased watermark.

    This class is wired to be API-compatible with utils/generation.generate():
    - exposes set_message() and converted_message
    - exposes flush_position() and _get_and_clear_stored_spike_ents()
    - tracks is_r / position_increment for compatibility (unused here).
    """

    def __init__(
        self,
        vocab: List[int],
        seeding_scheme: str = "simple_1",
        wm_type: str = "gamma",
        prefix_length: int = 0,
        ignore_history_generation: bool = False,
    ):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.core = UnbiasedCore(
            seeding_scheme=seeding_scheme,
            wm_type=wm_type,
            prefix_length=prefix_length,
            ignore_history_generation=ignore_history_generation,
            ignore_history_detection=False,
        )
        self.prefix_length = max(int(prefix_length), 0)
        self.ignore_history_generation = bool(ignore_history_generation)

        # Compatibility fields for utils/generation.py
        self.converted_message: str = ""
        self.embedded_message: Optional[List[int]] = None
        self.position_increment: int = 0
        self.spike_entropies = None
        self.is_r: bool = False

    # ------------------------------------------------------------------
    # Message and compatibility helpers
    # ------------------------------------------------------------------

    def set_message(self, binary_msg: str):
        """
        Unbiased watermark is zero-bit; we store the message string only
        for logging compatibility and ignore it in the algorithm.
        """
        self.converted_message = binary_msg or ""

    def flush_position(self) -> List[str]:
        """
        Unbiased watermark doesn't track per-position codes for analysis.
        Return a dummy list (length 1) to satisfy schema.
        """
        return [""]

    def _get_and_clear_stored_spike_ents(self):
        """
        Compatibility stub: Unbiased watermark does not track spike entropies.
        """
        return []

    def reset_history(self) -> None:
        """Reset context-code history (useful between separate generations)."""
        self.core.reset_history()

    # ------------------------------------------------------------------
    # Core logits processing
    # ------------------------------------------------------------------

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        """
        Apply Unbiased reweighting to the logits.

        Args:
            input_ids: [B, T] integer ids for current sequences.
            scores: [B, V] float logits for next token.
        """
        # If the current prefix is shorter than the required prefix length,
        # skip watermarking and leave logits unchanged.
        if input_ids.shape[1] < max(self.prefix_length, 1):
            return scores

        mask, reweighted_scores = self.core.apply_watermark(
            input_ids=input_ids,
            scores=scores,
            mode="generation",
        )

        if self.ignore_history_generation:
            return reweighted_scores
        else:
            return torch.where(mask[:, None], scores, reweighted_scores)


##########################################################################
# Detection-side utilities
##########################################################################


class UnbiasedWatermarkDetector:
    """
    Detector for Unbiased watermarks.

    This detector reconstructs the same reweighting as the generator
    (given the same seeding_scheme and wm_type), and computes a robust
    log-likelihood-ratio statistic over the continuation tokens.

    The public `detect()` API is compatible with utils.evaluation.compute_z_score.
    """

    def __init__(
        self,
        vocab: List[int],
        seeding_scheme: str = "simple_1",
        wm_type: str = "gamma",
        prefix_length: int = 0,
        n_grid: int = 64,
        device: torch.device | str = "cuda",
        model=None,
        tokenizer: Optional[Tokenizer] = None,
        normalizers: Optional[List[str]] = None,
        z_threshold: float = 0.0,
        ignore_history_detection: bool = False,
    ):
        assert model is not None, "UnbiasedWatermarkDetector requires a model."
        assert tokenizer is not None, "UnbiasedWatermarkDetector requires a tokenizer."

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.core = UnbiasedCore(
            seeding_scheme=seeding_scheme,
            wm_type=wm_type,
            prefix_length=prefix_length,
            ignore_history_generation=False,
            ignore_history_detection=ignore_history_detection,
        )
        self.prefix_length = max(int(prefix_length), 0)
        self.n_grid = max(int(n_grid), 1)
        self.z_threshold = float(z_threshold)
        self.ignore_history_detection = bool(ignore_history_detection)

        self.normalizers = []
        if normalizers is None:
            normalizers = []
        for strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(strategy))

        # For compatibility with evaluation utilities
        self.position_increment: int = 0

    # ------------------------------------------------------------------
    # Robust LLR helpers (adapted from UnbiasedWatermark implementation)
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_minus(q_logits: Tensor, p_logits: Tensor) -> Tensor:
        """Safe minus operation to avoid numerical instability."""
        llr = q_logits - p_logits
        llr.nan_to_num_(nan=0.0)
        return llr

    @staticmethod
    def _from_grid(
        dist_ps: Sequence[float],
        dist_qs: Sequence[float],
    ) -> List[Tuple[float, float]]:
        """Generate batch query from grid search."""
        dist_ps_arr = np.asarray(dist_ps, dtype=float)
        dist_qs_arr = np.asarray(dist_qs, dtype=float)
        assert dist_ps_arr.ndim == 1
        assert dist_qs_arr.ndim == 1
        assert np.all(dist_ps_arr >= 0) and np.all(dist_qs_arr >= 0)
        with np.errstate(divide="ignore"):
            dist_p_logs = np.log(dist_ps_arr)
            dist_q_logs = np.log(dist_qs_arr)
        dist_p_logs.sort()
        dist_q_logs.sort()
        batch_query = [
            (float(d_p_l), float(d_q_l))
            for d_p_l in dist_p_logs
            for d_q_l in dist_q_logs
        ]
        return batch_query

    def _get_max_llr(
        self,
        p_logits: Tensor,
        q_logits: Tensor,
        batch_query: List[Tuple[float, float]],
    ) -> Tensor:
        """
        Compute the maximum achievable LLR under perturbations parameterized
        by (dist_p_log, dist_q_log) in batch_query.

        p_logits, q_logits: [B, S, V]
        Returns:
            max_llr: [B, S, Q] where Q=len(batch_query)
        """
        llr = self._safe_minus(q_logits, p_logits)
        try:
            sort_index = torch.argsort(llr, dim=-1, descending=True)
        except torch.cuda.OutOfMemoryError:
            sort_index = torch.argsort(llr.cpu(), dim=-1, descending=True).to(llr.device)
        del llr

        p_logits = p_logits.gather(-1, sort_index)
        q_logits = q_logits.gather(-1, sort_index)
        del sort_index

        llr = self._safe_minus(q_logits, p_logits)

        sum_q_logits = torch.logcumsumexp(q_logits, dim=-1)
        sum_p_logits = torch.logcumsumexp(p_logits, dim=-1)
        del q_logits
        del p_logits

        max_llrs: List[Tensor] = []
        for dist_p_log, dist_q_log in batch_query:
            # shape = (..., vocab_size)
            modified_q_logits = torch.where(
                sum_q_logits <= dist_q_log,
                torch.tensor(
                    float("-inf"),
                    device=sum_q_logits.device,
                    dtype=sum_q_logits.dtype,
                ),
                sum_q_logits
                + torch.log(-torch.expm1(dist_q_log - sum_q_logits)),
            )
            modified_p_logits = torch.logaddexp(
                sum_p_logits,
                torch.tensor(
                    dist_p_log,
                    device=sum_p_logits.device,
                    dtype=sum_p_logits.dtype,
                ),
            )

            modified_llr = self._safe_minus(modified_q_logits, modified_p_logits)
            del modified_p_logits
            del modified_q_logits

            # pad left modified_llr with -inf
            modified_llr = F.pad(modified_llr, (1, 0), value=float("-inf"))
            # get index by argmax
            cut_index = torch.where(
                torch.any(llr < modified_llr[..., :-1], dim=-1),
                torch.argmax(
                    (llr < modified_llr[..., :-1]).to(torch.int), dim=-1
                ),
                torch.tensor(
                    modified_llr.shape[-1] - 1, device=modified_llr.device
                ),
            )
            max_llrs.append(modified_llr.gather(-1, cut_index.unsqueeze(-1)))

        max_llr = torch.cat(max_llrs, dim=-1)
        del max_llrs
        return max_llr

    @torch.no_grad()
    def _score_llr(
        self,
        p_logits: Tensor,
        q_logits: Tensor,
        batch_query: List[Tuple[float, float]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Return (llr, max_llr, min_llr)
        Shapes:
            llr:     [B, S, V]
            max_llr: [B, S, Q]
            min_llr: [B, S, Q]
        """
        q_logits = F.log_softmax(q_logits, dim=-1)
        p_logits = F.log_softmax(p_logits, dim=-1)

        max_llr = self._get_max_llr(p_logits, q_logits, batch_query)
        min_llr = -self._get_max_llr(
            q_logits,
            p_logits,
            [(q, p) for p, q in batch_query],
        )
        trivial_pos = max_llr < min_llr
        max_llr = torch.where(
            trivial_pos,
            torch.tensor(0.0, device=max_llr.device),
            max_llr,
        )
        min_llr = torch.where(
            trivial_pos,
            torch.tensor(0.0, device=min_llr.device),
            min_llr,
        )

        llr = self._safe_minus(q_logits, p_logits)
        return llr, max_llr, min_llr

    @staticmethod
    def _value_transformation(value: float) -> float:
        """Map value to [0, 1] for visualization."""
        return value / (value + 1) if value >= 0 else 0.0

    # ------------------------------------------------------------------
    # Core per-text scoring
    # ------------------------------------------------------------------

    def _score_sequence(
        self,
        text: str,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
        return_bit_match: bool = True,
        **kwargs,
    ) -> Dict[str, object]:
        """
        Score a single text string and return a HF-style score_dict.
        """
        # Apply optional normalizers
        for normalizer in self.normalizers:
            text = normalizer(text)

        # Step 1: construct grid for robust LLR
        n = self.n_grid
        dist = [float(i) / n for i in range(n + 1)]
        batch_query = self._from_grid([0.0], dist)

        # Step 2: get original and reweighted logits via teacher forcing
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        full_input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        if full_input_ids.shape[1] <= max(self.prefix_length + 1, 1):
            # Not enough tokens to score anything
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
                return_z_score_max=False,
            )

        # Ignore logits for the final token (standard teacher forcing)
        input_ids = full_input_ids[..., :-1]
        if attention_mask is not None:
            attention_mask_in = attention_mask[..., :-1]
        else:
            attention_mask_in = None

        # Ignore prefix tokens for scoring (algorithm-specific prefix_length)
        labels = full_input_ids[..., self.prefix_length :]

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask_in,
                use_cache=False,
            )
            logits = outputs.logits  # [B, L, V]

        old_logits = logits.clone()
        new_logits = logits.clone()

        # Reset history for this sequence
        self.core.reset_history()

        B, L, V = logits.shape
        start_idx = max(self.prefix_length - 1, 0)
        end_idx = L - 1  # last index with next-token logits

        for i in range(start_idx, end_idx):
            pre_input_ids = input_ids[:, : i + 1]
            t = logits[:, i]  # [B, V]
            mask, reweighted_scores = self.core.apply_watermark(
                pre_input_ids,
                t,
                mode="detection",
            )

            old_logits[:, i] = t
            if self.ignore_history_detection:
                new_logits[:, i] = reweighted_scores
            else:
                new_logits[:, i] = torch.where(mask[:, None], t, reweighted_scores)

        old_logits = old_logits[:, start_idx:]
        new_logits = new_logits[:, start_idx:]

        # Step 3: robust LLR scoring
        llr, max_llr, min_llr = self._score_llr(old_logits, new_logits, batch_query)

        # Step 4: extract & clamp llr for token ids
        # Align lengths between llr positions and labels
        B2, S, _ = llr.shape
        labels_slice = labels[..., :S]
        if labels_slice.shape[1] == 0:
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
                return_z_score_max=False,
            )

        unclipped_scores = torch.gather(
            llr,
            -1,
            labels_slice.unsqueeze(-1),
        ).squeeze(-1)
        # unclipped_scores: [B, S]
        scores = torch.clamp(
            unclipped_scores.unsqueeze(-1),
            min_llr,
            max_llr,
        )
        # scores: [B, S, Q]

        scores_np = scores[0].detach().cpu().numpy()
        labels_np = labels_slice[0].detach().cpu().numpy()

        # Step 5: choose best index in grid
        sum_scores = np.sum(scores_np, axis=0)  # [Q]
        best_index = int(np.argmax(sum_scores))
        final_score = float(sum_scores[best_index])

        # Step 6: per-token highlights (for completeness; not used by metrics)
        best_scores = scores_np[:, best_index]  # [S]
        highlight_values: List[Optional[float]] = [None] * int(full_input_ids.shape[1])
        for i, score in enumerate(best_scores):
            pos = self.prefix_length + i
            if 0 <= pos < len(highlight_values):
                highlight_values[pos] = self._value_transformation(float(score))

        # Theorem-style bound: p <= A * exp(-t) where A ~ n_grid
        p_val = float(self.n_grid * np.exp(-final_score))
        # Use final_score as a z-like test statistic where larger => more watermarked
        z_score = float(final_score) if return_z_score else 0.0

        # ------------------------------------------------------------------
        # Assemble HF-style score_dict
        # ------------------------------------------------------------------
        score_dict: Dict[str, object] = {}
        score_dict.update(dict(pred_message=""))  # zero-bit watermark

        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=int(len(labels_np))))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=float("nan")))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=float("nan")))

        if return_z_score:
            score_dict.update(dict(z_score=z_score))
        # For completeness: store the aggregate log-likelihood ratio
        score_dict.update(dict(raw_log_likelihood=final_score))

        if return_p_value:
            score_dict.update(dict(p_value=p_val))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=[]))
        if return_z_at_T:
            score_dict.update(dict(z_score_at_T=torch.tensor([])))

        if return_bit_match:
            score_dict.update(dict(bit_acc=float("nan")))
            score_dict.update(dict(bit_match=None))

        return score_dict

    # ------------------------------------------------------------------
    # Dummy detection (schema-only)
    # ------------------------------------------------------------------

    def dummy_detect(
        self,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float | None = None,
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
    ) -> Dict[str, object]:
        """
        Return a dummy detection result with all fields set to NaN / empty.
        Used when detection cannot be performed (e.g., text too short).
        """
        score_dict: Dict[str, object] = {}
        score_dict.update(dict(pred_message=""))
        score_dict.update(dict(sampled_positions=""))
        score_dict.update(dict(position_acc=float("nan")))
        score_dict.update(dict(bit_match=None))
        score_dict.update(dict(bit_acc=float("nan")))
        score_dict.update(dict(decoding_time=float("nan")))
        score_dict.update(dict(raw_log_likelihood=float("nan")))

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

        output_dict: Dict[str, object] = {}
        if return_scores:
            output_dict.update(score_dict)
        if return_prediction:
            z_threshold = z_threshold if z_threshold is not None else self.z_threshold
            output_dict["prediction"] = False

        return output_dict

    # ------------------------------------------------------------------
    # Public detect() API
    # ------------------------------------------------------------------

    def detect(
        self,
        text: str = None,
        tokenized_text: Tensor = None,
        window_size: str = None,
        window_stride: int = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float | None = None,
        convert_to_float: bool = False,
        **kwargs,
    ) -> Dict[str, object]:
        """
        Scores a given string of text and returns a dictionary of results.
        Windowed variants are not implemented; window_size/window_stride
        are accepted for API compatibility but ignored.
        """
        assert (text is not None) ^ (
            tokenized_text is not None
        ), "Must pass either the raw or tokenized string"

        if return_prediction:
            kwargs["return_p_value"] = True

        # For this detector we always work from raw text; if only token ids
        # are provided, decode them once.
        if text is None and tokenized_text is not None:
            text = self.tokenizer.decode(
                tokenized_text.tolist(), skip_special_tokens=True
            )
        assert text is not None

        output_dict: Dict[str, object] = {}

        if window_size is not None:
            # Windowed detection not implemented; return dummy.
            score_dict = self.dummy_detect(
                return_prediction=False,
                return_z_score=True,
                return_p_value=True,
            )
        else:
            score_dict = self._score_sequence(text, **kwargs)

        if return_scores:
            output_dict.update(score_dict)

        if return_prediction:
            z_threshold = (
                float(z_threshold)
                if z_threshold is not None
                else float(self.z_threshold)
            )
            z = score_dict.get("z_score", 0.0)
            try:
                z_val = float(z)
            except Exception:
                z_val = 0.0
            output_dict["prediction"] = z_val > z_threshold
            if output_dict["prediction"] and "p_value" in score_dict:
                try:
                    output_dict["confidence"] = 1 - float(
                        score_dict.get("p_value", 0.5)
                    )
                except Exception:
                    pass

        if convert_to_float:
            for key, value in list(output_dict.items()):
                if isinstance(value, int):
                    output_dict[key] = float(value)

        return output_dict


