# Copyright 2026 Authors of "QuantileMark: A Message-Symmetric Multi-bit Watermark for LLMs"
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantile watermark processor and detector.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import torch
from transformers import LogitsProcessor

from mb_watermark_processor import WatermarkBase

_MASK_U64 = (1 << 64) - 1
_DEFAULT_MAPPING_KEY = "quantile-map-key-v1"
_VALID_MAPPING_SCHEMES = frozenset({"identity", "cyclic", "permute"})


def _mask_u64(value: int) -> int:
    return int(value) & _MASK_U64


def _normalize_mapping_scheme(mapping_scheme: str, *, default: str) -> str:
    scheme = str(mapping_scheme or default).strip().lower()
    if scheme not in _VALID_MAPPING_SCHEMES:
        valid = ", ".join(sorted(_VALID_MAPPING_SCHEMES))
        raise ValueError(f"Unknown mapping_scheme={mapping_scheme!r}. Expected one of: {valid}.")
    return scheme


def _normalize_epsilon(epsilon: float) -> float:
    eps = float(epsilon)
    if not (0.0 <= eps < 0.5):
        raise ValueError(f"epsilon must satisfy 0 <= epsilon < 0.5, got {epsilon!r}.")
    return eps


def _normalize_binary_message(binary_msg: str, message_length: int) -> str:
    msg = str(binary_msg or "").strip()
    if msg.startswith("0b"):
        msg = msg[2:]
    msg = msg.replace(" ", "").replace("_", "")
    if any(ch not in "01" for ch in msg):
        raise ValueError("binary message must contain only 0/1 characters.")
    if len(msg) > message_length:
        raise ValueError(
            f"binary message length ({len(msg)}) exceeds message_length ({message_length})."
        )
    return msg.zfill(message_length)

# Generator and detector both use this helper so they score the same
# candidate set after top-k / top-p truncation.
def _apply_top_k_top_p_filtering(
    logits: torch.Tensor,
    *,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    filtered_logits = logits.clone()

    if top_k > 0:
        k = min(top_k, filtered_logits.size(-1))
        indices_to_remove = filtered_logits < torch.topk(filtered_logits, k)[0][..., -1, None]
        filtered_logits[indices_to_remove] = -float("Inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        filtered_logits[indices_to_remove] = -float("Inf")

    return filtered_logits


class QuantileWatermarkLogitsProcessor(LogitsProcessor):
    """
    Quantile-based watermark logits processor.
    Divides the probability distribution into buckets and samples from the target bucket.
    """

    def __init__(
        self,
        vocab: list[int],
        gamma: float = 0.5,
        seeding_scheme: str = "simple_1",
        chunk_capacity: int = 3,
        message_length: int = 24,
        top_p: float = 1.0,
        top_k: int = 0,
        device: str = "cuda",
        mapping_scheme: str = "permute",  # how message symbols are mapped onto quantile intervals
        mapping_key: str | None = None,
        epsilon: float = 0.0,  # optional shrink of interval boundaries during generation
        tokenizer=None,
        verbose: bool = False,
        **kwargs
    ):
        self.base_processor = WatermarkBase(
            vocab=vocab,
            gamma=gamma,
            seeding_scheme=seeding_scheme,
            device=device,
            **kwargs
        )

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.chunk_capacity = int(chunk_capacity)
        if self.chunk_capacity <= 0:
            raise ValueError(f"chunk_capacity must be > 0, got {chunk_capacity!r}.")
        self.num_buckets = 2 ** self.chunk_capacity
        self.message_length = int(message_length)
        if self.message_length <= 0:
            raise ValueError(f"message_length must be > 0, got {message_length!r}.")
        max_payload = (1 << self.message_length) - 1
        self.converted_msg_length = 1
        while max_payload >= self.num_buckets:
            max_payload //= self.num_buckets
            self.converted_msg_length += 1

        self.top_p = top_p
        self.top_k = top_k
        self.device = device
        self.verbose = bool(verbose)

        self.embedded_message = None
        self.converted_message = None
        self.seen_seeds = {}
        self.is_r = False
        self.n_gram_len = self.base_processor.context_width
        self.tokenizer = tokenizer

        self.mapping_scheme = _normalize_mapping_scheme(mapping_scheme, default="permute")
        self.mapping_key = mapping_key or _DEFAULT_MAPPING_KEY
        self.epsilon = _normalize_epsilon(epsilon)
        self._permute_rs = np.random.RandomState(0)
        self._permute_cache: dict[int, np.ndarray] = {}
        self._permute_cache_max = 16384

        self.position_increment = 0
        self.spike_entropies = None

    def _numberToBase(self, n, b):
        """Convert decimal number to base b."""
        if n == 0:
            return str(0)
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return "".join(map(str, digits[::-1]))

    def _hash_to_int(self, seed_val: int, bit_pos: int, label: bytes = b"map") -> int:
        """Stable hash to int from the PRF key and bit position. Independent of torch RNG state."""
        h = hashlib.sha256()
        x = _mask_u64(seed_val)
        h.update(x.to_bytes(8, "little", signed=False))
        if self.mapping_key:
            h.update(self.mapping_key.encode("utf-8"))
        h.update(label)
        h.update(int(bit_pos).to_bytes(2, "little", signed=False))
        return int.from_bytes(h.digest()[:8], "little")

    def _map_interval_index(self, seed_val: int, bit_pos: int, s: int) -> int:
        if self.mapping_scheme == "identity":
            return s
        h = self._hash_to_int(seed_val, bit_pos)
        if self.mapping_scheme == "cyclic":
            return int((s + (h % self.num_buckets)) % self.num_buckets)
        if self.mapping_scheme == "permute":
            seed32 = int(h) & ((1 << 32) - 1)
            perm = self._permute_cache.get(seed32)
            if perm is None:
                if len(self._permute_cache) >= self._permute_cache_max:
                    self._permute_cache.clear()
                self._permute_rs.seed(seed32)
                perm = self._permute_rs.permutation(self.num_buckets)
                self._permute_cache[seed32] = perm
            return int(perm[s])
        return s

    def _interval_bounds(self, idx: int) -> tuple[float, float]:
        w = 1.0 / self.num_buckets
        start = idx * w
        end = (idx + 1) * w
        if self.epsilon > 0.0:
            scale = (1.0 - 2.0 * self.epsilon)
            start = self.epsilon + start * scale
            end = self.epsilon + end * scale
        return float(start), float(end)

    def set_message(self, binary_msg: str):
        """Set the message to embed."""
        normalized_msg = _normalize_binary_message(binary_msg, self.message_length)
        self.seen_seeds.clear()
        decimal = int(normalized_msg, 2)
        converted_msg = self._numberToBase(decimal, self.num_buckets)
        converted_msg = "0" * (self.converted_msg_length - len(converted_msg)) + converted_msg
        self.embedded_message = [int(c) for c in converted_msg]
        self.converted_message = converted_msg

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to embed watermark."""
        if self.embedded_message is None:
            raise RuntimeError(
                "QuantileWatermarkLogitsProcessor requires set_message() before generation."
            )
        device = scores.device
        batch_size = input_ids.shape[0]

        if self.base_processor.rng is None:
            self.base_processor.rng = torch.Generator(device=input_ids.device)

        new_logits = torch.full_like(scores, -float("Inf"))

        # For each step, seed from the local context, choose one message
        # position, and keep only the probability mass inside its target interval.
        for batch_idx in range(batch_size):
            if input_ids.shape[1] < self.n_gram_len:
                new_logits[batch_idx] = scores[batch_idx]
                continue

            seed = input_ids[batch_idx:batch_idx+1, -self.n_gram_len:]
            self.base_processor._seed_rng(seed)
            prf_key = int(self.base_processor.prf_key)
            seen = self.seen_seeds.setdefault(batch_idx, set())
            if prf_key in seen:
                new_logits[batch_idx] = scores[batch_idx]
                continue
            seen.add(prf_key)
            seed_key = prf_key
            bit_pos = int(seed_key % self.converted_msg_length)
            target_bucket = self.embedded_message[bit_pos]

            filtered_scores = _apply_top_k_top_p_filtering(
                scores[batch_idx:batch_idx+1],
                top_k=self.top_k,
                top_p=self.top_p,
            )

            candidate_indices = (filtered_scores > -float("Inf")).squeeze(0).nonzero().squeeze()
            if candidate_indices.numel() == 0:
                new_logits[batch_idx] = scores[batch_idx]
                continue
            if candidate_indices.dim() == 0:
                candidate_indices = candidate_indices.unsqueeze(0)

            candidate_scores = filtered_scores.squeeze(0)[candidate_indices]

            probs = torch.softmax(candidate_scores, dim=-1)
            probs_double = probs.to(torch.float64)
            sorted_probs, sorted_indices_relative = torch.sort(probs_double, descending=True)

            cdf = torch.cumsum(sorted_probs, dim=0)
            cdf_intervals_end = cdf
            cdf_intervals_start = cdf - sorted_probs

            mapped_idx = self._map_interval_index(seed_key, bit_pos, target_bucket)
            bucket_start, bucket_end = self._interval_bounds(mapped_idx)

            intersection_start = torch.max(
                cdf_intervals_start,
                torch.tensor(bucket_start, device=device, dtype=torch.float64)
            )
            intersection_end = torch.min(
                cdf_intervals_end,
                torch.tensor(bucket_end, device=device, dtype=torch.float64)
            )
            intersection_len = torch.clamp(intersection_end - intersection_start, min=0.0)

            new_probs_sorted = intersection_len * self.num_buckets
            new_candidate_logits = torch.log(new_probs_sorted.clamp_min(1e-40))
            sorted_indices_absolute = candidate_indices[sorted_indices_relative]
            new_logits[batch_idx].scatter_(0, sorted_indices_absolute, new_candidate_logits.to(scores.dtype))

        self.is_r = False

        return new_logits

    def flush_position(self):
        return [""]

    def _get_and_clear_stored_spike_ents(self):
        return []


class QuantileWatermarkDetector:
    """
    Detector for quantile-based watermarks.
    Requires model access to compute logits.
    """

    def __init__(
        self,
        vocab: list[int],
        gamma: float = 0.5,
        seeding_scheme: str = "simple_1",
        chunk_capacity: int = 3,
        message_length: int = 24,
        top_p: float = 1.0,
        top_k: int = 256,
        device: str = "cuda",
        model=None,
        tokenizer=None,
        mapping_scheme: str = "identity",  # must match the generation-side interval mapping
        mapping_key: str | None = None,
        temperature: float = 1.0,
        posterior_eps: float = 1e-6,  # posterior clamp before taking log-odds
        debug: bool = False,
        skip_ratio: float = 0.1,
        glrt_mode: str = "lpo",  # 'lpo' for average log-odds, 'strict' for per-token GLRT
        wrap_output_in_chat_template: bool = False,
        verbose: bool = False,
        **kwargs
    ):
        self.base_processor = WatermarkBase(
            vocab=vocab,
            gamma=gamma,
            seeding_scheme=seeding_scheme,
            device=device,
            **kwargs
        )

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.chunk_capacity = int(chunk_capacity)
        if self.chunk_capacity <= 0:
            raise ValueError(f"chunk_capacity must be > 0, got {chunk_capacity!r}.")
        self.num_buckets = 2 ** self.chunk_capacity
        self.message_length = int(message_length)
        if self.message_length <= 0:
            raise ValueError(f"message_length must be > 0, got {message_length!r}.")
        max_payload = (1 << self.message_length) - 1
        self.converted_msg_length = 1
        while max_payload >= self.num_buckets:
            max_payload //= self.num_buckets
            self.converted_msg_length += 1

        self.top_p = top_p
        self.top_k = top_k
        self.device = device
        self.temperature = float(temperature)
        class _TempWrapper(torch.nn.Module):
            def __init__(self, base_model, temp):
                super().__init__()
                self.base_model = base_model
                self.temperature = temp

            def forward(self, *args, **kwargs):
                outputs = self.base_model(*args, **kwargs)
                try:
                    logits = outputs.logits
                except AttributeError:
                    return outputs
                if self.temperature and self.temperature != 1.0:
                    logits = logits / self.temperature
                    outputs.logits = logits
                return outputs

        self.model = _TempWrapper(model, self.temperature)
        self.tokenizer = tokenizer
        self.n_gram_len = self.base_processor.context_width

        self.position_increment = 0

        self.mapping_scheme = _normalize_mapping_scheme(mapping_scheme, default="identity")
        self.mapping_key = mapping_key or _DEFAULT_MAPPING_KEY
        self._permute_rs = np.random.RandomState(0)
        self._permute_cache_np: dict[int, np.ndarray] = {}
        self._permute_cache_torch: dict[int, torch.Tensor] = {}
        self._permute_cache_max = 16384
        self.posterior_eps = float(posterior_eps)
        if not (0.0 < self.posterior_eps < 0.5):
            raise ValueError(
                f"posterior_eps must satisfy 0 < posterior_eps < 0.5, got {posterior_eps!r}."
            )
        self.skip_ratio = max(0.0, min(0.9, float(skip_ratio)))

        self.debug = bool(debug)
        self._debug_printed_input = False

        if glrt_mode not in ("heuristic", "lpo", "strict"):
            raise ValueError(f"Invalid glrt_mode: {glrt_mode}")
        self.glrt_mode = glrt_mode
        self.wrap_output_in_chat_template = bool(wrap_output_in_chat_template)

    def _int_to_digits(self, n: int, b: int, width: int) -> list[int]:
        if n == 0:
            out = [0]
        else:
            out = []
            while n:
                out.append(int(n % b))
                n //= b
            out = out[::-1]
        if len(out) < width:
            out = [0] * (width - len(out)) + out
        return out

    def _digits_to_int(self, digits: list[int], b: int) -> int:
        val = 0
        for d in digits:
            val = val * b + int(d)
        return val

    def _hash_to_int(self, seed_val: int, bit_pos: int, label: bytes = b"map") -> int:
        h = hashlib.sha256()
        x = _mask_u64(seed_val)
        h.update(x.to_bytes(8, "little", signed=False))
        if self.mapping_key:
            h.update(self.mapping_key.encode("utf-8"))
        h.update(label)
        h.update(int(bit_pos).to_bytes(2, "little", signed=False))
        return int.from_bytes(h.digest()[:8], "little")

    def _get_bucket_permutation_torch(self, h: int) -> torch.Tensor:
        """
        Deterministic bucket permutation for mapping_scheme='permute'.

        Preserves exact behavior of `np.random.RandomState(seed32).permutation(M)`
        while avoiding per-token RandomState construction. Returns a tensor on
        `self.device` suitable for indexing.
        """
        seed32 = int(h) & ((1 << 32) - 1)
        cached = self._permute_cache_torch.get(seed32)
        if cached is not None:
            return cached

        perm_np = self._permute_cache_np.get(seed32)
        if perm_np is None:
            if len(self._permute_cache_np) >= self._permute_cache_max:
                self._permute_cache_np.clear()
                self._permute_cache_torch.clear()
            self._permute_rs.seed(seed32)
            perm_np = self._permute_rs.permutation(self.num_buckets)
            self._permute_cache_np[seed32] = perm_np

        perm_t = torch.from_numpy(
            perm_np.astype(np.int64, copy=False)
        ).to(self.device)
        self._permute_cache_torch[seed32] = perm_t
        return perm_t

    def _map_interval_index(self, seed_val: int, bit_pos: int, s: int) -> int:
        if self.mapping_scheme == "identity":
            return s
        h = self._hash_to_int(seed_val, bit_pos)
        if self.mapping_scheme == "cyclic":
            return int((s + (h % self.num_buckets)) % self.num_buckets)
        if self.mapping_scheme == "permute":
            seed32 = int(h) & ((1 << 32) - 1)
            perm = self._permute_cache_np.get(seed32)
            if perm is None:
                if len(self._permute_cache_np) >= self._permute_cache_max:
                    self._permute_cache_np.clear()
                    self._permute_cache_torch.clear()
                self._permute_rs.seed(seed32)
                perm = self._permute_rs.permutation(self.num_buckets)
                self._permute_cache_np[seed32] = perm
            return int(perm[s])
        return s

    def _empty_prefix_metrics(self) -> tuple[torch.Tensor, torch.Tensor]:
        empty = torch.tensor([], dtype=torch.float32, device=self.device)
        return empty, empty

    def _compute_prefix_metrics(
        self,
        *,
        max_len: int,
        predicted_msg: list[int],
        gold_message: str | None,
        final_bit_acc: float,
        per_pos_token_post,
        per_pos_token_weight,
        per_pos_token_step,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_len <= 0:
            return self._empty_prefix_metrics()

        delta_glrt = np.zeros(max_len, dtype=np.float64)
        delta_w = np.zeros(max_len, dtype=np.float64)

        for pos in range(self.converted_msg_length):
            step_list = per_pos_token_step[pos]
            post_list = per_pos_token_post[pos]
            weight_list = per_pos_token_weight[pos]
            if not step_list:
                continue
            try:
                detected_msg = int(predicted_msg[pos])
            except Exception:
                detected_msg = 0

            for step_idx, log_post_vec, w_pos in zip(step_list, post_list, weight_list):
                w = float(w_pos)
                if w <= 0.0:
                    continue
                T_idx = int(step_idx) + 1
                if T_idx <= 0 or T_idx > max_len:
                    continue

                v = log_post_vec
                p_det = float(torch.exp(v[detected_msg]).clamp_max(1.0))
                eps = 1e-100
                lpo_val = float(np.log(max(p_det, eps)) - np.log(max(1.0 - p_det, eps)))

                delta_glrt[T_idx - 1] += w * lpo_val
                delta_w[T_idx - 1] += w

        cum_glrt = np.cumsum(delta_glrt)
        cum_w = np.cumsum(delta_w)
        z_seq = np.full(max_len, np.nan, dtype=np.float64)
        valid = cum_w > 0.0
        z_seq[valid] = cum_glrt[valid] / cum_w[valid]
        z_score_at_T = torch.as_tensor(z_seq, dtype=torch.float32, device=self.device)

        if not gold_message:
            return z_score_at_T, torch.tensor([], dtype=torch.float32, device=self.device)

        M = self.num_buckets
        agg_lp = {
            pos: torch.zeros(M, device=self.device, dtype=torch.float64)
            for pos in range(self.converted_msg_length)
        }
        idx_pos = {pos: 0 for pos in range(self.converted_msg_length)}
        bit_acc_seq: list[float] = []

        for step_idx in range(max_len):
            for pos in range(self.converted_msg_length):
                step_list = per_pos_token_step[pos]
                post_list = per_pos_token_post[pos]
                weight_list = per_pos_token_weight[pos]
                k = idx_pos[pos]
                while k < len(step_list) and int(step_list[k]) == step_idx:
                    w = float(weight_list[k])
                    if w > 0.0:
                        v = post_list[k]
                        p_vec = torch.exp(v)
                        p_eps = float(self.posterior_eps)
                        p_vec = torch.clamp(p_vec, p_eps, 1.0 - p_eps)
                        logit_vec = torch.log(p_vec) - torch.log(1.0 - p_vec)
                        agg_lp[pos] += w * logit_vec
                    k += 1
                idx_pos[pos] = k

            digits_T: list[int] = []
            for pos in range(self.converted_msg_length):
                v_pos = agg_lp[pos]
                if torch.sum(torch.abs(v_pos)).item() == 0.0:
                    digits_T.append(0)
                else:
                    digits_T.append(int(torch.argmax(v_pos).item()))

            try:
                correct_bits_T, total_bits_T = self._compute_bit_accuracy(digits_T, gold_message)
                bit_acc_T = (correct_bits_T / total_bits_T) if total_bits_T > 0 else float("nan")
            except Exception:
                bit_acc_T = float("nan")
            bit_acc_seq.append(float(bit_acc_T))

        if bit_acc_seq and np.isfinite(final_bit_acc):
            bit_acc_seq[-1] = float(final_bit_acc)

        bit_acc_at_T = torch.as_tensor(bit_acc_seq, dtype=torch.float32, device=self.device)
        return z_score_at_T, bit_acc_at_T

    @torch.no_grad()
    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        message: str = "",
        prompt_len: int = 0,
        window_size: int = None,
        window_stride: int = None,
        return_green_token_mask: bool = False,
        convert_to_float: bool = True,
        return_z_at_T: bool = False,
        col_name: str = None,
        position: str = None,
        **kwargs
    ) -> dict:
        """Single-example wrapper over detect_batch()."""
        assert (text is not None) ^ (tokenized_text is not None), "Must pass either text or tokenized_text"

        texts = None
        tokenized_texts = None

        if tokenized_text is None:
            texts = [text]
        else:
            if isinstance(tokenized_text, torch.Tensor):
                tokenized_text = tokenized_text.detach().cpu().tolist()
            bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
            if tokenized_text and bos_token_id is not None and tokenized_text[0] == bos_token_id:
                tokenized_text = tokenized_text[1:]
            tokenized_texts = [tokenized_text]

        messages = [message] if isinstance(message, str) and len(message) > 0 else None
        prompt_lens = [prompt_len]

        outs = self.detect_batch(
            texts=texts,
            tokenized_texts=tokenized_texts,
            return_prediction=return_prediction,
            return_scores=return_scores,
            messages=messages,
            prompt_lens=prompt_lens,
            window_size=window_size,
            window_stride=window_stride,
            return_green_token_mask=return_green_token_mask,
            convert_to_float=convert_to_float,
            return_z_at_T=return_z_at_T,
            col_name=col_name,
            position=[position] if position is not None else None,
            **kwargs,
        )
        return outs[0]

    @torch.no_grad()
    def detect_batch(
        self,
        texts: list[str] = None,
        tokenized_texts: list[list[int]] = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        messages: list[str] | None = None,
        prompt_lens: list[int] | None = None,
        window_size: int | None = None,
        window_stride: int | None = None,
        return_green_token_mask: bool = False,
        convert_to_float: bool = True,
        return_z_at_T: bool = False,
        col_name: str | None = None,
        position: list[str] | None = None,
        **kwargs,
    ) -> list[dict]:
        """Batched detection for quantile watermark."""
        assert (texts is not None) ^ (tokenized_texts is not None), "Must pass either texts or tokenized_texts"

        def _maybe_cuda_sync():
            try:
                dev = getattr(self, "device", None)
                is_cuda = False
                if isinstance(dev, torch.device):
                    is_cuda = (dev.type == "cuda")
                elif isinstance(dev, str):
                    is_cuda = ("cuda" in dev)
                if is_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass

        if (
            texts is not None
            and self.wrap_output_in_chat_template
            and getattr(self, "tokenizer", None) is not None
        ):
            apply_wrap = (prompt_lens is None) or all(((pl or 0) == 0) for pl in prompt_lens)
            if apply_wrap:
                wrapped_texts = []
                computed_prompt_lens = []
                assistant_hdr = "<|start_header_id|>assistant<|end_header_id|>"
                user_hdr = "<|start_header_id|>user<|end_header_id|>"

                for t in texts:
                    raw = t or ""
                    try:
                        s = raw.lstrip()
                        for marker in ("<|begin_of_text|>", "<|eot_id|>"):
                            while s.startswith(marker):
                                s = s[len(marker) :].lstrip()
                        if s.startswith(user_hdr):
                            s = s[len(user_hdr) :].lstrip()
                        if s.startswith(assistant_hdr):
                            s = s[len(assistant_hdr) :].lstrip()

                        body = s
                        prefix = assistant_hdr + "\n\n"
                        combined = prefix + body
                        wrapped_texts.append(combined)

                        p_ids = self.tokenizer(
                            prefix,
                            return_tensors=None,
                            add_special_tokens=False,
                        )["input_ids"]
                        bos_id = getattr(self.tokenizer, "bos_token_id", None)
                        if bos_id is not None and len(p_ids) > 0 and p_ids[0] == bos_id:
                            computed_prompt_lens.append(len(p_ids) - 1)
                        else:
                            computed_prompt_lens.append(len(p_ids))
                    except Exception:
                        wrapped_texts.append(raw)
                        computed_prompt_lens.append(0)

                texts = wrapped_texts
                prompt_lens = computed_prompt_lens

        if tokenized_texts is None:
            assert self.tokenizer is not None, "Need tokenizer for raw text"
            tokenized_lists = []
            for t in texts:
                ids = self.tokenizer(t, return_tensors=None, add_special_tokens=False)["input_ids"]
                bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
                if ids and bos_token_id is not None and ids[0] == bos_token_id:
                    ids = ids[1:]
                tokenized_lists.append(ids)
        else:
            tokenized_lists = tokenized_texts

        batch_size = len(tokenized_lists)
        if prompt_lens is None:
            prompt_lens = [0] * batch_size
        lengths = [len(x) for x in tokenized_lists]
        results: list[dict | None] = [None] * batch_size
        valid_indices = []
        for b in range(batch_size):
            max_len = lengths[b] - (prompt_lens[b] or 0)
            if max_len <= 0:
                results[b] = self.dummy_detect(
                    return_prediction=return_prediction,
                    return_scores=return_scores,
                    return_green_token_mask=return_green_token_mask,
                    return_z_at_T=return_z_at_T,
                )
            else:
                valid_indices.append(b)
        if len(valid_indices) == 0:
            return results  

        if self.debug and not self._debug_printed_input:
            try:
                b0 = valid_indices[0]
                ids0 = tokenized_lists[b0]
                pl0 = int(prompt_lens[b0] or 0)
                print("\n[QuantileWatermarkDetector DEBUG] First processed input entering detect logic")
                print(f"[QuantileWatermarkDetector DEBUG] prompt_len={pl0} total_tokens={len(ids0)}")
                if texts is not None:
                    print("[QuantileWatermarkDetector DEBUG] raw_text_after_wrap:")
                    print(texts[b0])
                if self.tokenizer is not None:
                    print("[QuantileWatermarkDetector DEBUG] decoded_from_token_ids:")
                    print(self.tokenizer.decode(ids0, skip_special_tokens=False))
                else:
                    print("[QuantileWatermarkDetector DEBUG] token_ids:")
                    print(ids0)
            except Exception as e:
                print(f"[QuantileWatermarkDetector DEBUG] Failed to print debug input: {e}")
            finally:
                self._debug_printed_input = True

        _maybe_cuda_sync()
        t0 = time.perf_counter()

        pad_id = getattr(self.tokenizer, "pad_token_id", 0) if self.tokenizer is not None else 0
        max_seq_len = max(lengths) if lengths else 0
        batch_ids = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=self.device)
        for i, ids in enumerate(tokenized_lists):
            L = len(ids)
            if L > 0:
                batch_ids[i, :L] = torch.tensor(ids, dtype=torch.long, device=self.device)
                attention_mask[i, :L] = 1

        if self.base_processor.rng is None:
            self.base_processor.rng = torch.Generator(device=self.device)

        out = self.model(batch_ids, attention_mask=attention_mask, use_cache=False)
        logits_all = out.logits

        M = int(self.num_buckets)
        P = int(self.converted_msg_length)
        glrt_mode = getattr(self, "glrt_mode", "lpo")
        need_prefix_stats = (
            return_scores
            and return_z_at_T
            and glrt_mode != "strict"
        )
        need_strict = (glrt_mode == "strict")
        need_score_token_lists = need_prefix_stats or need_strict

        log_posterior_total = [
            {i: torch.zeros(M, device=self.device, dtype=torch.float64) for i in range(P)}
            for _ in range(batch_size)
        ]
        log_post_total = [
            {i: torch.zeros(M, device=self.device, dtype=torch.float64) for i in range(P)}
            for _ in range(batch_size)
        ]
        token_count_score = [
            {i: 0.0 for i in range(P)}
            for _ in range(batch_size)
        ]

        per_pos_token_ll = None
        per_pos_token_post = None
        per_pos_token_weight = None
        per_pos_token_step = None
        if need_score_token_lists:
            per_pos_token_post = [{i: [] for i in range(P)} for _ in range(batch_size)]
            per_pos_token_weight = [{i: [] for i in range(P)} for _ in range(batch_size)]
            if need_prefix_stats:
                per_pos_token_step = [{i: [] for i in range(P)} for _ in range(batch_size)]
            if need_strict:
                per_pos_token_ll = [{i: [] for i in range(P)} for _ in range(batch_size)]

        idx_f = torch.arange(M, device=self.device, dtype=torch.float64)
        inv_M = 1.0 / float(M)
        bucket_starts = idx_f * inv_M
        bucket_ends = (idx_f + 1.0) * inv_M
        bucket_perm_identity = torch.arange(M, device=self.device, dtype=torch.long)
        log_likelihood_total_dec = [
            {i: torch.zeros(M, device=self.device, dtype=torch.float64) for i in range(P)}
            for _ in range(batch_size)
        ]
        log_posterior_total_dec = [
            {i: torch.zeros(M, device=self.device, dtype=torch.float64) for i in range(P)}
            for _ in range(batch_size)
        ]
        for b in range(batch_size):
            seq_len = lengths[b]
            if seq_len == 0:
                continue
            p_len = prompt_lens[b] or 0
            max_len = seq_len - p_len
            if max_len <= 0:
                continue
            skip_n = int(max_len * self.skip_ratio)

            for step in range(max_len):
                if p_len == 0 and step == 0:
                    continue
                if step < skip_n:
                    continue

                if step < self.n_gram_len:
                    continue

                l = p_len + step - self.n_gram_len
                r = p_len + step
                cur_seed = batch_ids[b:b+1, l:r]
                self.base_processor._seed_rng(cur_seed)
                seed_key = int(self.base_processor.prf_key)
                bit_position = int(seed_key % self.converted_msg_length)

                pred_logits = logits_all[b, p_len + step - 1, :].unsqueeze(0)
                filtered_logits = _apply_top_k_top_p_filtering(
                    pred_logits,
                    top_k=self.top_k,
                    top_p=self.top_p,
                )

                candidate_indices = (filtered_logits > -float("Inf")).squeeze(0).nonzero().squeeze()
                if candidate_indices.numel() == 0:
                    continue
                if candidate_indices.dim() == 0:
                    candidate_indices = candidate_indices.unsqueeze(0)

                candidate_scores = filtered_logits.squeeze(0)[candidate_indices]
                probs = torch.softmax(candidate_scores, dim=-1)
                probs_double = probs.to(torch.float64)
                sorted_probs, sorted_indices_relative = torch.sort(probs_double, descending=True)

                sorted_indices_absolute = candidate_indices[sorted_indices_relative]
                cdf = torch.cumsum(sorted_probs, dim=0)

                observed_token_id = batch_ids[b, p_len + step]
                rank_tensor = (sorted_indices_absolute == observed_token_id).nonzero(as_tuple=False).view(-1)
                if rank_tensor.numel() == 0:
                    continue

                rank = rank_tensor[0]
                token_cdf_end = cdf.gather(0, rank.unsqueeze(0)).squeeze(0)
                token_prob = sorted_probs.gather(0, rank.unsqueeze(0)).squeeze(0)
                token_cdf_start = token_cdf_end - token_prob

                if self.mapping_scheme == 'identity':
                    starts_m = bucket_starts
                    ends_m = bucket_ends
                else:
                    h = self._hash_to_int(seed_key, bit_position)
                    if self.mapping_scheme == 'cyclic':
                        shift = int(h % M)
                        perm = (bucket_perm_identity + shift) % M
                    elif self.mapping_scheme == 'permute':
                        perm = self._get_bucket_permutation_torch(h)
                    else:
                        perm = bucket_perm_identity
                    starts_m = bucket_starts.index_select(0, perm)
                    ends_m = bucket_ends.index_select(0, perm)

                intersection_start = torch.maximum(token_cdf_start, starts_m)
                intersection_end = torch.minimum(token_cdf_end, ends_m)
                intersection_len = torch.clamp(intersection_end - intersection_start, min=0.0)
                likelihoods = intersection_len * M

                log_likelihoods = torch.log(likelihoods.clamp_min(1e-200))
                total_like = torch.sum(likelihoods)
                if float(total_like.item()) <= 0.0:
                    continue

                post = likelihoods / total_like
                eps = float(self.posterior_eps)
                post_clamped = torch.clamp(post, min=eps, max=1.0 - eps)
                log_post = torch.log(post_clamped)
                log_odds = log_post - torch.log(1.0 - post_clamped)
                pos = int(bit_position)

                log_post_total[b][pos] += log_post
                log_posterior_total[b][pos] += log_odds
                token_count_score[b][pos] += 1.0
                if per_pos_token_post is not None and per_pos_token_weight is not None:
                    per_pos_token_post[b][pos].append(log_post.detach())
                    per_pos_token_weight[b][pos].append(1.0)
                    if per_pos_token_step is not None:
                        per_pos_token_step[b][pos].append(step)
                    if per_pos_token_ll is not None:
                        per_pos_token_ll[b][pos].append(log_likelihoods.detach())

                log_likelihood_total_dec[b][pos] += log_likelihoods
                log_posterior_total_dec[b][pos] += log_odds

        # After evidence accumulation, decode each message position and then
        # derive scalar metrics or prefix curves from the same buffers.
        for b in range(batch_size):
            if results[b] is not None:
                continue
            out_dict = {
                'decoding_time': float("nan"),
            }

            predicted_msg = []
            token_cnt = 0
            raw_sum = 0.0
            block_margins = []
            block_token_counts = []
            if self.glrt_mode == 'strict':
                # Strict GLRT: use likelihood ratio derived from per-token channel posterior.
                # Under a uniform prior over buckets, for detected bucket r*:
                #   LLR_t(r*) = log( p_wm(x_t | r*) / p_0(x_t) )
                #            = log(M) + log P(r* | x_t)
                # where P(r* | x_t) is the bucket posterior computed from the per-bucket
                # likelihood vector (and clipped for numerical stability).
                llr_sum = 0.0
                log_M = float(np.log(self.num_buckets))
                for pos in range(self.converted_msg_length):
                    if torch.sum(log_likelihood_total_dec[b][pos]).item() == 0:
                        predicted_msg.append(0)
                        block_margins.append(0.0)
                        block_token_counts.append(0)
                        continue
                    ll_vec = log_likelihood_total_dec[b][pos]
                    detected_msg = int(torch.argmax(ll_vec).item())
                    predicted_msg.append(detected_msg)
                    try:
                        max_ll = float(torch.max(ll_vec).item())
                        mean_ll = float(torch.mean(ll_vec).item())
                        block_margins.append(max_ll - mean_ll)
                    except Exception:
                        block_margins.append(0.0)
                    c_tok = 0.0
                    ll_list = per_pos_token_ll[b][pos]
                    post_list = per_pos_token_post[b][pos]
                    weight_list = per_pos_token_weight[b][pos]
                    for log_like_vec, log_post_vec, w_pos in zip(ll_list, post_list, weight_list):
                        w = float(w_pos)
                        if w <= 0.0:
                            continue
                        v_target_ll = float(log_like_vec[detected_msg].item())
                        raw_sum += w * v_target_ll
                        llr = log_M + float(log_post_vec[detected_msg].item())
                        llr_sum += w * llr
                        token_cnt += w
                        c_tok += w
                    block_token_counts.append(c_tok)
                wm_score_glrt = (llr_sum / token_cnt) if token_cnt > 0 else 0.0
            else:
                # Default / LPO-style GLRT using posterior odds.
                glrt_sum = 0.0
                for pos in range(self.converted_msg_length):
                    if torch.sum(log_posterior_total_dec[b][pos]).item() == 0:
                        predicted_msg.append(0)
                        block_margins.append(0.0)
                        block_token_counts.append(0)
                        continue
                    lp = log_posterior_total_dec[b][pos]
                    detected_msg = int(torch.argmax(lp).item())
                    predicted_msg.append(detected_msg)
                    try:
                        max_lp = torch.max(lp).item()
                        mean_lp = torch.mean(lp).item()
                        block_margins.append(max_lp - mean_lp)
                    except Exception:
                        block_margins.append(0.0)

                    c_tok = float(token_count_score[b][pos])
                    if c_tok > 0.0:
                        try:
                            raw_sum += float(log_post_total[b][pos][detected_msg].item())
                        except Exception:
                            pass
                        try:
                            glrt_sum += float(log_posterior_total[b][pos][detected_msg].item())
                        except Exception:
                            pass
                        token_cnt += c_tok
                    block_token_counts.append(c_tok)
                wm_score_glrt = (glrt_sum / token_cnt) if token_cnt > 0 else 0.0

            wm_score = wm_score_glrt

            if messages is not None and b < len(messages) and isinstance(messages[b], str) and messages[b] != "":
                correct_bits, total_bits = self._compute_bit_accuracy(predicted_msg, messages[b])
                out_dict["bit_acc"] = correct_bits / total_bits if total_bits > 0 else float("nan")
                out_dict["bit_match"] = (correct_bits == total_bits)
                try:
                    gold_dec = int(messages[b], 2)
                    gold_vec = self._int_to_digits(
                        gold_dec,
                        self.num_buckets,
                        self.converted_msg_length,
                    )

                    eff_cnt = 0
                    match_cnt = 0
                    block_match_vec = []
                    for pos in range(self.converted_msg_length):
                        if block_token_counts[pos] > 0:
                            eff_cnt += 1
                            m = int(predicted_msg[pos] == gold_vec[pos])
                            match_cnt += m
                            block_match_vec.append(m)
                        else:
                            block_match_vec.append(0)
                    out_dict["block_match_rate"] = (match_cnt / eff_cnt) if eff_cnt > 0 else 0.0
                    out_dict["block_match_vec"] = block_match_vec
                    out_dict["block_margins"] = [float(x) for x in block_margins]
                    out_dict["block_token_counts"] = block_token_counts
                    out_dict["pred_digits"] = "".join(map(str, predicted_msg))
                    out_dict["gold_digits"] = "".join(map(str, gold_vec))
                except Exception:
                    out_dict["block_match_rate"] = float("nan")
                    out_dict["block_match_vec"] = []
                    out_dict["block_margins"] = [float(x) for x in block_margins]
                    out_dict["block_token_counts"] = block_token_counts
                    out_dict["pred_digits"] = "".join(map(str, predicted_msg))
                    out_dict["gold_digits"] = ""
            else:
                out_dict["bit_acc"] = float("nan")
                out_dict["bit_match"] = None
                out_dict["block_match_rate"] = float("nan")
                out_dict["block_match_vec"] = []
                out_dict["block_margins"] = [float(x) for x in block_margins]
                out_dict["block_token_counts"] = block_token_counts

            if (
                return_scores
                and return_z_at_T
                and getattr(self, "glrt_mode", "lpo") != "strict"
            ):
                p_len = prompt_lens[b] or 0
                max_len = lengths[b] - p_len
                gold_message = None
                if (
                    messages is not None
                    and b < len(messages)
                    and isinstance(messages[b], str)
                    and messages[b] != ""
                ):
                    gold_message = messages[b]
                final_bit_acc = float(out_dict.get("bit_acc", float("nan")))
                z_score_at_T, bit_acc_at_T = self._compute_prefix_metrics(
                    max_len=max_len,
                    predicted_msg=predicted_msg,
                    gold_message=gold_message,
                    final_bit_acc=final_bit_acc,
                    per_pos_token_post=per_pos_token_post[b],
                    per_pos_token_weight=per_pos_token_weight[b],
                    per_pos_token_step=per_pos_token_step[b],
                )
                out_dict["z_score_at_T"] = z_score_at_T
                out_dict["bit_acc_at_T"] = bit_acc_at_T
            elif return_z_at_T:
                out_dict["z_score_at_T"], out_dict["bit_acc_at_T"] = self._empty_prefix_metrics()

            if return_scores:
                out_dict["wm_score_glrt"] = wm_score_glrt
                out_dict["z_score"] = wm_score
                out_dict["raw_log_likelihood"] = raw_sum
                out_dict["pred_message"] = "".join(map(str, predicted_msg))
                out_dict["token_count_scored"] = token_cnt

            if return_prediction:
                out_dict["prediction"] = wm_score > -50.0

            results[b] = out_dict

        _maybe_cuda_sync()
        elapsed = time.perf_counter() - t0
        per_example = float(elapsed) / float(len(valid_indices)) if len(valid_indices) > 0 else float("nan")
        for b in valid_indices:
            try:
                if results[b] is not None:
                    results[b]["decoding_time"] = float(per_example)
            except Exception:
                pass
        return results  

    def dummy_detect(
        self,
        return_prediction: bool = True,
        return_scores: bool = True,
        return_green_token_mask: bool = False,
        return_z_at_T: bool = True,
    ):
        score_dict = {
            "pred_message": "",
            "bit_acc": float("nan"),
            "bit_match": None,
            "decoding_time": float("nan"),
            "raw_log_likelihood": float("nan"),
            "token_count_scored": float("nan"),
            "block_match_rate": float("nan"),
            "block_match_vec": [],
            "block_margins": [],
            "block_token_counts": [],
            "pred_digits": "",
            "gold_digits": "",
            "wm_score_glrt": float("nan"),
            "z_score": 0.0,
        }
        if return_green_token_mask:
            score_dict["green_token_mask"] = []
        if return_z_at_T:
            score_dict["z_score_at_T"] = torch.tensor([])
            score_dict["bit_acc_at_T"] = torch.tensor([])

        output_dict = {}
        if return_scores:
            output_dict.update(score_dict)
        if return_prediction:
            output_dict["prediction"] = False

        return output_dict

    def _compute_bit_accuracy(self, pred_msg: list, gold_message: str):
        decimal = self._digits_to_int(list(pred_msg), self.num_buckets)
        decimal = min(decimal, 2 ** self.message_length - 1)
        binary_pred = format(decimal, f"0{self.message_length}b")

        if len(binary_pred) != len(gold_message):
            raise RuntimeError("Predicted message length != gold message length")

        match = sum(g == p for g, p in zip(gold_message, binary_pred))
        return match, len(gold_message)
