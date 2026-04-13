# coding=utf-8
"""
Message-space processors for structured, reversible encodings.

Currently provides:

    BlockLocalMessageProcessor

which performs a block-wise, invertible transform on base-M message digits to
induce locality / correlation within each block, while preserving the total
payload bit-length.

This module is intentionally independent of any specific watermarking scheme so
that different processors (quantile, MPAC, etc.) can reuse it.
"""

from __future__ import annotations

from typing import List


class BlockLocalMessageProcessor:
    """
    Block-wise locality-inducing message processor based on a simple
    repetition code in base-M.

    Given:
        - message_bits: total payload length in bits (B)
        - chunk_capacity: log2(M), where M is the base / num_buckets
        - block_size: number of repeated base-M digits per latent symbol (r)

    we:
        1) map a binary payload of length B to K_latent base-M digits
           D \in [0, M-1]^{K_latent}, where K_latent matches the "full
           capacity" representation used by the quantile processor:
               K_latent = len(base-M representation of (2^B - 1))
        2) map each latent digit d to a block of r repeated digits:
               (d) -> (d, d, ..., d)  (length r)
           concatenating blocks to obtain S \in [0, M-1]^{K_latent * r}.
        3) to decode, we group S into blocks of length r and apply a simple
           nearest-codeword / majority-vote rule within each block to recover
           the latent digit, then invert the base-M representation back to B
           bits.

    This introduces redundancy (codeword length > payload symbol length) and
    yields strong locality in each block (ideally all digits identical, with
    robustness to a few corrupt positions), while preserving the total payload
    bit-length at B via a many-to-one mapping from codewords back to messages.
    """

    def __init__(
        self,
        message_bits: int,
        chunk_capacity: int,
        block_size: int,
        mapping_key: str | None = None,
    ) -> None:
        if message_bits <= 0:
            raise ValueError(f"message_bits must be > 0, got {message_bits}")
        if chunk_capacity <= 0:
            raise ValueError(f"chunk_capacity must be > 0, got {chunk_capacity}")
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")

        self.message_bits: int = int(message_bits)
        self.chunk_capacity: int = int(chunk_capacity)
        self.base: int = 2**self.chunk_capacity  # M

        # Compute the number of latent base-M digits required to represent any
        # B-bit payload, using the same convention as quantile processors:
        # use the representation of an all-ones B-bit integer.
        decimal = int("1" * self.message_bits, 2)
        self.latent_num_digits: int = len(
            self._int_to_base_digits(decimal, self.base)
        )
        if self.latent_num_digits <= 0:
            raise ValueError("Internal error: latent_num_digits computed as <= 0.")

        self.block_size: int = int(block_size)
        # Each latent digit expands into one block of repeated digits.
        self.num_blocks: int = self.latent_num_digits
        self.num_digits: int = self.latent_num_digits * self.block_size

        # Currently unused, but kept for potential key-dependent variants.
        self.mapping_key: str | None = mapping_key

    # ------------------------------------------------------------------
    # Basic integer / digit conversions
    # ------------------------------------------------------------------

    def _int_to_base_digits(self, n: int, base: int, width: int | None = None) -> List[int]:
        """
        Convert integer n to a list of base-`base` digits, optionally left-padded
        with zeros to length `width`.
        """
        if n < 0:
            raise ValueError("n must be non-negative.")
        if base <= 1:
            raise ValueError("base must be >= 2.")

        if n == 0:
            digits: List[int] = [0]
        else:
            digits = []
            while n:
                digits.append(int(n % base))
                n //= base
            digits = digits[::-1]

        if width is not None and len(digits) < width:
            digits = [0] * (width - len(digits)) + digits
        return digits

    def _digits_to_int(self, digits: List[int], base: int) -> int:
        """
        Convert a list of base-`base` digits back to an integer.
        """
        if base <= 1:
            raise ValueError("base must be >= 2.")
        val = 0
        for d in digits:
            if d < 0 or d >= base:
                raise ValueError(f"Digit {d} out of range for base {base}.")
            val = val * base + int(d)
        return val

    # ------------------------------------------------------------------
    # Block-wise repetition code (and inverse)
    # ------------------------------------------------------------------

    def _encode_block(self, digit: int) -> List[int]:
        """
        Encode one latent digit as a block of repeated digits.
        """
        d = int(digit) % self.base
        return [d] * self.block_size

    def _decode_block(self, block_syms: List[int]) -> int:
        """
        Decode one block of digits back to a latent digit using a simple
        majority-vote / nearest-codeword rule in Hamming distance.
        """
        if not block_syms:
            return 0
        # Count occurrences of each possible symbol and pick the argmax.
        counts = [0] * self.base
        for x in block_syms:
            x_int = int(x) % self.base
            counts[x_int] += 1
        # Argmax with stable tie-breaking on symbol index.
        best_symbol = 0
        best_count = -1
        for sym, cnt in enumerate(counts):
            if cnt > best_count:
                best_count = cnt
                best_symbol = sym
        return best_symbol

    # ------------------------------------------------------------------
    # Public encode/decode API
    # ------------------------------------------------------------------

    def encode(self, binary_msg: str) -> List[int]:
        """
        Encode a binary payload into block-local base-M digits.

        Args:
            binary_msg: bit-string of length == message_bits

        Returns:
            List[int]: base-M digits after applying the block-wise transform,
                       length == num_digits.
        """
        if not isinstance(binary_msg, str):
            raise TypeError("binary_msg must be a string of '0'/'1' characters.")
        if len(binary_msg) != self.message_bits:
            raise ValueError(
                f"Expected binary_msg of length {self.message_bits}, got {len(binary_msg)}."
            )

        # Map bits -> integer -> latent base-M digits (length = latent_num_digits)
        m_val = int(binary_msg, 2)
        if m_val < 0 or m_val >= (1 << self.message_bits):
            raise ValueError("binary_msg is not a valid {message_bits}-bit value.")

        latent_digits = self._int_to_base_digits(m_val, self.base, width=self.latent_num_digits)
        if len(latent_digits) != self.latent_num_digits:
            raise RuntimeError("Internal error: latent digit length mismatch in encode().")

        # Apply block-wise repetition code.
        out: List[int] = []
        for d in latent_digits:
            out.extend(self._encode_block(d))

        if len(out) != self.num_digits:
            raise RuntimeError("Internal error: encoded length mismatch in encode().")
        return out

    def decode(self, digits: List[int]) -> str:
        """
        Decode base-M digits produced by `encode` back to a binary payload.

        Args:
            digits: list of base-M digits, length == num_digits

        Returns:
            str: bit-string of length message_bits.
        """
        if len(digits) != self.num_digits:
            raise ValueError(
                f"Expected {self.num_digits} digits for decode, got {len(digits)}."
            )

        # Invert block-wise repetition code to recover latent digits.
        latent: List[int] = []
        for b in range(self.num_blocks):
            start = b * self.block_size
            end = start + self.block_size
            block_syms = [int(x) for x in digits[start:end]]
            latent_sym = self._decode_block(block_syms)
            latent.append(latent_sym)

        # Map latent base-M digits -> integer -> bits.
        m_val = self._digits_to_int(latent, self.base)
        # Clamp to the valid range for message_bits to guard against any
        # pathological issues (should not trigger under normal operation).
        max_val = (1 << self.message_bits) - 1
        if m_val > max_val:
            m_val = max_val

        return format(m_val, f"0{self.message_bits}b")


class LinearBlockMessageProcessor:
    """
    Short block linear code on base-M digits to induce locality with redundancy.

    This implements a simple [3,2] linear code per block:

        (u, v) -> (s1, s2, s3) = (u, v, u+v mod M)

    where M = 2**chunk_capacity is the base. The overall mapping is:

        bits (length B)
          -> integer
          -> latent digits D (length K_latent)
          -> groups of 2 digits per block, each encoded as 3 code digits
          -> code digits S (length num_digits = 3 * (K_latent/2)).

    Decoding in this processor is a hard decoder, primarily used for computing
    bit accuracy given a sequence of (possibly noisy) code digits. Detection-
    side soft-decoding should use per-position posteriors rather than this
    class directly.
    """

    def __init__(
        self,
        message_bits: int,
        chunk_capacity: int,
        mapping_key: str | None = None,
    ) -> None:
        if message_bits <= 0:
            raise ValueError(f"message_bits must be > 0, got {message_bits}")
        if chunk_capacity <= 0:
            raise ValueError(f"chunk_capacity must be > 0, got {chunk_capacity}")

        self.message_bits: int = int(message_bits)
        self.chunk_capacity: int = int(chunk_capacity)
        self.base: int = 2**self.chunk_capacity  # M

        # Latent digits: base-M representation of the all-ones B-bit integer.
        decimal = int("1" * self.message_bits, 2)
        self.latent_num_digits: int = len(
            self._int_to_base_digits(decimal, self.base)
        )
        if self.latent_num_digits <= 0:
            raise ValueError("Internal error: latent_num_digits computed as <= 0.")
        if self.latent_num_digits % 2 != 0:
            raise ValueError(
                f"latent_num_digits={self.latent_num_digits} must be even for [3,2] block code."
            )

        # Two latent digits per block, three code digits per block.
        self.k_latent_per_block: int = 2
        self.n_code_per_block: int = 3
        self.num_blocks: int = self.latent_num_digits // self.k_latent_per_block
        self.num_digits: int = self.num_blocks * self.n_code_per_block

        # Currently unused, but kept for potential key-dependent variants.
        self.mapping_key: str | None = mapping_key

    # ------------------------------------------------------------------
    # Basic integer / digit conversions (shared semantics with above)
    # ------------------------------------------------------------------

    def _int_to_base_digits(self, n: int, base: int, width: int | None = None) -> List[int]:
        if n < 0:
            raise ValueError("n must be non-negative.")
        if base <= 1:
            raise ValueError("base must be >= 2.")
        if n == 0:
            digits: List[int] = [0]
        else:
            digits = []
            while n:
                digits.append(int(n % base))
                n //= base
            digits = digits[::-1]
        if width is not None and len(digits) < width:
            digits = [0] * (width - len(digits)) + digits
        return digits

    def _digits_to_int(self, digits: List[int], base: int) -> int:
        if base <= 1:
            raise ValueError("base must be >= 2.")
        val = 0
        for d in digits:
            if d < 0 or d >= base:
                raise ValueError(f"Digit {d} out of range for base {base}.")
            val = val * base + int(d)
        return val

    # ------------------------------------------------------------------
    # Block-wise [3,2] encoder / decoder
    # ------------------------------------------------------------------

    def _encode_block(self, u: int, v: int) -> List[int]:
        """
        Encode a pair of latent digits into a [3,2] codeword:
            (u, v) -> (u, v, u+v mod base)
        """
        u = int(u) % self.base
        v = int(v) % self.base
        s3 = (u + v) % self.base
        return [u, v, s3]

    def _decode_block_hard(self, s1: int, s2: int, s3: int) -> tuple[int, int]:
        """
        Hard decoder for one [3,2] block. For simplicity and speed in this
        context, we trust the first two positions as the latent digits and
        ignore s3 for decoding.

        This is sufficient to guarantee exact invertibility when the digits
        come from the encoder, while keeping decode() inexpensive. Detection-
        side soft-decoding should use per-position posteriors instead.
        """
        u = int(s1) % self.base
        v = int(s2) % self.base
        return u, v

    # ------------------------------------------------------------------
    # Public encode/decode API
    # ------------------------------------------------------------------

    def encode(self, binary_msg: str) -> List[int]:
        """
        Encode a binary payload into base-M digits using [3,2] blocks.

        Args:
            binary_msg: bit-string of length == message_bits

        Returns:
            List[int]: base-M digits of length == num_digits.
        """
        if not isinstance(binary_msg, str):
            raise TypeError("binary_msg must be a string of '0'/'1' characters.")
        if len(binary_msg) != self.message_bits:
            raise ValueError(
                f"Expected binary_msg of length {self.message_bits}, got {len(binary_msg)}."
            )

        m_val = int(binary_msg, 2)
        if m_val < 0 or m_val >= (1 << self.message_bits):
            raise ValueError("binary_msg is not a valid {message_bits}-bit value.")

        latent = self._int_to_base_digits(
            m_val, self.base, width=self.latent_num_digits
        )
        if len(latent) != self.latent_num_digits:
            raise RuntimeError("Internal error: latent digit length mismatch in encode().")

        out: List[int] = []
        for b in range(self.num_blocks):
            u = latent[2 * b]
            v = latent[2 * b + 1]
            out.extend(self._encode_block(u, v))

        if len(out) != self.num_digits:
            raise RuntimeError("Internal error: encoded length mismatch in encode().")
        return out

    def decode(self, digits: List[int]) -> str:
        """
        Decode base-M digits back into the original binary payload using a
        hard [3,2] block decoder.

        Args:
            digits: list of base-M digits, length == num_digits

        Returns:
            str: bit-string of length message_bits.
        """
        if len(digits) != self.num_digits:
            raise ValueError(
                f"Expected {self.num_digits} digits for decode, got {len(digits)}."
            )

        latent: List[int] = []
        for b in range(self.num_blocks):
            start = b * self.n_code_per_block
            s1, s2, s3 = (
                int(digits[start]) % self.base,
                int(digits[start + 1]) % self.base,
                int(digits[start + 2]) % self.base,
            )
            u, v = self._decode_block_hard(s1, s2, s3)
            latent.append(u)
            latent.append(v)

        m_val = self._digits_to_int(latent, self.base)
        max_val = (1 << self.message_bits) - 1
        if m_val > max_val:
            m_val = max_val

        return format(m_val, f"0{self.message_bits}b")


__all__ = ["BlockLocalMessageProcessor", "LinearBlockMessageProcessor"]
