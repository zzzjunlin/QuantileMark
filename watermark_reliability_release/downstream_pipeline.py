#!/usr/bin/env python
# coding=utf-8
"""
Downstream tasks pipeline for watermark experiments (MT + summarization).

This script is designed to:
  - Run English→Romanian MT on WMT16 En–Ro with MBART.
  - Run summarization on CNN/DailyMail with BART-large.
  - Support different watermarking methods
  - Compute text quality metrics such as BLEU / ROUGE-1 / BERTScore / PPL.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    LogitsProcessorList,
    DataCollatorWithPadding,
)

# Reuse generation helper from existing codebase
from utils.generation import generate as wm_generate

# Quantile-based watermark implementation
from quantile_watermark_processor import QuantileWatermarkLogitsProcessor
# Multibit (MPAC-style) watermark implementation
from mb_watermark_processor import WatermarkLogitsProcessor as MPACWatermarkLogitsProcessor
# StealthInk (reweight-based) watermark implementation
from stealthink_watermark_processor import ReweightProcessor, ReweightLogitsProcessor


@dataclass
class PipelineArgs:
    task: str
    run_name: str
    output_dir: str

    model_name_or_path: str
    dataset_name: str
    dataset_config_name: Optional[str]
    dataset_split: str

    # MT-specific
    src_lang: Optional[str] = None
    tgt_lang: Optional[str] = None

    # Summarization-specific
    source_field: Optional[str] = None
    target_field: Optional[str] = None

    # Watermark
    watermark_method: str = "none"  # none | quantile_watermark | mpac | stealthink
    unit_capacity: int = 1          # bits per sequence (capacity m)
    num_chunks: int = 1             # H (number of chunks)

    prf_type: str = "sha256"
    key_bits: int = 1024
    texture_h: int = 3

    # Quantile watermark parameters (used when watermark_method == "quantile")
    quantile_gamma: float = 0.5
    quantile_chunk_capacity: int = 2
    quantile_message_length: Optional[int] = None  # default: unit_capacity
    quantile_seeding_scheme: str = "lefthash_5"
    quantile_mapping_scheme: str = "permute"
    quantile_epsilon: float = 0.0

    # Sampling / generation
    temperature: float = 1.0
    do_sample: bool = True
    max_new_tokens: int = 60
    batch_size: int = 8
    top_k: int = 0
    top_p: float = 1.0
    random_seed: int = 42

    # Metrics
    eval_metrics: str = "bleu,rouge1,bertscore,ppl"
    debug_metrics: bool = False  # if True, print extra info for BERTScore / PPL

    mpac_delta: float = 2.0


def parse_args() -> PipelineArgs:
    p = argparse.ArgumentParser(description="Downstream MT / summarization watermark pipeline")

    # Core setup
    p.add_argument("--task", type=str, choices=["mt", "summ"], required=True)
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--dataset_config_name", type=str, default=None)
    p.add_argument("--dataset_split", type=str, default="test")

    # MT
    p.add_argument("--src_lang", type=str, default=None)
    p.add_argument("--tgt_lang", type=str, default=None)

    # Summarization
    p.add_argument("--source_field", type=str, default=None)
    p.add_argument("--target_field", type=str, default=None)

    # Watermark
    p.add_argument(
        "--watermark_method",
        type=str,
        default="none",
        choices=["none", "quantile_watermark", "mpac", "stealthink"],
    )
    p.add_argument("--unit_capacity", type=int, default=1)
    p.add_argument("--num_chunks", type=int, default=1)

    p.add_argument("--prf_type", type=str, default="sha256")
    p.add_argument("--key_bits", type=int, default=1024)
    p.add_argument("--texture_h", type=int, default=3)

    # Quantile watermark params
    p.add_argument("--quantile_gamma", type=float, default=0.5)
    p.add_argument("--quantile_chunk_capacity", type=int, default=2)
    p.add_argument("--quantile_message_length", type=int, default=24)
    p.add_argument("--quantile_seeding_scheme", type=str, default="lefthash_5")
    p.add_argument("--quantile_mapping_scheme", type=str, default="permute")
    p.add_argument("--quantile_epsilon", type=float, default=0.0)

    # Generation
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--do_sample", type=str, default="True")
    p.add_argument("--max_new_tokens", type=int, default=60)
    p.add_argument("--top_k", type=int, default=128)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--random_seed", type=int, default=42)

    # Metrics
    p.add_argument("--eval_metrics", type=str, default="bleu,rouge1,bertscore,ppl")
    p.add_argument(
        "--debug_metrics",
        action="store_true",
        help="Print detailed debug info for BERTScore and PPL on a small number of examples.",
    )

    p.add_argument("--mpac_delta", type=float, default=2.0)

    args_ns = p.parse_args()

    # Normalize boolean-like flag
    do_sample_str = str(args_ns.do_sample).lower()
    args_ns.do_sample = do_sample_str in ["true", "1", "yes", "y", "t"]

    return PipelineArgs(**vars(args_ns))


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(args: PipelineArgs) -> Tuple[Any, Any, torch.device]:
    """Load HF model and tokenizer; support both seq2seq and decoder-only."""
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    is_encoder_decoder = getattr(config, "is_encoder_decoder", False)
    if is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
        setattr(args, "is_decoder_only_model", False)
    else:
        # Decoder-only model (e.g., Llama-style instruct)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.float16)
        # Ensure pad token is set for causal models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        # For decoder-only we prefer left padding for generation
        try:
            tokenizer.padding_side = "left"
        except Exception:
            pass
        setattr(args, "is_decoder_only_model", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # For MBART-style models we may need to set src/tgt language codes explicitly.
    if args.task == "mt" and is_encoder_decoder:
        if hasattr(tokenizer, "src_lang") and args.src_lang is not None:
            tokenizer.src_lang = args.src_lang
        if hasattr(tokenizer, "tgt_lang") and args.tgt_lang is not None:
            tokenizer.tgt_lang = args.tgt_lang

    # Store a reasonable maximum sequence length for truncation (encoder/decoder).
    max_len = getattr(model.config, "max_position_embeddings", None)
    if max_len is None:
        max_len = getattr(tokenizer, "model_max_length", None)
    try:
        max_len = int(max_len) if max_len is not None else None
    except Exception:
        max_len = None
    setattr(args, "model_max_length", max_len)

    return model, tokenizer, device


def load_dataset_for_task(args: PipelineArgs):
    """Load HF dataset for MT or summarization."""
    ds = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        split=args.dataset_split,
    )
    return ds


def encode_inputs(
    args: PipelineArgs,
    tokenizer,
    batch: List[str],
    device: torch.device,
    max_source_length: Optional[int] = None,
):
    """Tokenize a batch of source texts."""
    max_len = max_source_length if max_source_length is not None else getattr(args, "model_max_length", None)
    enc = tokenizer(
        batch,
        padding=True,
        truncation=True if max_len is not None else False,
        max_length=max_len,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


def build_summarization_prompt(
    args: PipelineArgs,
    tokenizer,
    article: str,
) -> str:
    """
    Construct a summarization prompt for decoder-only models, using the
    tokenizer's chat_template when available; fall back to a plain instruction
    for seq2seq models or when no chat template is present.
    """
    base_prompt = f"Summarize the following article:\n\n{article}"

    if not getattr(args, "is_decoder_only_model", False):
        return base_prompt

    # Decoder-only: prefer chat template if available
    try:
        has_template = getattr(tokenizer, "chat_template", None)
        if has_template and hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": base_prompt}]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        pass

    return base_prompt


def build_quantile_processor(
    args: PipelineArgs,
    tokenizer,
    device: torch.device,
) -> QuantileWatermarkLogitsProcessor:
    """Instantiate the quantile watermark logits processor."""
    vocab_ids = list(tokenizer.get_vocab().values())

    # Use args.message_length when set (unified bit-length for WM),
    # otherwise fall back to quantile_message_length or unit_capacity.
    message_length = getattr(args, "message_length", None)
    if message_length is None:
        message_length = (
            args.quantile_message_length
            if args.quantile_message_length is not None
            else args.unit_capacity
        )
    message_length = max(1, int(message_length))

    processor = QuantileWatermarkLogitsProcessor(
        vocab=vocab_ids,
        gamma=args.quantile_gamma,
        seeding_scheme=args.quantile_seeding_scheme,
        chunk_capacity=args.quantile_chunk_capacity,
        message_length=message_length,
        top_p=1.0,
        top_k=128,
        device=str(device),
        mapping_scheme=args.quantile_mapping_scheme,
        mapping_key=None,
        epsilon=args.quantile_epsilon,
        tokenizer=tokenizer,
    )

    # For now we use a single global random message of length = message_length.
    rnd = random.Random(args.random_seed)
    binary_msg = "".join(str(rnd.randint(0, 1)) for _ in range(message_length))
    processor.set_message(binary_msg)

    return processor


def build_mpac_processor(
    args: PipelineArgs,
    tokenizer,
    device: torch.device,
) -> MPACWatermarkLogitsProcessor:
    """
    Instantiate the multibit (MPAC-style) watermark logits processor from
    mb_watermark_processor.WatermarkLogitsProcessor.

    We follow the defaults from generation_pipeline:
      - base=2
      - message_length = unit_capacity (bits per sequence)
      - code_length   = message_length
      - gamma ~ 0.5
      - delta = args.mpac_delta
      - seeding_scheme = 'simple_1'
    """
    vocab_ids = list(tokenizer.get_vocab().values())

    message_length = max(1, int(args.unit_capacity))
    wm_kwargs = {
        "use_position_prf": True,
        "use_fixed_position": False,
        "code_length": message_length,
        "use_feedback": False,
        "feedback_args": {},
    }

    processor = MPACWatermarkLogitsProcessor(
        vocab=vocab_ids,
        gamma=0.5,
        delta=args.mpac_delta,
        base=2,
        seeding_scheme="lefthash_5",
        store_spike_ents=False,
        select_green_tokens=True,
        message_length=message_length,
        device=str(device),
        **wm_kwargs,
    )

    # For downstream we let utils.generation.generate sample the message per batch.
    return processor


def build_stealthink_processor(
    args: PipelineArgs,
    tokenizer,
    device: torch.device,
) -> ReweightLogitsProcessor:
    """
    Instantiate the StealthInk (reweight-based) watermark logits processor.

    We mirror the quantile watermark hyperparameters by default:
      - R (gamma)  = args.quantile_gamma
      - base       = int(1 / R)
      - message_length = bit_length (set later via args.message_length)
      - code_length    = message_length
      - seeding_scheme = args.quantile_seeding_scheme
    """
    vocab_ids = list(tokenizer.get_vocab().values())

    # Use unified bit-length set in generate_outputs_quantile (args.message_length)
    message_length = getattr(args, "message_length", None)
    if message_length is None:
        # Fallback: reuse quantile_message_length or unit_capacity
        message_length = (
            args.quantile_message_length
            if getattr(args, "quantile_message_length", None) is not None
            else args.unit_capacity
        )
    message_length = max(1, int(message_length))

    R = float(getattr(args, "quantile_gamma", 0.5))
    base = int(1.0 / R) if R > 0 else 2

    re_proc = ReweightProcessor(
        vocab=vocab_ids,
        gamma=R,
        delta=args.mpac_delta,
        seeding_scheme=getattr(args, "quantile_seeding_scheme", "lefthash_5"),
        select_green_tokens=True,
        base=base,
        message_length=message_length,
        code_length=message_length,
        use_position_prf=True,
        use_fixed_position=False,
        device=str(device),
    )

    processor = ReweightLogitsProcessor(
        reweight_processor=re_proc,
        R=R,
    )

    # For downstream we let utils.generation.generate sample the message per batch.
    return processor


def build_watermark_processor_list(
    args: PipelineArgs,
    tokenizer,
    device: torch.device,
) -> LogitsProcessorList:
    """Build the list of logits processors including any watermark processor."""
    processors = LogitsProcessorList()

    if args.watermark_method == "none":
        # No watermarking; return empty processor list.
        return processors

    if args.watermark_method == "quantile_watermark":
        q_proc = build_quantile_processor(args, tokenizer, device)
        processors.append(q_proc)
        return processors

    if args.watermark_method == "mpac":
        mpac_proc = build_mpac_processor(args, tokenizer, device)
        processors.append(mpac_proc)
        return processors

    if args.watermark_method == "stealthink":
        st_proc = build_stealthink_processor(args, tokenizer, device)
        processors.append(st_proc)
        return processors

    # Stub: other methods not implemented yet.
    # We keep explicit error to avoid silently mis-reporting baselines.
    raise NotImplementedError(
        f"Watermark method '{args.watermark_method}' is not implemented in this pipeline. "
        f"Currently supported: 'none', 'quantile_watermark'."
    )


def generate_outputs_quantile(
    args: PipelineArgs,
    model,
    tokenizer,
    device: torch.device,
    ds,
) -> Tuple[List[str], List[str]]:
    """
    Generate system outputs and collect references using the existing
    utils.generation.generate helper with a watermark processor.

    Supports 'quantile_watermark', 'mpac', and 'stealthink' as watermark_method.
    """
    if args.watermark_method not in ("quantile_watermark", "mpac", "stealthink"):
        raise ValueError(
            "generate_outputs_quantile() requires a watermark_method "
            "of 'quantile_watermark', 'mpac', or 'stealthink'."
        )

    model.eval()

    preds: List[str] = []
    refs: List[str] = []

    # Field extractors per task
    if args.task == "mt":
        src_key = args.src_lang or "en"
        tgt_key = args.tgt_lang or "ro"

        def get_src(ex):
            return ex["translation"][src_key]

        def get_ref(ex):
            return ex["translation"][tgt_key]

    elif args.task == "summ":
        source_field = args.source_field or "article"
        target_field = args.target_field or "highlights"

        def get_src(ex):
            article = ex[source_field]
            if getattr(args, "is_decoder_only_model", False):
                return build_summarization_prompt(args, tokenizer, article)
            else:
                return article

        def get_ref(ex):
            return ex[target_field]
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    # Data collator for batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8)

    # Prepare args object for utils.generation.generate.
    # Unify the bit-length used by sample_message() and the quantile processor.
    bit_length = (
        args.quantile_message_length
        if args.quantile_message_length is not None
        else args.unit_capacity
    )
    bit_length = max(1, int(bit_length))
    setattr(args, "message_length", bit_length)
    setattr(args, "zero_bit", False)
    # keep is_decoder_only_model as set in load_model_and_tokenizer
    setattr(args, "generation_seed", args.random_seed)
    if not hasattr(args, "empty_cache_between_batches"):
        setattr(args, "empty_cache_between_batches", True)

    # Instantiate watermark processor once; its message will be set
    # inside wm_generate() for each batch.
    if args.watermark_method == "quantile_watermark":
        wm_proc = build_quantile_processor(args, tokenizer, device)
    elif args.watermark_method == "mpac":
        wm_proc = build_mpac_processor(args, tokenizer, device)
    elif args.watermark_method == "stealthink":
        wm_proc = build_stealthink_processor(args, tokenizer, device)
    else:
        raise ValueError(f"Unsupported watermark_method for generation: {args.watermark_method}")

    # Generation kwargs
    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=args.max_new_tokens,
        num_beams=1,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=getattr(args, "top_k", 0),
        top_p=getattr(args, "top_p", 1.0),
    )

    # Batch over dataset
    batch_size = args.batch_size
    current_src: List[str] = []
    current_ref: List[str] = []

    for ex in ds:
        s = get_src(ex)
        r = get_ref(ex)
        current_src.append(s)
        current_ref.append(r)

        if len(current_src) >= batch_size:
            batch_preds = _generate_batch_quantile(
                args, model, tokenizer, device, current_src, data_collator, wm_proc, gen_kwargs
            )
            preds.extend(batch_preds)
            refs.extend(current_ref)
            current_src = []
            current_ref = []

    if current_src:
        batch_preds = _generate_batch_quantile(
            args, model, tokenizer, device, current_src, data_collator, wm_proc, gen_kwargs
        )
        preds.extend(batch_preds)
        refs.extend(current_ref)

    return preds, refs


def _generate_batch_quantile(
    args: PipelineArgs,
    model,
    tokenizer,
    device: torch.device,
    src_texts: List[str],
    data_collator: DataCollatorWithPadding,
    watermark_processor: QuantileWatermarkLogitsProcessor,
    gen_kwargs: Dict[str, Any],
) -> List[str]:
    """
    Generate a batch with quantile watermark, while safely handling prompts
    that are shorter than the seeding context width.
    """
    # Build encoded inputs and partition into watermarkable vs short prompts
    input_ids_list: List[torch.Tensor] = []
    for text in src_texts:
        max_len = getattr(args, "model_max_length", None)
        # If the tokenizer provides a chat template, `text` is typically
        # already wrapped (e.g., via build_summarization_prompt). In that
        # case we should not add extra special tokens (BOS) here.
        has_template = getattr(tokenizer, "chat_template", None)
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True if max_len is not None else False,
            max_length=max_len,
            add_special_tokens=False if has_template else True,
        )
        input_ids_list.append(enc["input_ids"])

    ctx_width = getattr(watermark_processor, "n_gram_len", 1)
    wm_indices: List[int] = []
    short_indices: List[int] = []
    for idx, ids in enumerate(input_ids_list):
        if ids.shape[1] >= ctx_width:
            wm_indices.append(idx)
        else:
            short_indices.append(idx)

    outputs_all: List[Optional[str]] = [None] * len(src_texts)  # type: ignore[assignment]

    # 1) Watermarked subset (sufficient context length)
    if wm_indices:
        wm_input_ids = [input_ids_list[i] for i in wm_indices]
        examples_wm = {"input_ids": wm_input_ids}

        generate_without_watermark = lambda **kw: model.generate(**kw, **gen_kwargs)
        generate_with_watermark = lambda **kw: model.generate(
            **kw,
            logits_processor=LogitsProcessorList([watermark_processor]),
            **gen_kwargs,
        )

        out_examples = wm_generate(
            examples=examples_wm,
            data_collator=data_collator,
            generate_without_watermark=generate_without_watermark,
            generate_with_watermark=generate_with_watermark,
            watermark_processor=watermark_processor,
            tokenizer=tokenizer,
            device=device,
            args=args,
        )
        decoded_wm = [d.strip() for d in out_examples["w_wm_output"]]
        for idx_local, idx_global in enumerate(wm_indices):
            outputs_all[idx_global] = decoded_wm[idx_local]

    # 2) Short prompts: generate without watermark
    if short_indices:
        # Build features list for DataCollatorWithPadding
        short_ids = [input_ids_list[i].squeeze(0) for i in short_indices]  # [L]
        short_features = [{"input_ids": ids} for ids in short_ids]
        batch = data_collator(short_features)
        input_batch = batch["input_ids"].to(device)
        attn_mask = batch.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_batch,
                attention_mask=attn_mask,
                **gen_kwargs,
            )
        decoded_short = tokenizer.batch_decode(out, skip_special_tokens=True)
        decoded_short = [d.strip() for d in decoded_short]
        for idx_local, idx_global in enumerate(short_indices):
            outputs_all[idx_global] = decoded_short[idx_local]

    # All positions should now be filled
    return [o if o is not None else "" for o in outputs_all]


def generate_outputs(
    args: PipelineArgs,
    model,
    tokenizer,
    device: torch.device,
    ds,
) -> Tuple[List[str], List[str]]:
    """
    Generate system outputs and collect references using simple HF generation.
    This path is used for non-watermarked baselines (watermark_method=='none').

    Returns:
        preds: list of system outputs (strings)
        refs: list of reference texts (strings)
    """
    if args.watermark_method != "none":
        raise ValueError(
            "generate_outputs() is intended for non-watermarked baselines "
            "(watermark_method == 'none'). For quantile_watermark, use the "
            "quantile-specific generation path."
        )

    model.eval()

    preds: List[str] = []
    refs: List[str] = []

    # Determine fields for current task
    if args.task == "mt":
        # For HF wmt16 with config 'ro-en', typical structure is
        #   example["translation"] = {"en": ..., "ro": ...}
        src_key = args.src_lang or "en"
        tgt_key = args.tgt_lang or "ro"

        def get_src(ex):
            return ex["translation"][src_key]

        def get_ref(ex):
            return ex["translation"][tgt_key]

    elif args.task == "summ":
        source_field = args.source_field or "article"
        target_field = args.target_field or "highlights"

        def get_src(ex):
            article = ex[source_field]
            if getattr(args, "is_decoder_only_model", False):
                return build_summarization_prompt(args, tokenizer, article)
            else:
                return article

        def get_ref(ex):
            return ex[target_field]

    else:
        raise ValueError(f"Unsupported task: {args.task}")

    # Batch over dataset
    batch_size = args.batch_size
    current_src: List[str] = []
    current_ref: List[str] = []

    for ex in ds:
        s = get_src(ex)
        r = get_ref(ex)
        current_src.append(s)
        current_ref.append(r)

        if len(current_src) >= batch_size:
            batch_preds = _generate_batch_plain(
                args, model, tokenizer, device, current_src
            )
            preds.extend(batch_preds)
            refs.extend(current_ref)
            current_src = []
            current_ref = []

    # Last partial batch
    if current_src:
        batch_preds = _generate_batch_plain(
            args, model, tokenizer, device, current_src
        )
        preds.extend(batch_preds)
        refs.extend(current_ref)

    return preds, refs


def _generate_batch_plain(
    args: PipelineArgs,
    model,
    tokenizer,
    device: torch.device,
    src_texts: List[str],
) -> List[str]:
    inputs = encode_inputs(args, tokenizer, src_texts, device)

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=args.max_new_tokens,
        num_beams=1,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=getattr(args, "top_k", 0),
        top_p=getattr(args, "top_p", 1.0),
    )

    outputs = model.generate(
        **inputs,
        **gen_kwargs,
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [d.strip() for d in decoded]


def compute_metrics(
    args: PipelineArgs,
    preds: List[str],
    refs: List[str],
    model,
    tokenizer,
    device: torch.device,
    ds=None,
) -> Dict[str, float]:
    """
    Compute selected metrics over (preds, refs).

    For MT, BLEU / ROUGE-1 / BERTScore-F1 / PPL.
    For summarization, ROUGE-1 / BERTScore-F1 / PPL.
    """
    metrics = [m.strip().lower() for m in args.eval_metrics.split(",") if m.strip()]
    results: Dict[str, float] = {}

    # BLEU (using sacrebleu)
    if "bleu" in metrics:
        try:
            import evaluate

            bleu_metric = evaluate.load("sacrebleu")
            # sacrebleu expects list of list-of-refs
            bleu = bleu_metric.compute(
                predictions=preds,
                references=[[r] for r in refs],
            )
            results["bleu"] = float(bleu["score"])
        except Exception as e:  # pragma: no cover - metric is optional
            results["bleu"] = float("nan")
            print(f"[WARN] BLEU computation failed: {e}")

    # ROUGE-1
    if "rouge1" in metrics:
        try:
            import evaluate

            rouge_metric = evaluate.load("rouge")
            rouge = rouge_metric.compute(
                predictions=preds,
                references=refs,
                use_stemmer=True,
            )
            results["rouge1"] = float(rouge.get("rouge1", float("nan")))
        except Exception as e:  # pragma: no cover
            results["rouge1"] = float("nan")
            print(f"[WARN] ROUGE-1 computation failed: {e}")

    # BERTScore-F1
    if "bertscore" in metrics:
        try:
            import evaluate

            if args.task == "mt":
                lang = args.tgt_lang or "ro"
            else:
                lang = "en"
            bert_metric = evaluate.load("bertscore")
            print(f"[INFO] Computing BERTScore with lang={lang}")
            bert = bert_metric.compute(
                predictions=preds,
                references=refs,
                lang='en',  # implementation choice; consistent across tasks
                rescale_with_baseline=True,
            )
            f1_list = bert.get("f1", [])
            if f1_list:
                results["bertscore_f1"] = float(np.mean(f1_list))

                # Optional debugging: show the exact texts used, plus a few per-example scores.
                if getattr(args, "debug_metrics", False):
                    print(f"[DEBUG] BERTScore computed on {len(preds)} examples; showing up to 3.")
                    for idx in range(min(3, len(preds))):
                        f1_val = float(f1_list[idx]) if idx < len(f1_list) else float("nan")
                        pred_snip = preds[idx].replace("\n", " ")[:200]
                        ref_snip = refs[idx].replace("\n", " ")[:200]
                        print(f"[DEBUG]   Example {idx}: F1={f1_val:.4f}")
                        print(f"[DEBUG]     pred: {pred_snip}")
                        print(f"[DEBUG]     ref : {ref_snip}")
            else:
                results["bertscore_f1"] = float("nan")
        except Exception as e:  # pragma: no cover
            results["bertscore_f1"] = float("nan")
            print(f"[WARN] BERTScore computation failed: {e}")

    # Perplexity under the same model (conditional PPL).
    if "ppl" in metrics and ds is not None:
        try:
            if getattr(args, "is_decoder_only_model", False):
                ppl_vals = compute_causal_perplexities(
                    args, preds, refs, model, tokenizer, device, ds
                )
            else:
                ppl_vals = compute_seq2seq_perplexities(
                    args, preds, refs, model, tokenizer, device, ds
                )
            if len(ppl_vals) > 0:
                results["ppl_median"] = float(np.median(ppl_vals))
                results["ppl_mean"] = float(np.mean(ppl_vals))
            else:
                results["ppl_median"] = float("nan")
                results["ppl_mean"] = float("nan")
        except Exception as e:  # pragma: no cover
            results["ppl_median"] = float("nan")
            results["ppl_mean"] = float("nan")
            print(f"[WARN] PPL computation failed: {e}")

    return results


def compute_causal_perplexities(
    args: PipelineArgs,
    preds: List[str],
    refs: List[str],
    model,
    tokenizer,
    device: torch.device,
    ds,
    max_length: Optional[int] = None,
) -> List[float]:
    """
    Compute conditional perplexity for decoder-only models (e.g., Llama-Instruct).

    For summarization, we treat:
        prompt = get_src(ex)  (same as generation)
        target = pred summary
    and compute P(target | prompt).
    """
    model.eval()
    ppl_values: List[float] = []

    if args.task == "summ":
        source_field = args.source_field or "article"
        # Rebuild prompts as in generate_outputs (using the same template logic)
        prompts: List[str] = []
        for ex in ds:
            article = ex[source_field]
            if getattr(args, "is_decoder_only_model", False):
                prompt = build_summarization_prompt(args, tokenizer, article)
            else:
                prompt = article
            prompts.append(prompt)
    else:
        # For now, restrict causal PPL to summarization
        raise ValueError(f"Causal PPL not implemented for task={args.task}")

    n = min(len(preds), len(prompts))
    batch_size = min(4, int(args.batch_size/4))
    model_max_len = max_length if max_length is not None else getattr(args, "model_max_length", None)

    for i in range(0, n, batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_preds = preds[i : i + batch_size]

        input_ids_list = []
        labels_list = []
        for prompt, summary in zip(batch_prompts, batch_preds):
            # Encode prompt and summary separately
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True if model_max_len is not None else False,
                max_length=model_max_len,
                return_tensors="pt",
            )["input_ids"][0]
            with tokenizer.as_target_tokenizer() if hasattr(tokenizer, "as_target_tokenizer") else _nullcontext():
                summary_ids = tokenizer(
                    summary,
                    add_special_tokens=False,
                    truncation=True if model_max_len is not None else False,
                    max_length=model_max_len,
                    return_tensors="pt",
                )["input_ids"][0]

            # Concatenate prompt + summary
            full_input = torch.cat([prompt_ids, summary_ids], dim=0)
            # Labels: -100 for prompt tokens, summary_ids for summary tokens
            labels = torch.full_like(full_input, fill_value=-100)
            labels[len(prompt_ids) :] = summary_ids

            # Optionally truncate, but prefer to keep the full summary intact.
            if model_max_len is not None and full_input.shape[0] > model_max_len:
                total_len = full_input.shape[0]
                prompt_len = len(prompt_ids)
                summary_len = len(summary_ids)
                if summary_len >= model_max_len:
                    # If the summary itself is longer than max_len, keep the last max_len summary tokens
                    full_input = summary_ids[-model_max_len:]
                    labels = full_input.clone()
                else:
                    # Keep all of the summary and as much of the tail of the prompt as fits.
                    keep_prompt = model_max_len - summary_len
                    prompt_tail = prompt_ids[-keep_prompt:] if keep_prompt > 0 else prompt_ids.new_empty((0,), dtype=prompt_ids.dtype)
                    full_input = torch.cat([prompt_tail, summary_ids], dim=0)
                    labels = torch.full_like(full_input, fill_value=-100)
                    labels[keep_prompt:] = summary_ids

            input_ids_list.append(full_input)
            labels_list.append(labels)

        # Pad batch
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        max_len_batch = max(x.shape[0] for x in input_ids_list)
        batch_input = []
        batch_labels = []
        for inp, lab in zip(input_ids_list, labels_list):
            pad_len = max_len_batch - inp.shape[0]
            if getattr(tokenizer, "padding_side", "right") == "left":
                inp_padded = torch.cat([torch.full((pad_len,), pad_id, dtype=inp.dtype), inp], dim=0)
                lab_padded = torch.cat([torch.full((pad_len,), -100, dtype=lab.dtype), lab], dim=0)
            else:
                inp_padded = torch.cat([inp, torch.full((pad_len,), pad_id, dtype=inp.dtype)], dim=0)
                lab_padded = torch.cat([lab, torch.full((pad_len,), -100, dtype=lab.dtype)], dim=0)
            batch_input.append(inp_padded)
            batch_labels.append(lab_padded)

        batch_input_ids = torch.stack(batch_input, dim=0).to(device)
        batch_labels_ids = torch.stack(batch_labels, dim=0).to(device)
        attention_mask = (batch_input_ids != pad_id).long()

        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
            )
            from torch.nn import CrossEntropyLoss

            logits = outputs.logits  # [B, T, V]
            loss_fct = CrossEntropyLoss(reduction="none")
            B, T, V = logits.shape

            # Causal LM shift: logits at position t predict token at position t+1.
            shift_logits = logits[:, :-1, :]  # [B, T-1, V]
            shift_labels = batch_labels_ids[:, 1:]  # [B, T-1]

            # Per-token negative log-likelihoods.
            losses = loss_fct(
                shift_logits.reshape(-1, V),
                shift_labels.reshape(-1),
            ).reshape(B, T - 1)

            # Mask: only positions with valid labels (summary tokens) contribute.
            mask = (shift_labels != -100).float()  # [B, T-1]
            token_counts = mask.sum(dim=-1)
            safe_counts = token_counts.clone()
            zero_mask = safe_counts == 0
            safe_counts[zero_mask] = 1.0

            # Optional detailed debug for the first example in the first batch.
            if getattr(args, "debug_metrics", False) and not getattr(args, "_logged_causal_ppl_debug", False):
                setattr(args, "_logged_causal_ppl_debug", True)
                b0 = 0
                valid_pos = (mask[b0] > 0).nonzero(as_tuple=False).squeeze(-1)
                if valid_pos.numel() > 0:
                    pos0 = int(valid_pos[0].item())
                    token_id = int(shift_labels[b0, pos0].item())
                    # Decode a single-token string; safe even if not printable.
                    token_str = tokenizer.decode([token_id])
                    logits_pos = shift_logits[b0, pos0]  # [V]
                    probs_pos = torch.softmax(logits_pos, dim=-1)
                    topk_vals, topk_idx = torch.topk(probs_pos, k=5)
                    top_tokens = tokenizer.convert_ids_to_tokens(topk_idx.tolist())
                    # Entropy of the next-token distribution at this position (in nats).
                    entropy_pos = float(
                        -(probs_pos * probs_pos.clamp_min(1e-12).log()).sum().item()
                    )
                    per_token_nll = float(losses[b0, pos0].item())
                    sample_loss = float(
                        (losses[b0] * mask[b0]).sum().item() / (mask[b0].sum().item() + 1e-8)
                    )

                    print("[DEBUG] Causal PPL detailed example (decoder-only model)")
                    print(f"[DEBUG]   batch_index: {i + b0}")
                    prompt_snip = batch_prompts[b0].replace("\n", " ")[:200]
                    summary_snip = batch_preds[b0].replace("\n", " ")[:200]
                    print(f"[DEBUG]   prompt : {prompt_snip}")
                    print(f"[DEBUG]   summary: {summary_snip}")
                    print(f"[DEBUG]   first summary token id: {token_id}, token: {repr(token_str)}")
                    print(f"[DEBUG]   per-token NLL @pos {pos0}: {per_token_nll:.4f}")
                    print(f"[DEBUG]   entropy @pos {pos0} (nats): {entropy_pos:.4f}")
                    print("[DEBUG]   top-5 next-token probs at this position:")
                    for tid, tok, p in zip(topk_idx.tolist(), top_tokens, topk_vals.tolist()):
                        print(f"[DEBUG]     id={tid}, token={repr(tok)}, p={float(p):.4f}")
                    print(f"[DEBUG]   mean NLL over summary tokens: {sample_loss:.4f}")
                    print(f"[DEBUG]   PPL for this summary: {math.exp(sample_loss):.4f}")

            losses = (losses * mask).sum(dim=-1) / safe_counts
            losses[zero_mask] = float("inf")
            batch_ppl = torch.exp(losses).cpu().tolist()
            ppl_values.extend(batch_ppl)

        # Optional: clear CUDA cache between batches to reduce memory pressure
        if getattr(args, "empty_cache_between_batches", False) and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            except Exception:
                pass

    return ppl_values

def compute_seq2seq_perplexities(
    args: PipelineArgs,
    preds: List[str],
    refs: List[str],
    model,
    tokenizer,
    device: torch.device,
    ds,
    max_source_length: Optional[int] = None,
    max_target_length: Optional[int] = None,
) -> List[float]:
    """
    Compute conditional perplexity of preds given source inputs using the same
    basic scheme as standard seq2seq PPL:
        - encoder input = source text
        - decoder labels = predicted text (shifted by one token)

    For MT: P(prediction | source text).
    For summarization: P(summary | article).
    """
    model.eval()
    ppl_values: List[float] = []

    # Collect source texts in the same order as preds/refs were generated.
    src_texts: List[str] = []
    if args.task == "mt":
        src_key = args.src_lang or "en"
        for ex in ds:
            src_texts.append(ex["translation"][src_key])
    elif args.task == "summ":
        source_field = args.source_field or "article"
        for ex in ds:
            src_texts.append(ex[source_field])
    else:
        raise ValueError(f"Unsupported task for PPL: {args.task}")

    # Defensive: align lengths
    n = min(len(preds), len(src_texts))
    batch_size = min(64, int(args.batch_size/4))

    for i in range(0, n, batch_size):
        batch_src = src_texts[i : i + batch_size]
        batch_preds = preds[i : i + batch_size]

        # Encode encoder inputs = source texts
        max_len_src = max_source_length if max_source_length is not None else getattr(args, "model_max_length", None)
        enc = tokenizer(
            batch_src,
            padding=True,
            truncation=True if max_len_src is not None else False,
            max_length=max_len_src,
            return_tensors="pt",
        ).to(device)

        # Encode decoder outputs = predicted texts (as labels).
        # Use target-tokenizer context when available so that seq2seq models
        # (e.g., MBART) apply the correct language-specific special tokens.
        if hasattr(tokenizer, "as_target_tokenizer"):
            ctx = tokenizer.as_target_tokenizer()
        else:
            ctx = _nullcontext()
        with ctx:
            max_len_tgt = max_target_length if max_target_length is not None else getattr(args, "model_max_length", None)
            dec = tokenizer(
                batch_preds,
                padding=True,
                truncation=True if max_len_tgt is not None else False,
                max_length=max_len_tgt,
                return_tensors="pt",
            ).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask", None),
                labels=dec["input_ids"],
            )
            # outputs.logits: [B, L_dec, V]; labels: [B, L_dec]
            from torch.nn import CrossEntropyLoss

            logits = outputs.logits
            labels = dec["input_ids"]
            attention_out = dec.get("attention_mask", None)

            # Build mask over non-padding (and non-ignored) positions.
            if attention_out is not None:
                label_attention_mask = attention_out
            else:
                label_attention_mask = torch.ones_like(labels, device=device)

            # Set pad tokens to -100 so they are ignored by CrossEntropyLoss.
            labels = labels.clone()
            pad_id = tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100
            labels[label_attention_mask == 0] = -100

            loss_fct = CrossEntropyLoss(reduction="none")
            vocab_size = logits.shape[-1]
            # Use reshape instead of view to handle non-contiguous tensors
            logits_flat = logits.reshape(-1, vocab_size)
            labels_flat = labels.reshape(-1)
            losses = loss_fct(logits_flat, labels_flat).reshape(labels.shape)

            # Mask out padding positions and avoid div-by-zero
            mask = (labels != -100).float()
            token_counts = mask.sum(dim=-1)
            safe_counts = token_counts.clone()
            zero_mask = safe_counts == 0
            safe_counts[zero_mask] = 1.0

            # Optional detailed debug for the first example in the first batch.
            if getattr(args, "debug_metrics", False) and not getattr(args, "_logged_seq2seq_ppl_debug", False):
                setattr(args, "_logged_seq2seq_ppl_debug", True)
                b0 = 0
                valid_pos = (mask[b0] > 0).nonzero(as_tuple=False).squeeze(-1)
                if valid_pos.numel() > 0:
                    pos0 = int(valid_pos[0].item())
                    token_id = int(labels[b0, pos0].item())
                    token_str = tokenizer.decode([token_id])
                    logits_pos = logits[b0, pos0]  # [V]
                    probs_pos = torch.softmax(logits_pos, dim=-1)
                    topk_vals, topk_idx = torch.topk(probs_pos, k=5)
                    top_tokens = tokenizer.convert_ids_to_tokens(topk_idx.tolist())
                    entropy_pos = float(
                        -(probs_pos * probs_pos.clamp_min(1e-12).log()).sum().item()
                    )
                    per_token_nll = float(losses[b0, pos0].item())
                    sample_loss = float(
                        (losses[b0] * mask[b0]).sum().item() / (mask[b0].sum().item() + 1e-8)
                    )

                    print("[DEBUG] Seq2seq PPL detailed example (encoder-decoder model)")
                    print(f"[DEBUG]   batch_index: {i + b0}")
                    src_snip = batch_src[b0].replace("\n", " ")[:200]
                    pred_snip = batch_preds[b0].replace("\n", " ")[:200]
                    print(f"[DEBUG]   source : {src_snip}")
                    print(f"[DEBUG]   pred   : {pred_snip}")
                    print(f"[DEBUG]   first label token id: {token_id}, token: {repr(token_str)}")
                    print(f"[DEBUG]   per-token NLL @pos {pos0}: {per_token_nll:.4f}")
                    print(f"[DEBUG]   entropy @pos {pos0} (nats): {entropy_pos:.4f}")
                    print("[DEBUG]   top-5 next-token probs at this position:")
                    for tid, tok, p in zip(topk_idx.tolist(), top_tokens, topk_vals.tolist()):
                        print(f"[DEBUG]     id={tid}, token={repr(tok)}, p={float(p):.4f}")
                    print(f"[DEBUG]   mean NLL over decoded tokens: {sample_loss:.4f}")
                    print(f"[DEBUG]   PPL for this sequence: {math.exp(sample_loss):.4f}")

            losses = (losses * mask).sum(dim=-1) / safe_counts
            # For sequences with zero valid tokens, set loss to +inf so PPL is inf
            losses[zero_mask] = float("inf")
            batch_ppl = torch.exp(losses).cpu().tolist()
            ppl_values.extend(batch_ppl)

        # Optional: clear CUDA cache between batches to reduce memory pressure
        if getattr(args, "empty_cache_between_batches", False) and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            except Exception:
                pass

    return ppl_values


class _nullcontext:
    """Simple stand-in for contextlib.nullcontext (to avoid an extra import)."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def save_results(
    args: PipelineArgs,
    preds: List[str],
    refs: List[str],
    metrics: Dict[str, float],
) -> None:
    """Save predictions, references and metrics to output_dir."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Save plain-text predictions and references
    preds_path = os.path.join(args.output_dir, "predictions.txt")
    refs_path = os.path.join(args.output_dir, "references.txt")
    with open(preds_path, "w", encoding="utf-8") as f_pred:
        for line in preds:
            f_pred.write(line.replace("\n", " ") + "\n")
    with open(refs_path, "w", encoding="utf-8") as f_ref:
        for line in refs:
            f_ref.write(line.replace("\n", " ") + "\n")

    # Save metrics as a simple JSON
    import json

    metrics_payload = {
        "run_name": args.run_name,
        "task": args.task,
        "watermark_method": args.watermark_method,
        "unit_capacity": args.unit_capacity,
        "num_chunks": args.num_chunks,
        "metrics": metrics,
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f_json:
        json.dump(metrics_payload, f_json, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    set_random_seeds(args.random_seed)

    print(f"[INFO] Starting downstream pipeline: task={args.task}, run_name={args.run_name}")
    print(f"[INFO] Model: {args.model_name_or_path}")
    print(f"[INFO] Dataset: {args.dataset_name} ({args.dataset_config_name}), split={args.dataset_split}")
    print(f"[INFO] Watermark method: {args.watermark_method}, unit_capacity={args.unit_capacity}, num_chunks={args.num_chunks}")

    ds = load_dataset_for_task(args)

    # ds = ds.select(range(1000))
    model, tokenizer, device = load_model_and_tokenizer(args)

    if args.watermark_method == "none":
        preds, refs = generate_outputs(args, model, tokenizer, device, ds)
    else:
        # All watermarking methods (quantile_watermark, mpac, etc.) go through
        # the shared watermark generation path.
        preds, refs = generate_outputs_quantile(args, model, tokenizer, device, ds)

    # Pass ds so that PPL can be computed conditional on source texts.
    metrics = compute_metrics(args, preds, refs, model, tokenizer, device, ds)

    print(f"[INFO] Metrics: {metrics}")
    save_results(args, preds, refs, metrics)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
