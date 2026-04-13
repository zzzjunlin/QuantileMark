# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
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
import os
import random
import time
import torch

# HF classes

from datasets import load_dataset, IterableDataset

from torch import Tensor
from tokenizers import Tokenizer

from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    LlamaForCausalLM,
    LlamaTokenizer
)

from .data.lfqa import load_lfqa, load_lfqa_hf
from .data.essays import load_essays
from .data.wikitext import load_wikitext
from .data.humaneval import load_humaneval
from .data.code_contests import load_code_contests

MAX_GENERATIONS = int(10000)  # Hardcoded max length to avoid infinite loop
HF_TOKEN = os.environ['HF_ACCESS_TOKEN']

def normalize_fixed_message(fixed_message: str, msg_length: int) -> str:
    msg = (fixed_message or "").strip()
    if msg.startswith("0b"):
        msg = msg[2:]
    msg = msg.replace(" ", "").replace("_", "")
    if any(c not in "01" for c in msg):
        raise ValueError("fixed_message must be a binary string containing only 0/1.")
    if len(msg) > msg_length:
        raise ValueError(
            f"fixed_message length ({len(msg)}) exceeds message_length ({msg_length})."
        )
    return msg.zfill(msg_length)

def load_model(args):
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any(
        [(model_type in args.model_name_or_path) for model_type in ["t5", "T0"]]
    )
    args.is_decoder_only_model = any(
        [(model_type in args.model_name_or_path.lower()) for model_type in ["gpt", "opt", "bloom", "llama", "mistral"]]
    )
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model or True:
        args.is_decoder_only_model = True
        dtype = torch.float16 if args.load_fp16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=dtype, device_map="auto"
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.load_fp16:
            pass
        else:
            model = model.to(device)
    else:
        device = "cpu"
    model.eval()

    if args.is_decoder_only_model:
        padding_side = "left"
    else:
        raise NotImplementedError(
            "Need to check how to handle padding for seq2seq models when calling generate"
        )

    if "llama" in args.model_name_or_path and "hf" in args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(
            args.model_name_or_path, padding_side=padding_side
        )
        # tokenizer = AutoTokenizer.from_pretrained(
        #     args.model_name_or_path, padding_side=padding_side
        # )
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, padding_side=padding_side
        )
        # for GPT2 and other variants that do not have padding tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    args.model_max_length = model.config.max_position_embeddings

    return model, tokenizer, device


def _maybe_apply_chat_template(tokenizer, user_text: str, add_generation_prompt: bool = True) -> str:
    """If tokenizer provides a chat_template/apply_chat_template, wrap user_text as a single-turn chat.

    Returns the original text when no chat template is available.
    """
    try:
        has_template = getattr(tokenizer, "chat_template", None)
        has_apply = hasattr(tokenizer, "apply_chat_template")
        if has_template and has_apply and isinstance(user_text, str):
            messages = [{"role": "user", "content": user_text}]
            # tokenize=False returns a string; model will tokenize later
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
    except Exception:
        pass
    return user_text


def add_idx(example, idx):
    example.update({"idx": idx})
    return example


def load_hf_dataset(args):
    dataset_name, dataset_config_name = args.dataset_name, args.dataset_config_name

    if dataset_name == "lfqa":
        # Support both the original local LFQA JSONL and an HF-based
        # full LFQA dataset. Default is the original local version to
        # preserve backwards compatibility.
        lfqa_source = getattr(args, "lfqa_source", "local")
        if lfqa_source == "hf":
            dataset = load_lfqa_hf(args)
        else:
            dataset = load_lfqa(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": False,
                "input_col_name": "prefix",
                "ref_output_col_name": "gold_completion",
            }
        )
        # other args set within the load_lfqa function
    elif dataset_name == "wikitext":
        dataset = load_wikitext(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": True,
                "input_col_name": "text",
                "ref_output_col_name": None,
            }
        )
        # other args set within the load_wikitext function
    elif dataset_name == "essays":
        dataset = load_essays(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": False,
                "input_col_name": "instructions",
                "ref_output_col_name": "essays",
            }
        )
    elif dataset_name == "humaneval":
        # HumanEval code generation benchmark.
        # Use HF fields directly:
        #   - prompt: model input
        #   - canonical_solution: reference completion
        dataset = load_humaneval(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": False,
                "input_col_name": "prompt",
                "ref_output_col_name": "canonical_solution",
            }
        )
    elif dataset_name in ("code_contests", "deepmind/code_contests"):
        # DeepMind CodeContests: competitive programming problems with
        # long solutions. We expose a text prompt and one canonical
        # solution via a dedicated loader.
        dataset = load_code_contests(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": False,
                "input_col_name": "prompt",
                "ref_output_col_name": "canonical_solution",
            }
        )
    elif dataset_name == "humaneval":
        # HumanEval code generation benchmark.
        # Use HF fields directly:
        #   - prompt: model input
        #   - canonical_solution: reference completion
        dataset = load_humaneval(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": False,
                "input_col_name": "prompt",
                "ref_output_col_name": "canonical_solution",
            }
        )
    elif dataset_name == "cml_pile":
        subsets = [dataset_config_name]
        dataset = load_dataset(
            "./utils/data/cml_pile.py",
            subsets=subsets,
            streaming=args.stream_dataset,
            split=None,
            ignore_verifications=True,
        )[args.dataset_split]
        args.__dict__.update(
            {
                "truncate_input_for_prompt": True,
                "input_col_name": "text",
                "ref_output_col_name": None,
            }
        )
    else:
        if "c4" in dataset_name:
            dataset_name = "allenai/c4"
            if "realnewslike" in dataset_config_name:
                dataset_config_name = "en"
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            split=args.dataset_split,
            streaming=args.stream_dataset,
        )
        if "c4" in dataset_name:
            args.__dict__.update(
                {
                    "truncate_input_for_prompt": True,
                    "input_col_name": "text",
                    "ref_output_col_name": None,
                }
            )
            args.columns_to_remove = list(
                set(args.columns_to_remove + ["text", "timestamp", "url"])
            )
        elif "pile" in dataset_name:
            args.__dict__.update(
                {
                    "truncate_input_for_prompt": True,
                    "input_col_name": "text",
                    "ref_output_col_name": None,
                }
            )
            args.columns_to_remove = list(set(args.columns_to_remove + ["text", "meta"]))
        else:
            raise NotImplementedError(
                f"Dataset {dataset_name} not yet supported. Please add specs to load_hf_dataset function."
            )

    # add index to each row of dataset
    indexed_dataset = dataset.map(add_idx, batched=False, with_indices=True)

    # shuffle the first shuffle_buffer_size rows of streaming dataset, or whole dataset if not streaming
    # and take/select only the first n rows of the dataset (which caps the total number of pipeline iters possible)
    if isinstance(indexed_dataset, IterableDataset):
        shuffled_dataset = (
            indexed_dataset.shuffle(seed=args.shuffle_seed, buffer_size=args.shuffle_buffer_size)
            if args.shuffle_dataset
            else indexed_dataset
        )
        limited_dataset = (
            shuffled_dataset.take(args.limit_indices)
            if args.limit_indices is not None
            else shuffled_dataset
        )
    else:
        shuffled_dataset = (
            indexed_dataset.shuffle(seed=args.shuffle_seed)
            if args.shuffle_dataset
            else indexed_dataset
        )
        if args.limit_indices is not None:
            # Clamp to valid range to avoid IndexError when the requested
            # limit exceeds the dataset size (e.g., HumanEval has only
            # ~160 examples but limit_indices may be much larger).
            try:
                n_rows = len(shuffled_dataset)
                limit = min(args.limit_indices, n_rows)
            except Exception:
                limit = args.limit_indices
            limited_dataset = shuffled_dataset.select(range(limit))
            # Keep args.limit_indices consistent with what we actually used.
            args.limit_indices = limit
        else:
            limited_dataset = shuffled_dataset

    if args.limit_indices is None:
        try:
            args.limit_indices = len(limited_dataset)
        except Exception as e:
            # can't infer length of dataset, probably because it's an IterableDataset
            pass
    return limited_dataset


def check_input_lengths(
    example,
    min_sample_len=0,
    min_prompt_len=0,
    min_completion_len=0,
    max_input_len=None,
    max_new_tokens=None,
):
    orig_sample_length = example["orig_sample_length"]
    prompt_length = example["prompt_length"]
    real_completion_length = example["baseline_completion_length"]

    if max_input_len is not None:
        assert (
            max_new_tokens is not None
        ), "need to specify max_new_tokens if max_input_length is specified"

    conds = all(
        [
            orig_sample_length >= min_sample_len,
            prompt_length >= min_prompt_len,
            real_completion_length >= min_completion_len,
            (
                ((prompt_length + max_new_tokens) <= max_input_len)
                if max_input_len is not None
                else True
            ),
        ]
    )
    return conds


def check_output_lengths(example, min_output_len=0):
    # FIXME, maybe should check baseline completion length too
    no_wm_output_len = example["no_wm_output_length"]
    w_wm_output_len = example["w_wm_output_length"]
    conds = all(
        [
            no_wm_output_len >= min_output_len,
            w_wm_output_len >= min_output_len,
        ]
    )
    return conds


def tokenize_and_truncate(
    example: dict,
    input_col_name: str = "text",
    completion_length: int = None,
    prompt_length: int = None,
    hf_model_name: str = None,
    tokenizer=None,
    truncate_left=False,
    model_max_length=None,
    apply_chat_template: bool = False,
):
    """take hf dataset entry and preprocess it for completion by a model"""
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert input_col_name in example, f"expects {input_col_name} field to be present"
    # tokenize
    prompt_text = example[input_col_name]
    if apply_chat_template:
        prompt_text = _maybe_apply_chat_template(
            tokenizer, prompt_text, add_generation_prompt=True
        )
    # When a chat template has already been applied, it typically inserts BOS
    # and header tokens (e.g., <|begin_of_text|>). In that case we must avoid
    # adding extra special tokens again, otherwise we get duplicated BOS.
    inputs_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=not apply_chat_template,
    )["input_ids"]
    example.update({"untruncated_inputs": inputs_ids})

    if truncate_left:
        # truncate left
        inputs_ids = inputs_ids[:, -model_max_length:]
        if example["untruncated_inputs"].shape != inputs_ids.shape:
            print(
                "Input too long for model! ",
                "Left truncating under assumption that this is the prompt+output ",
                "to be fed to the *oracle* model",
            )
        example.update({"untruncated_inputs": inputs_ids})

    if (completion_length is not None) and (prompt_length is None):
        # leave at least one token as prefix # FIXME I think plus 1 since 0 is start tok
        slice_length = min(inputs_ids.shape[1] - 1, completion_length)
    elif (prompt_length is not None) and (completion_length is None):
        desired_comp_len = (inputs_ids.shape[1] - 1) - prompt_length
        slice_length = desired_comp_len if desired_comp_len > 0 else 0
    else:
        raise ValueError(
            (
                f"Can only tokenize and truncate based on either the desired prompt length or desired completion length,",
                f" but got completion_length:{completion_length},prompt_length:{prompt_length}",
            )
        )

    # truncate
    inputs_ids = inputs_ids[:, : inputs_ids.shape[1] - slice_length]
    # logic depending on special tokens for the model
    if "t5" in hf_model_name or "T0" in hf_model_name:
        inputs_ids[0, -1] = 1
    # else: pass
    example.update({"input_ids": inputs_ids})
    return example


def tokenize_only(
    example: dict,
    input_col_name: str = "text",
    ref_output_col_name: str = None,
    tokenize_ref_output: bool = False,
    hf_model_name: str = None,
    tokenizer=None,
    model_max_length=None,
    apply_chat_template: bool = False,
):
    """take hf dataset entry and preprocess it for completion by a model
    (but don't truncate) where the dataset optionally has a secondary column
    that is the reference output to be scored against"""

    """take hf dataset entry and preprocess it for completion by a model"""
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert input_col_name in example, f"expects {input_col_name} field to be present"
    if ref_output_col_name is not None:
        assert ref_output_col_name in example, f"expects {ref_output_col_name} field to be present"

    # tokenize input
    prompt_text = example[input_col_name]
    if apply_chat_template:
        prompt_text = _maybe_apply_chat_template(
            tokenizer, prompt_text, add_generation_prompt=True
        )
    # Same reasoning as in tokenize_and_truncate: if we already wrapped the
    # text in a chat template, do not inject another BOS/special tokens.
    input_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=model_max_length,
        add_special_tokens=not apply_chat_template,
    )["input_ids"]

    example.update({"input_ids": input_ids})

    if tokenize_ref_output:
        # NOTE not sure this logic is useful/required
        if ref_output_col_name is not None:
            # tokenize ref output (do NOT apply chat template to reference output)
            ref_output_ids = tokenizer(
                example[ref_output_col_name],
                return_tensors="pt",
                truncation=True,
                max_length=model_max_length,
            )["input_ids"]

        tokd_input_len, tokd_ref_output_length = input_ids.shape[1], ref_output_ids.shape[1]
        if tokd_input_len + tokd_ref_output_length > model_max_length:
            # truncate the ref output
            original_ref_output_len = tokd_ref_output_length
            ref_output_ids = ref_output_ids[:, : model_max_length - tokd_input_len]
            if original_ref_output_len != ref_output_ids.shape[1]:
                print(
                    "Right truncating output, input+ref output too long for model. "
                    "Note, since this is generation time truncating the reference doesn't affect anything really."
                )
        example.update({"ref_output_ids": ref_output_ids})

    # logic depending on special tokens for the model
    if "t5" in hf_model_name or "T0" in hf_model_name:
        raise NotImplementedError("T5 style model not yet supported")

    return example


def tokenize_for_generation(
    example: dict,
    max_new_tokens: int = None,
    min_prompt_tokens: int = None,
    hf_model_name: str = None,
    tokenizer: Tokenizer = None,
    args: dict = None,
):
    # preprocessing, generation & scoring
    # HF Datasets may pass a LazyRow or other mapping type here; ensure we
    # always work with a plain dict representing a *single* example.
    if not isinstance(example, dict):
        try:
            example = dict(example)
        except Exception as e:
            raise AssertionError("Expect no batch dimension currently!") from e
    # Decide whether to wrap raw text with a chat template (for chat-style models).
    # Defaults to True (previous behavior) if the flag is absent.
    use_chat_template = True
    if args is not None and hasattr(args, "apply_chat_template"):
        # Explicit cast since argparse with str2bool may leave a string.
        use_chat_template = bool(getattr(args, "apply_chat_template"))
    # example['instructions'] = "<s>[INST] Write a complete essay with an introduction, main body, and conclusion following the below instructions.[/INST]" \
    #                   + example['instructions']
    # example['text'] = "<s>[INST] Complete the following news article: [/INST]" + example['text']
    if not args.truncate_input_for_prompt:
        tokenize_ref_output = True  # NOTE, note really sure how necessary this is
        # preprocess for model generation/completion
        example = tokenize_only(
            example,
            input_col_name=args.input_col_name,
            ref_output_col_name=args.ref_output_col_name,
            hf_model_name=hf_model_name,
            tokenizer=tokenizer,
            model_max_length=args.model_max_length,
            tokenize_ref_output=tokenize_ref_output,
            apply_chat_template=use_chat_template,
        )
        # Parse the results of tokenization. Decode the exact prompt tokens
        # (keep special tokens so downstream sees the chat template if present).
        re_decoded_input = tokenizer.batch_decode(example["input_ids"], skip_special_tokens=False)[0]
        decoded_baseline_completion = example[args.ref_output_col_name]
        prompt_len = example["input_ids"].shape[1]
        baseline_completion_len = example["ref_output_ids"].shape[1]
        full_sample_len = prompt_len + baseline_completion_len
        # for now, remove this here, since it's not used downstream
        example.pop("ref_output_ids")
    else:
        # preprocess for model generation/completion
        example = tokenize_and_truncate(
            example,
            completion_length=max_new_tokens,
            prompt_length=min_prompt_tokens,
            hf_model_name=hf_model_name,
            tokenizer=tokenizer,
            apply_chat_template=use_chat_template,
        )
        # Logic to parse the results of tokenzation and splitting to
        # construct string versions of the prompt and baseline completion
        inputs = example["input_ids"]
        prompt_len = inputs.shape[1]
        # for isolating the "gold" baseline completion
        untruncated_inputs = example.pop("untruncated_inputs")
        full_sample_len = untruncated_inputs.shape[1]
        # decode the preprocessed input to store for audit (keep special tokens)
        re_decoded_input = tokenizer.batch_decode(inputs, skip_special_tokens=False)[0]
        # also decode the original suffix of the input for audit as the baseline
        baseline_completion_tokens = untruncated_inputs[:, inputs.shape[-1] :]
        decoded_baseline_completion = tokenizer.batch_decode(
            baseline_completion_tokens, skip_special_tokens=True
        )[0]
        baseline_completion_len = full_sample_len - prompt_len

    example.update(
        {
            "truncated_input": re_decoded_input,
            "baseline_completion": decoded_baseline_completion,
            "orig_sample_length": full_sample_len,
            "prompt_length": prompt_len,
            "baseline_completion_length": baseline_completion_len,
        }
    )
    return example


def collate_batch(input_ids: list, collator: DataCollatorWithPadding = None):
    """Collate batch of input_ids into padded tensors and return both ids and attention_mask.

    NOTE: We must return attention_mask and pass it to model.generate().
    Otherwise, when pad_token==eos_token (e.g., GPT2-like), HF cannot infer
    the mask reliably and will warn and possibly behave unexpectedly.
    """
    # Older code expected each element to be a tensor of shape [1, L].
    # With some datasets / HF versions, `input_ids` may instead be a list
    # of 1D token-id lists, or even a nested list-of-lists (e.g. [[...]])
    # when a [1, L] tensor was serialized. Handle all cases robustly.
    if hasattr(input_ids[0], "shape"):
        assert (
            input_ids[0].shape[0] == 1 and input_ids[0].shape[1] > 0
        ), "expecting batch dimension of each tensor to be 1"
        # remove batch dimension for each tensor
        flat_input_ids = [x.squeeze(0) for x in input_ids]
    else:
        # Normalize possible nested python lists coming from HF datasets.
        # - If each element is already a 1D list[int], keep as-is.
        # - If each element is a 2D list (e.g. [[1,2,3]]), flatten the
        #   leading singleton dimension to get [1,2,3].
        def _normalize(seq):
            if isinstance(seq, (list, tuple)) and seq:
                if isinstance(seq[0], (list, tuple)):
                    # Flatten one level (handles [[...]] or [[...],[...]])
                    return [tok for sub in seq for tok in sub]
            return seq

        flat_input_ids = [_normalize(seq) for seq in input_ids]

    batch = collator({"input_ids": flat_input_ids})
    # Ensure attention_mask exists
    if "attention_mask" not in batch:
        # If collator/tokenizer did not provide, create mask where non-pad tokens are 1
        pad_id = getattr(collator.tokenizer, "pad_token_id", None) if collator else None
        if pad_id is not None:
            batch["attention_mask"] = (batch["input_ids"] != pad_id).long()
    return batch


def generate(
    examples,
    data_collator=None,
    generate_without_watermark=None,
    generate_with_watermark=None,
    watermark_processor=None,
    tokenizer=None,
    device=None,
    args=None,
):
    batch = collate_batch(input_ids=examples["input_ids"], collator=data_collator)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Sample multi-bit message: a batch will have the same message due to how the watermark processor operates.
    # This won't be an issue hopefully when the number of samples is adequately large enough (~500)
    msg_length = args.message_length
    fixed_message = getattr(args, "fixed_message", None)
    embedded_message = getattr(args, "embedded_message", None)
    watermark_type = getattr(args, "watermark_type", None)
    if fixed_message:
        msg_binary = normalize_fixed_message(fixed_message, msg_length)
        if getattr(args, "zero_bit", False) and set(msg_binary) != {"0"}:
            raise ValueError("zero_bit requires fixed_message to be all zeros.")
        msg_encoded = msg_binary
    elif embedded_message and watermark_type in ("quantile", "quantile_black"):
        # Quantile pipelines choose a per-run message at setup time.
        msg_binary = str(embedded_message)
        msg_encoded = msg_binary
    elif args.zero_bit:
        msg_binary = "0"
        msg_encoded = "0"
    else:
        use_ecc = False
        msg_binary, msg_encoded = sample_message(msg_length, use_ecc)
        # msg_binary = msg_encoded = msg_length * "0"


    # watermark_processor.set_message(msg_binary)
    watermark_processor.set_message(msg_encoded)
    print(f"Binary msg:\n{msg_binary}")
    print(f"Binary encoded msg:\n{msg_encoded}")
    print(f"Converted msg:\n{watermark_processor.converted_message}")
    messages = [msg_binary] * len(examples['input_ids'])


    with torch.no_grad():
        if args.generation_seed is not None:
            torch.manual_seed(args.generation_seed)
        s_time = time.time()
        output_without_watermark = generate_without_watermark(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        non_wm_time = time.time() - s_time

        if args.generation_seed is not None:
            torch.manual_seed(args.generation_seed)
        s_time = time.time()
        output_with_watermark = generate_with_watermark(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        wm_time = time.time() - s_time

        sampled_positions = watermark_processor.flush_position()
        watermark_processor.position_increment = 0

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:, input_ids.shape[-1] :]
        output_with_watermark = output_with_watermark[:, input_ids.shape[-1] :]

    decoded_output_without_watermark = tokenizer.batch_decode(
        output_without_watermark, skip_special_tokens=True
    )
    decoded_output_with_watermark = tokenizer.batch_decode(
        output_with_watermark, skip_special_tokens=True
    )

    # Get batch size
    batch_size = len(examples['input_ids'])

    # Ensure sampled_positions has the correct length for batch processing
    if len(sampled_positions) == 1 and batch_size > 1:
        # If watermark processor returns single value, replicate for batch
        sampled_positions = sampled_positions * batch_size
    elif len(sampled_positions) != batch_size:
        # If length mismatch, pad or truncate to match batch size
        sampled_positions = (sampled_positions + [""] * batch_size)[:batch_size]

    # Compute effective continuation lengths: count tokens up to the first EOS (exclusive),
    # ignoring any right-side padding. This avoids over-counting repeated EOS tokens used for padding.
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    def effective_lengths(seq_2d):
        B, T = seq_2d.shape
        device = seq_2d.device
        if pad_id is not None:
            nonpad_len = (seq_2d != pad_id).long().sum(dim=1)
        else:
            nonpad_len = torch.full((B,), T, dtype=torch.long, device=device)
        if eos_id is not None:
            idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            eos_mask = (seq_2d == eos_id)
            first_eos_pos = torch.where(eos_mask, idx, torch.full_like(idx, T)).min(dim=1).values
            eff_len = torch.minimum(nonpad_len, first_eos_pos)
        else:
            eff_len = nonpad_len
        return eff_len

    eff_len_no_wm = effective_lengths(output_without_watermark)
    eff_len_w_wm = effective_lengths(output_with_watermark)

    examples.update(
        {
            "no_wm_output": decoded_output_without_watermark,
            "w_wm_output": decoded_output_with_watermark,
            "sampled_positions": sampled_positions,
            "message": messages,
            "no_wm_output_length": eff_len_no_wm.tolist(),
            "w_wm_output_length": eff_len_w_wm.tolist(),
            'wm_encoding_time': [wm_time] * batch_size,
            'non_wm_encoding_time': [non_wm_time] * batch_size
        }
    )
    if watermark_processor.spike_entropies is not None:
        examples["spike_entropies"] = watermark_processor._get_and_clear_stored_spike_ents()
        examples["spike_entropies"] = [
            ents[:num_toks]
            for ents, num_toks in zip(examples["spike_entropies"], examples["w_wm_output_length"])
        ]

    # Free large tensors and optionally clear CUDA cache to mitigate fragmentation/OOM
    try:
        del output_without_watermark
        del output_with_watermark
        del input_ids
        if attention_mask is not None:
            del attention_mask
    except Exception:
        pass
    try:
        if getattr(args, 'empty_cache_between_batches', True) and torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    except Exception:
        pass

    return examples

try:
    from reedmuller import reedmuller
except:
    print("Error loading error correcting code module")

def sample_message(msg_length, use_ecc, ecc_params=None):
    msg_decimal = random.getrandbits(msg_length)
    msg_binary = format(msg_decimal, f"0{msg_length}b")
    if use_ecc:
        rm = reedmuller.ReedMuller(2, 5)
        msg_encoded = ''.join(map(str, rm.encode(list(map(int, msg_binary)))))
    else:
        msg_encoded = msg_binary

    return msg_binary, msg_encoded
