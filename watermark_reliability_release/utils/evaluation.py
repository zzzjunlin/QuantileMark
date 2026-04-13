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

import torch
import numpy as np
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from utils.generation import tokenize_and_truncate, collate_batch
from metrics.repetition_diversity import (
    measure_repetition_and_diversity,
    dummy_rep_div_result,
)
from metrics.p_sp import evaluate_p_sp
from metrics.detect_retrieval import detect_retrieval
from metrics.coherence import get_coherence_score
from metrics.mauve import get_mauve_score
from utils.hypothesis_testing import (
    chi_squared_runs_test,
    F_succ_T_runs_dummy_dict_w_bins,
    F_succ_T_runs_dummy_dict_no_bins,
    T_and_F_runs_dummy_dict_w_bins,
    T_and_F_runs_dummy_dict_no_bins,
)

from mb_watermark_processor import WatermarkDetector
# from watermark_processor import WatermarkDetector

# These areguments are ignored when doing checks between meta file and cmdline args
NO_CHECK_ARGS = [
    "evaluation_metrics",
    "verbose",
    "wandb",
    # Newer evaluation-only knobs that may differ between runs without
    # requiring regeneration of the base dataset.
    "zscore_T_list",
    "wandb_entity",
    "input_dir",
    "output_dir",
    "run_name",
    "overwrite_output_file",
    "overwrite_args",
    "limit_rows",
    "concat_rows",
    "max_prefix_length",
]


def conditional_no_check_args(no_check_args, evaluation_metrics, args):
    if "ppl" not in evaluation_metrics:
        no_check_args.append("oracle_model_name_or_path")
        no_check_args.append("load_fp16")
        no_check_args.append("ppl_batch_size")

    return no_check_args


# Series of configuration variables for the evaluation script
# These are the metrics we support
SUPPORTED_METRICS = [
    "z-score",
    # "windowed-z-score",
    # "run-len-chisqrd",
    "ppl",
    "api-judge-5",
    # "diversity",
    # "repetition",
    # "p-sp",
    # "coherence",
    # "mauve",
    # "detect-retrieval",
    # "detectgpt",
]

# These are the output text columns we want to compute metrics on
OUTPUT_TEXT_COLUMN_NAMES = [
    "baseline_completion",
    "no_wm_output",
    "w_wm_output",
    "w_wm_output_attacked",
]

# etc for other evaluation types
ZSCORE_TEXT_COLUMN_NAMES = OUTPUT_TEXT_COLUMN_NAMES
RUN_LEN_CHISQRD_TEXT_COLUMN_NAMES = OUTPUT_TEXT_COLUMN_NAMES
REPETITION_TEXT_COLUMN_NAMES = OUTPUT_TEXT_COLUMN_NAMES
# note the convention of including the input as 0th column
COHERENCE_TEXT_COLUMN_NAMES = ["truncated_input"] + OUTPUT_TEXT_COLUMN_NAMES

# These are the column pairs we want to compute p-sp for
OUTPUT_TEXT_PAIR_COLUMN_NAMES = [
    ["baseline_completion", "no_wm_output"],
    ["baseline_completion", "w_wm_output"],
    ["baseline_completion", "w_wm_output_attacked"],
    ["no_wm_output", "w_wm_output"],
    ["w_wm_output", "w_wm_output_attacked"],
]

P_SP_TEXT_PAIR_COLUMN_NAMES = OUTPUT_TEXT_PAIR_COLUMN_NAMES
MAUVE_TEXT_PAIR_COLUMN_NAMES = OUTPUT_TEXT_PAIR_COLUMN_NAMES


ROC_TEST_STAT_SUFFIXES = [
    # "custom_metric",
    "z_score",
    # "win20-1_z_score",
    # "win40-1_z_score",
    # "winmax-1_z_score",
    # "run_len_chisqrd_statistic",
    # "retrieval_score",
    # "detectgpt_score_100_z",
    # "detectgpt_score_100_d",
]

FILTER_BY_COLUMNS = ["baseline_completion", "w_wm_output"]


def concat_rows(examples, tokenizer=None, args=None):
    # concat the rows (there will be k rows per example)
    # just joining the strings by a space
    for col_name in examples.keys():
        if col_name in OUTPUT_TEXT_COLUMN_NAMES:
            examples[col_name] = " ".join(examples[col_name])
        else:
            # # check that all other columns have len args.concat_rows
            # if len(examples[col_name]) != args.concat_rows:
            #     # append None to the col to make it the right length
            #     examples[col_name] = examples[col_name] + [None] * (
            #         args.concat_rows - len(examples[col_name])
            #     )
            # EH for now just set them to be the first element of their respective column
            # quite mangled...
            examples[col_name] = examples[col_name][0]

    # Now, update the lengths
    for col_name in OUTPUT_TEXT_COLUMN_NAMES:
        if col_name in examples:
            examples[f"{col_name}_length"] = len(
                tokenizer(examples[col_name], add_special_tokens=False)["input_ids"]
            )
    return examples


def load_tokenizer(args):
    model_name = args.model_name_or_path
    print(f"Loading tokenizer for: {model_name}")
    if "llama" in model_name and False:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = 0  # unk
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def load_detector(args):
    if "llama" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.pad_token_id = 0  # unk
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    device = "cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"
    wm_kwargs = {
        'use_position_prf': args.use_position_prf,
        'use_fixed_position': args.use_fixed_position,
        'code_length': args.code_length
    }
    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        seeding_scheme=args.seeding_scheme,
        device=device,
        tokenizer=tokenizer,
        z_threshold=args.detection_z_threshold,
        normalizers=args.normalizers,
        ignore_repeated_ngrams=args.ignore_repeated_ngrams,
        message_length=args.message_length,
        base=args.base,
        **wm_kwargs
    )

    return watermark_detector


def compute_z_score(
    example,
    text_column_name=None,
    watermark_detector=None,
    args=None,
    window_size=None,
    window_stride=None,
):
    # for now, don't get the green token mask
    # if we're using normalizers
    return_green_token_mask = args.return_green_token_mask
    if args.normalizers != []:
        return_green_token_mask = None

    watermark_detector.position_increment = 0
    input_text = example[text_column_name]

    # Optionally include prompt for detection (important for quantile watermark)
    prompt_len = 0
    if args.include_prompt_in_detection and "truncated_input" in example and "prompt_length" in example:
        # Prepend prompt to the generated text
        prompt_text = example["truncated_input"]
        input_text = prompt_text + input_text
        prompt_len = example["prompt_length"]

    if example.get("sampled_positions", None) is None:
        # entering dummy values
        example['sampled_positions'] = "0000"
    error = False
    measured_decoding_time = None
    if input_text == "":
        error = True
    else:
        debug = args.debug
        if debug:
            t0 = time.time()
            score_dict = watermark_detector.detect(
                input_text,
                window_size=window_size,
                window_stride=window_stride,
                return_green_token_mask=return_green_token_mask,
                return_prediction=False,  # this conversion to "decision" only desired in demo context
                convert_to_float=True,  # this helps with integrity under NaNs
                # Always enable per-prefix z-score computation when we plan to
                # log multi-T statistics, even if compute_scores_at_T was
                # disabled on the command line.
                return_z_at_T=(args.compute_scores_at_T or bool(getattr(args, "zscore_T_list", []))),
                message=example['message'],
                col_name=text_column_name,
                position=example['sampled_positions'],
                prompt_len=prompt_len,
            )
            measured_decoding_time = time.time() - t0
        else:
            try:
                t0 = time.time()
                score_dict = watermark_detector.detect(
                    input_text,
                    window_size=window_size,
                    window_stride=window_stride,
                    return_green_token_mask=return_green_token_mask,
                    return_prediction=False,  # this conversion to "decision" only desired in demo context
                    convert_to_float=True,  # this helps with integrity under NaNs
                    return_z_at_T=(args.compute_scores_at_T or bool(getattr(args, "zscore_T_list", []))),
                    message=example['message'],
                    col_name=text_column_name,
                    position=example['sampled_positions'],
                    prompt_len=prompt_len,
                )
                measured_decoding_time = time.time() - t0
            except ValueError as err:
                print(err)
                error = True
            except Exception as err:
                print(err)
                error = True

    if error:
        problem_text = f"'{input_text[:40]} {'[...]' if len(input_text) > 40 else ''}'"
        if args.verbose:
            print(
                f"{(f'Windowed({window_size})' if window_size else '')} Detection error on text: {problem_text}"
            )
        # "Error string too short to compute metrics"
        score_dict = watermark_detector.dummy_detect(
            return_prediction=False,
            return_green_token_mask=return_green_token_mask,
            return_z_at_T=(args.compute_scores_at_T or bool(getattr(args, "zscore_T_list", []))),
        )

    # current detect logic causes issues bc it only reports this sometimes
    score_dict.pop("confidence", None)
    # We don't use position_acc downstream and it causes schema issues when
    # mixing different watermark types / detectors. Drop it here to avoid
    # creating `<col>_position_acc` columns in the dataset.
    score_dict.pop("position_acc", None)
    # Ensure decoding_time is consistently available for logging / aggregation,
    # even for detectors that do not populate it (e.g., some watermark types).
    dt = score_dict.get("decoding_time", None)
    dt_ok = False
    try:
        dt_ok = np.isfinite(float(dt))
    except Exception:
        dt_ok = False
    if not dt_ok:
        try:
            if measured_decoding_time is not None and np.isfinite(float(measured_decoding_time)):
                score_dict["decoding_time"] = float(measured_decoding_time)
            else:
                score_dict.setdefault("decoding_time", float("nan"))
        except Exception:
            score_dict.setdefault("decoding_time", float("nan"))

    # Multi-T z-score / bit-accuracy projection for this column (non-windowed only).
    # We operate on the raw score_dict before column-name prefixing.
    zscore_T_list = getattr(args, "zscore_T_list", []) or []
    if zscore_T_list and window_size is None:
        # Helper to robustly extract a scalar at prefix length T from various
        # container types returned by different detectors.
        def _get_at_T(seq, T):
            if seq is None:
                return float("nan")
            try:
                import torch
            except Exception:  # pragma: no cover - torch always present in pipeline env
                torch = None  # type: ignore[assignment]

            # Mapping form: {T: value}
            if isinstance(seq, dict):
                val = seq.get(T, float("nan"))
                try:
                    return float(val)
                except Exception:
                    return float("nan")

            # 1D tensor / list / ndarray form indexed by position (1-based T)
            if torch is not None and isinstance(seq, torch.Tensor):
                if seq.ndim == 0:
                    return float("nan")
                if T <= 0 or T > int(seq.shape[0]):
                    return float("nan")
                try:
                    return float(seq[T - 1].item())
                except Exception:
                    return float("nan")

            try:
                # list, tuple, numpy array
                n = len(seq)
                if T <= 0 or T > n:
                    return float("nan")
                return float(seq[T - 1])
            except Exception:
                return float("nan")

        z_seq = score_dict.get("z_score_at_T", None)
        bit_seq = score_dict.get("bit_acc_at_T", None)
        col_prefix = text_column_name + "_"
        for T in zscore_T_list:
            if z_seq is not None:
                example[f"{col_prefix}z_T{T}"] = _get_at_T(z_seq, T)
            if bit_seq is not None:
                example[f"{col_prefix}bit_acc_T{T}"] = _get_at_T(bit_seq, T)

    # replace every key name in score dict with the text_column_name + key name
    # and then add them to the example dict
    prefixed_score_dict = {
        text_column_name
        + (f"_win{window_size}-{window_stride}" if window_size else "")
        + "_"
        + k: v
        for k, v in score_dict.items()
    }
    example.update(prefixed_score_dict)
    return example


def compute_z_scores(example, watermark_detector=None, args=None):
    # this just iterates the z-score function over the columns we want to compute z-scores for
    gt_positions = None
    corrupted_positions = None

    for col_name in ZSCORE_TEXT_COLUMN_NAMES:
        if col_name in example:
            example = compute_z_score(
                example, text_column_name=col_name, watermark_detector=watermark_detector, args=args
            )
        if "w_wm_output_attacked" in example and col_name == "w_wm_output":
            gt_positions = example.get('w_wm_output_sampled_positions', '')
        if "w_wm_output_attacked" in example and col_name == "w_wm_output_attacked":
            corrupted_positions = example.get('w_wm_output_attacked_sampled_positions', '')

    # Only compute position match for watermark types that track positions
    # (not quantile / quantile_black / StealthInk)
    if "w_wm_output_attacked" in example:
        # Check if watermark type supports position tracking
        if (
            args
            and hasattr(args, "watermark_type")
            and args.watermark_type in ["quantile", "quantile_black", "stealthink", "unbiased"]
        ):
            # Quantile / QuantileBlack / StealthInk watermarks don't track positions, skip this metric
            example['corrupted_position_match'] = None
        elif gt_positions is not None and corrupted_positions is not None and gt_positions and corrupted_positions:
            # For multibit watermark with position tracking
            match_cnt = sum([x == y for x, y in zip(corrupted_positions, gt_positions)])
            if len(gt_positions) > 0:
                example['corrupted_position_match'] = match_cnt / len(gt_positions)
            else:
                example['corrupted_position_match'] = None
        else:
            example['corrupted_position_match'] = None

    return example


def compute_windowed_z_scores(example, watermark_detector=None, args=None):
    # this iterates the z-score function over the columns we want to compute z-scores for
    for col_name in ZSCORE_TEXT_COLUMN_NAMES:
        if col_name in example:
            for window_size in args.window_settings:
                example = compute_z_score(
                    example,
                    text_column_name=col_name,
                    watermark_detector=watermark_detector,
                    args=args,
                    window_size=window_size,
                    window_stride=1,
                )
    return example


def compute_run_len_chisqrd_stat(
    example,
    text_column_name=None,
    bool_arr_suffix=None,
    bool_arr=None,
    watermark_detector=None,  # unused under the "z-score required to be run first" assumption
    args=None,
    force_error=False,
):
    if bool_arr is not None:
        bool_array = bool_arr
    else:
        bool_array_col_name = text_column_name + bool_arr_suffix
        bool_array = example[bool_array_col_name]
    if isinstance(bool_array, list):
        bool_array = np.array(bool_array)

    run_len_kwargs = dict(
        bool_arr=bool_array,
        succ_prob=1 - args.gamma,  # this applies for both variants
        variant=args.run_len_chisqrd_variant,
        bin_spec=args.run_len_chisqrd_bin_spec,
        verbose=False,  # likely never in this context
        invert_bools=False,  # legacy
        return_bin_counts=False,  # debugging only, may not work currently
        mask_zeros=args.run_len_chisqrd_mask_zeros,
        mask_leading_bins=args.run_len_chisqrd_mask_leading_bins,
        diy=False,  # legacy
        lambda_=args.run_len_chisqrd_lambda,
        return_dict=True,  # always in this context
    )

    error = True if force_error else False
    try:
        score_dict = chi_squared_runs_test(**run_len_kwargs)
    except Exception as e:
        print(e)
        error = True
    if error:
        print(f"Run length test error, got: '{bool_array}'")
        if run_len_kwargs["variant"] == "F_succ_T_runs":
            if run_len_kwargs["return_bin_counts"]:
                score_dict = F_succ_T_runs_dummy_dict_w_bins
            else:
                score_dict = F_succ_T_runs_dummy_dict_no_bins
        elif run_len_kwargs["variant"] == "T_and_F_runs":
            if run_len_kwargs["return_bin_counts"]:
                score_dict = T_and_F_runs_dummy_dict_w_bins
            else:
                score_dict = T_and_F_runs_dummy_dict_no_bins
        else:
            raise ValueError("Unknown run length test variant and return_bin_counts setting")

    # replace every key name in score dict with the text_column_name + key name
    # and then add them to the example dict
    score_dict = {text_column_name + "_run_len_chisqrd_" + k: v for k, v in score_dict.items()}
    example.update(score_dict)

    return example


def compute_run_len_chsqrd_stats(
    example,
    watermark_detector=None,
    args=None,
    bool_arr_suffix="_green_token_mask",
    score_suffix="_run_len_chisqrd_statistic",
):
    # this just iterates the run_len_chisqrd function over the columns we want to compute stats for
    for col_name in RUN_LEN_CHISQRD_TEXT_COLUMN_NAMES:
        if col_name in example:
            if args.compute_scores_at_T:
                full_bool_arr = example[f"{col_name}{bool_arr_suffix}"]
                len_sequence = len(full_bool_arr)
                if len_sequence < 1:
                    force_error = True
                    full_bool_arr = [None]  # to cause loop to happen
                    len_sequence = 1
                else:
                    force_error = False
                stats_at_T = []
                for t in range(1, len_sequence + 1):
                    bool_arr = full_bool_arr[:t]
                    example = compute_run_len_chisqrd_stat(
                        example,
                        bool_arr=bool_arr,  # this overrides the normal access of the bool_arr
                        text_column_name=col_name,
                        bool_arr_suffix=bool_arr_suffix,
                        watermark_detector=watermark_detector,
                        args=args,
                        force_error=force_error,
                    )
                    stats_at_T.append(example[f"{col_name}{score_suffix}"])
                example[f"{col_name}{score_suffix}_at_T"] = stats_at_T
            else:
                example = compute_run_len_chisqrd_stat(
                    example,
                    text_column_name=col_name,
                    bool_arr_suffix=bool_arr_suffix,
                    watermark_detector=watermark_detector,
                    args=args,
                )
    return example


def load_oracle_model(args):
    oracle_model_name = args.oracle_model_name_or_path
    print(f"Loading oracle model: {oracle_model_name}")
    if args.load_fp16:
        oracle_model = AutoModelForCausalLM.from_pretrained(
            oracle_model_name, torch_dtype=torch.float16, device_map="auto"
        )
    else:
        oracle_model = AutoModelForCausalLM.from_pretrained(oracle_model_name)
    if "llama" in oracle_model_name:
        oracle_tokenizer = AutoTokenizer.from_pretrained(oracle_model_name)
        oracle_model.config.pad_token_id = oracle_tokenizer.pad_token_id = 0  # unk
        oracle_model.config.bos_token_id = 1
        oracle_model.config.eos_token_id = 2
    else:
        oracle_tokenizer = AutoTokenizer.from_pretrained(oracle_model_name)
    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not args.load_fp16:
            oracle_model = oracle_model.to(device)
    else:
        device = "cpu"
    oracle_model.eval()

    return oracle_model, oracle_tokenizer, device


from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast


def opt_unpooled_loss(logits, labels, model):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))
    loss = loss.reshape(shift_logits.shape[:-1])
    # compute the mean for each elm in batch where the label is not pad
    # we assume the losses are zero for pad indices
    loss = torch.sum(loss, dim=-1) / torch.sum(shift_labels != -100, dim=-1)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
    )


UNPOOL_FN_TABLE = {
    "opt": opt_unpooled_loss,
}


def get_unpool_fn(model_name):
    """Return an unpooling fn for decoder-only causal LMs.

    The current implementation (opt_unpooled_loss) works for most decoder-only
    models (OPT, LLaMA, Qwen, Mistral, GPT-like), since it simply applies the
    standard shift and per-token CE reduction.
    """
    name = model_name.lower()
    if any(k in name for k in ["opt", "llama", "qwen", "mistral", "gpt", "baichuan", "yi", "gemma", "chatglm"]):
        return UNPOOL_FN_TABLE["opt"]
    # Fallback: use the generic decoder-only implementation
    return UNPOOL_FN_TABLE["opt"]


def compute_ppl_batch(
    prefix_and_output_text=None,
    output_text=None,
    oracle_model_name=None,
    oracle_model=None,
    oracle_tokenizer=None,
    data_collator=None,
):
    inputs = []
    labels = []
    for idx in range(len(prefix_and_output_text)):
        tokd_prefix = tokenize_and_truncate(
            {"text": prefix_and_output_text[idx]},
            completion_length=0,
            hf_model_name=oracle_model_name,
            tokenizer=oracle_tokenizer,
            truncate_left=True,  # we add this to cover if the generation is longer than the oracle's max length
            model_max_length=oracle_model.config.max_position_embeddings,
        )["input_ids"]

        # if only want to score the "generation" part we need the suffix tokenization length
        tokd_suffix = tokenize_and_truncate(
            {"text": output_text[idx]},
            completion_length=0,
            hf_model_name=oracle_model_name,
            tokenizer=oracle_tokenizer,
        )["input_ids"]

        tokd_labels = tokd_prefix.clone().detach()
        tokd_labels[:, : tokd_labels.shape[1] - tokd_suffix.shape[1] + 1] = -100

        inputs.append(tokd_prefix)
        labels.append(tokd_labels)

    inputs_batch = collate_batch(input_ids=inputs, collator=data_collator)
    labels_batch = collate_batch(input_ids=labels, collator=data_collator)

    input_ids = inputs_batch["input_ids"].to(oracle_model.device)
    attention_mask = inputs_batch.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(oracle_model.device)
    labels = labels_batch["input_ids"].to(oracle_model.device)

    # mask out pad tokens for loss
    labels[labels == oracle_tokenizer.pad_token_id] = -100

    with torch.no_grad():
        pooled_outputs = oracle_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs = get_unpool_fn(oracle_model_name)(pooled_outputs.logits, labels, oracle_model)
        loss = (
            outputs.loss
        )  # avg CE loss all sequence positions (except where labels -100, i.e. pad)
        # ppl = torch.tensor(math.exp(loss))
        ppl = torch.exp(loss)

    # Free tensors before returning (caller may optionally clear CUDA cache)
    try:
        del inputs, labels
        del inputs_batch, labels_batch
        del input_ids, attention_mask
        del pooled_outputs, outputs
    except Exception:
        pass

    return loss.tolist(), ppl.tolist()


def evaluate_ppl(
    examples: dict,
    oracle_model_name=None,
    oracle_model=None,
    oracle_tokenizer=None,
    data_collator=None,
    empty_cache_between_batches: bool = True,
    include_prompt_in_ppl: bool = True,
):
    # Collect data for aggregated statistics
    all_stats = {}

    inputs_plus_baseline_outputs = []
    baseline_outputs = []
    inputs_plus_no_wm_outputs = []
    no_wm_outputs = []
    inputs_plus_w_wm_outputs = []
    w_wm_outputs = []
    inputs_plus_w_wm_output_attackeds = []
    w_wm_output_attackeds = []

    for idx in range(len(examples["truncated_input"])):
        # pull out the required fields from the pipeline results
        prefix = examples.get('truncated_input', [''] * len(examples['baseline_completion']))[idx]
        inputs_plus_baseline_output = (
            f"{prefix}{examples['baseline_completion'][idx]}" if include_prompt_in_ppl else f"{examples['baseline_completion'][idx]}"
        )
        baseline_output = f"{examples['baseline_completion'][idx]}"

        inputs_plus_no_wm_output = (
            f"{prefix}{examples['no_wm_output'][idx]}" if include_prompt_in_ppl else f"{examples['no_wm_output'][idx]}"
        )
        no_wm_output = f"{examples['no_wm_output'][idx]}"

        inputs_plus_w_wm_output = (
            f"{prefix}{examples['w_wm_output'][idx]}" if include_prompt_in_ppl else f"{examples['w_wm_output'][idx]}"
        )
        w_wm_output = f"{examples['w_wm_output'][idx]}"

        if "w_wm_output_attacked" in examples:
            inputs_plus_w_wm_output_attacked = (
                f"{prefix}{examples['w_wm_output_attacked'][idx]}" if include_prompt_in_ppl else f"{examples['w_wm_output_attacked'][idx]}"
            )
            w_wm_output_attacked = f"{examples['w_wm_output_attacked'][idx]}"

        # add to lists
        inputs_plus_baseline_outputs.append(inputs_plus_baseline_output)
        baseline_outputs.append(baseline_output)
        inputs_plus_no_wm_outputs.append(inputs_plus_no_wm_output)
        no_wm_outputs.append(no_wm_output)
        inputs_plus_w_wm_outputs.append(inputs_plus_w_wm_output)
        w_wm_outputs.append(w_wm_output)
        if "w_wm_output_attacked" in examples:
            inputs_plus_w_wm_output_attackeds.append(inputs_plus_w_wm_output_attacked)
            w_wm_output_attackeds.append(w_wm_output_attacked)

    # add metrics
    loss, ppl = compute_ppl_batch(
        inputs_plus_baseline_outputs,
        baseline_outputs,
        oracle_model_name,
        oracle_model,
        oracle_tokenizer,
        data_collator=data_collator,
    )
    examples["baseline_completion_loss"] = loss
    examples["baseline_completion_ppl"] = ppl

    # Collect data for aggregated statistics
    lengths = []
    for idx in range(len(baseline_outputs)):
        tokd = oracle_tokenizer(baseline_outputs[idx], return_tensors="pt", add_special_tokens=False)["input_ids"]
        lengths.append(tokd.shape[1])
    all_stats['baseline_completion'] = {
        'lengths': lengths,
        'ppls': ppl,
        'outputs': baseline_outputs
    }
    # Optional cache clear between sub-batches
    try:
        if empty_cache_between_batches and torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    except Exception:
        pass

    loss, ppl = compute_ppl_batch(
        inputs_plus_no_wm_outputs,
        no_wm_outputs,
        oracle_model_name,
        oracle_model,
        oracle_tokenizer,
        data_collator=data_collator,
    )
    examples["no_wm_output_loss"] = loss
    examples["no_wm_output_ppl"] = ppl

    # Collect data for aggregated statistics
    lengths = []
    for idx in range(len(no_wm_outputs)):
        tokd = oracle_tokenizer(no_wm_outputs[idx], return_tensors="pt", add_special_tokens=False)["input_ids"]
        lengths.append(tokd.shape[1])
    all_stats['no_wm_output'] = {
        'lengths': lengths,
        'ppls': ppl,
        'outputs': no_wm_outputs
    }
    try:
        if empty_cache_between_batches and torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    except Exception:
        pass

    loss, ppl = compute_ppl_batch(
        inputs_plus_w_wm_outputs,
        w_wm_outputs,
        oracle_model_name,
        oracle_model,
        oracle_tokenizer,
        data_collator=data_collator,
    )
    examples["w_wm_output_loss"] = loss
    examples["w_wm_output_ppl"] = ppl

    # Collect data for aggregated statistics
    lengths = []
    for idx in range(len(w_wm_outputs)):
        tokd = oracle_tokenizer(w_wm_outputs[idx], return_tensors="pt", add_special_tokens=False)["input_ids"]
        lengths.append(tokd.shape[1])
    all_stats['w_wm_output'] = {
        'lengths': lengths,
        'ppls': ppl,
        'outputs': w_wm_outputs
    }
    try:
        if empty_cache_between_batches and torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    except Exception:
        pass

    if "w_wm_output_attacked" in examples:
        loss, ppl = compute_ppl_batch(
            inputs_plus_w_wm_output_attackeds,
            w_wm_output_attackeds,
            oracle_model_name,
            oracle_model,
            oracle_tokenizer,
            data_collator=data_collator,
        )
        examples["w_wm_output_attacked_loss"] = loss
        examples["w_wm_output_attacked_ppl"] = ppl

        # Collect data for aggregated statistics
        lengths = []
        for idx in range(len(w_wm_output_attackeds)):
            tokd = oracle_tokenizer(w_wm_output_attackeds[idx], return_tensors="pt", add_special_tokens=False)["input_ids"]
            lengths.append(tokd.shape[1])
        all_stats['w_wm_output_attacked'] = {
            'lengths': lengths,
            'ppls': ppl,
            'outputs': w_wm_output_attackeds
        }

        try:
            if empty_cache_between_batches and torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        except Exception:
            pass

    # Print aggregated PPL statistics
    # print(f"\n{'#'*80}")
    # print(f"PPL ANALYSIS SUMMARY (all batches aggregated)")
    # print(f"{'#'*80}")

    # # Compare at different filtering levels
    # for min_len in [0, 150, 200, 250]:
    #     print(f"\nToken Length Filter: >= {min_len} tokens" + (" (matching lower_tolerance_T)" if min_len == 150 else ""))
    #     print(f"{'-'*80}")

    #     for col_name in ['baseline_completion', 'no_wm_output', 'w_wm_output', 'w_wm_output_attacked']:
    #         if col_name not in all_stats:
    #             continue

    #         lengths = all_stats[col_name]['lengths']
    #         ppls = all_stats[col_name]['ppls']

    #         # Filter by length
    #         filtered_ppl = [ppls[i] for i in range(len(ppls)) if lengths[i] >= min_len and not (np.isnan(ppls[i]) or np.isinf(ppls[i]))]

    #         if filtered_ppl:
    #             print(f"  {col_name:25s}: mean={np.mean(filtered_ppl):6.3f}, median={np.median(filtered_ppl):6.3f}, n={len(filtered_ppl):4d}")

    # print(f"\n{'#'*80}\n")

    return examples


def compute_repetition_diversity(example, include_repetition=False, include_diversity=False):
    for col_name in REPETITION_TEXT_COLUMN_NAMES:
        if col_name in example:
            try:
                results_tuple = measure_repetition_and_diversity(example[col_name])
            except Exception as e:
                print(
                    f"Error for '{col_name}' computing repetition and diversity on text: '{example[col_name]}'\nError:{e}"
                )
                results_tuple = dummy_rep_div_result

            if include_repetition:
                # returns pred_seq_2, pred_seq_3, pred_seq_4, pred_div
                # add each key from the result tuple to the example, prepending the col_name
                metrics_dict = {f"{col_name}_{key}": value for key, value in results_tuple.items()}
                example.update(metrics_dict)
            if include_diversity:
                # returns diversity only
                example[f"{col_name}_diversity"] = results_tuple["diversity"]
                example[f"{col_name}_log_diversity"] = results_tuple["log_diversity"]
    return example


def compute_p_sp(dataset):
    for column_pair in P_SP_TEXT_PAIR_COLUMN_NAMES:
        if column_pair[0] in dataset.features and column_pair[1] in dataset.features:
            p_sp_scores = evaluate_p_sp(dataset[column_pair[0]], dataset[column_pair[1]])
            if f"{column_pair[0]}_vs_{column_pair[1]}_p_sp" in dataset.features:
                print(
                    f"WARNING: Removing existing {column_pair[0]}_vs_{column_pair[1]}_p_sp column because it was already present"
                )
                dataset = dataset.remove_columns([f"{column_pair[0]}_vs_{column_pair[1]}_p_sp"])
            dataset = dataset.add_column(f"{column_pair[0]}_vs_{column_pair[1]}_p_sp", p_sp_scores)
    return dataset


def compute_mauve(dataset):
    """
    The current convention is to repeat the score for all rows in the dataset
    under the assumption that the final score will be retreived via
    a groupby + take(1) operation or similar (even a `mean` would be fine)
    """
    for column_pair in MAUVE_TEXT_PAIR_COLUMN_NAMES:
        if column_pair[0] in dataset.features and column_pair[1] in dataset.features:
            mauve_score = get_mauve_score(dataset[column_pair[0]], dataset[column_pair[1]])
            if f"{column_pair[0]}_vs_{column_pair[1]}_mauve" in dataset.features:
                print(
                    f"WARNING: Removing existing {column_pair[0]}_vs_{column_pair[1]}_mauve column because it was already present"
                )
                dataset = dataset.remove_columns([f"{column_pair[0]}_vs_{column_pair[1]}_mauve"])
            dataset = dataset.add_column(
                f"{column_pair[0]}_vs_{column_pair[1]}_mauve", [mauve_score] * len(dataset)
            )
    return dataset


def compute_coherence(dataset):
    """
    Assumes the first column is the prefix or prompt to the model
    and the current convention is to repeat the score for all rows in the dataset
    under the assumption that the final score will be retreived via
    a groupby + take(1) operation or similar (even a `mean` would be fine)
    """
    prefix_column = dataset[COHERENCE_TEXT_COLUMN_NAMES[0]]
    for generated_text_column in COHERENCE_TEXT_COLUMN_NAMES[1:]:
        if generated_text_column in dataset.features:
            coherence_score = get_coherence_score(prefix_column, dataset[generated_text_column])
            if f"{generated_text_column}_coherence" in dataset.features:
                print(
                    f"WARNING: Removing existing {generated_text_column}_coherence column because it was already present"
                )
                dataset = dataset.remove_columns([f"{generated_text_column}_coherence"])
            dataset = dataset.add_column(
                f"{generated_text_column}_coherence", [coherence_score] * len(dataset)
            )
    return dataset


def compute_detect_retrieval(dataset, args=None):
    # if we don't have the attacked column,
    # then mock it using the w_wm_output, just means the two score cols will be the same
    # and we'll need to delete it after
    was_real_attacked_ds = True
    if "w_wm_output_attacked" not in dataset.features:
        # were faking it
        was_real_attacked_ds = False
        dataset = dataset.add_column("w_wm_output_attacked", dataset[args.retrieval_db_column])
        dataset = dataset.add_column(
            "w_wm_output_attacked_length", dataset[f"{args.retrieval_db_column}_length"]
        )

    human_detect, paraphrase_detect, generation_detect = detect_retrieval(dataset, args=args)

    if f"baseline_completion_retrieval_score" in dataset.features:
        print(
            f"WARNING: Removing existing baseline_completion_retrieval_score column because it was already present"
        )
        dataset = dataset.remove_columns(["baseline_completion_retrieval_score"])
    dataset = dataset.add_column(f"baseline_completion_retrieval_score", human_detect)

    if f"{args.retrieval_db_column}_retrieval_score" in dataset.features:
        print(
            f"WARNING: Removing existing {args.retrieval_db_column}_retrieval_score column because it was already present"
        )
        dataset = dataset.remove_columns([f"{args.retrieval_db_column}_retrieval_score"])
    dataset = dataset.add_column(f"{args.retrieval_db_column}_retrieval_score", generation_detect)

    if was_real_attacked_ds:
        if f"w_wm_output_attacked_retrieval_score" in dataset.features:
            print(
                f"WARNING: Removing existing w_wm_output_attacked_retrieval_score column because it was already present"
            )
            dataset = dataset.remove_columns(["w_wm_output_attacked_retrieval_score"])
        dataset = dataset.add_column(f"w_wm_output_attacked_retrieval_score", paraphrase_detect)
        # else this is a dummy column, so delete it
    else:
        # sanity check that the scores are the same for the dummy column and the original
        assert all(
            [
                s1 == s2 if (not np.isnan(s1) and not np.isnan(s2)) else True
                for s1, s2 in zip(paraphrase_detect, generation_detect)
            ]
        )
        dataset = dataset.remove_columns(["w_wm_output_attacked", "w_wm_output_attacked_length"])
    return dataset


from utils.submitit import str2bool

def scheme_hparam_extractor(x):
    is_ff = "ff-" in x
    is_algorithm_3 = ("algorithm-3" in x) or ("selfhash" in x)
    is_anchored = "anchored" in x

    x_norm = x.replace("ff-", "").replace("_prf", "").replace("anchored_", "")
    tup_x = x_norm.split("-")

    # Freeform ff-<prf>-<ctx>-<selfsalt>
    if is_ff:
        return {
            "prf_type": tup_x[0],
            "anchored": is_anchored,
            "context_width": int(tup_x[1]),
            "self_salt": str2bool(tup_x[2]),
        }

    # Explicit context width patterns: lefthash_<n> or simple_<n>
    if x.startswith("lefthash_") or (x.startswith("simple_") and x != "simple_1"):
        try:
            cw = int(x.split("_", 1)[1])
            assert cw >= 1
        except Exception:
            raise ValueError(f"Invalid scheme name {x} found.")
        return {
            "prf_type": "additive",
            "anchored": False,
            "context_width": cw,
            "self_salt": False,
        }

    # Legacy aliases: simple_1 / lefthash
    if ("simple_1" in x) or ("lefthash" in x):
        # Keep this in sync with alternative_prf_schemes.seeding_scheme_lookup
        return {
            "prf_type": "additive",
            "anchored": False,
            "context_width": 5,
            "self_salt": False,
        }

    if is_algorithm_3:
        return {
            "prf_type": "minhash",
            "anchored": True,
            "context_width": 4,
            "self_salt": True,
        }

    raise ValueError(f"Invalid scheme name {x} found.")
