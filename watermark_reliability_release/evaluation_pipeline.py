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

# NoneType is deprecated
# from types import NoneType
NoneType = type(None)

from typing import Union
import os
import argparse
from functools import partial
from tqdm import tqdm
import json

import wandb
import torch
import numpy as np
import sklearn.metrics as metrics

from datasets import Dataset, Sequence
from transformers import DataCollatorWithPadding

from utils.submitit import str2bool  # better bool flag type for argparse
from utils.io import read_jsonlines, read_json, write_json, write_jsonlines
from utils.notebooks import filter_text_col_length, infer_length_column

from utils.evaluation import (
    SUPPORTED_METRICS,
    NO_CHECK_ARGS,
    ROC_TEST_STAT_SUFFIXES,
    FILTER_BY_COLUMNS,
    conditional_no_check_args,
    load_oracle_model,
    evaluate_ppl,
    load_detector,
    compute_z_scores,
    compute_windowed_z_scores,
    compute_run_len_chsqrd_stats,
    compute_repetition_diversity,
    compute_p_sp,
    compute_coherence,
    compute_mauve,
    compute_detect_retrieval,
    load_tokenizer,
    concat_rows,
    ZSCORE_TEXT_COLUMN_NAMES,
)

print(f"Current huggingface cache dir: {os.environ['HF_HOME']}")

from datasets import disable_caching

disable_caching()



def main(args):
    ###########################################################################
    # Create output dir if it doesn't exist, and warn if it contains metric file
    ###########################################################################
    gen_table_w_metrics_path = f"{args.output_dir}/gen_table_w_metrics.jsonl"
    metrics_meta_path = f"{args.output_dir}/gen_table_w_metrics_meta.json"

    print(f"Output dir for this run: {args.output_dir}")
    # notify if exists
    if os.path.exists(args.output_dir):
        print(f"Output dir for this run already exists!")
        print(f"Contents: {sorted(os.listdir(args.output_dir))}")
        # warn if metrics file exists
        if os.path.exists(gen_table_w_metrics_path):
            if not args.overwrite_output_file:
                print(
                    f"WARNING: Exiting to avoid overwriting output file. "
                    f"Pass the '--overwrite_output_file' flag to ignore this check."
                )
                exit()
            else:
                print(
                    f"WARNING: Found existing generation files with metrics added at this output dir. "
                    f"Overwriting anyway :/"
                )
    else:
        # create the output dir where run artifacts are stored
        os.makedirs(args.output_dir)

    ###########################################################################
    # Parse metrics to log - ppl, zscore, etc
    ###########################################################################

    # check that all metrics are supported
    metric_support = [metric in SUPPORTED_METRICS for metric in args.evaluation_metrics]
    assert all(metric_support), (
        f"Unsupported metric '{args.evaluation_metrics[metric_support.index(False)]}' in"
        f" {args.evaluation_metrics}. Supported metrics are: {SUPPORTED_METRICS}"
    )
    # Hack check that if prefix_lengths exists then the method must be
    # detect-retrieval (for now) because other methods don't support the
    # sparse dataset with Nones all over the place
    if "prefix_lengths" in args.__dict__:
        # assert args.evaluation_metrics == [
        #     "detect-retrieval"
        # ], f"Currently, only the detect-retrieval metric supports the prefix_lengths column. "
        print(
            f"WARNING: Found prefix_lengths column assuming that this is either retireval or detectgpt"
        )

    print(f"Evaluation metrics to compute: {args.evaluation_metrics}")

    ###########################################################################
    # Load generations
    ###########################################################################
    print(f"Input dir for this run: {args.input_dir}")
    print(f"Loading previously generated outputs for evaluation via oracle model and metrics...")

    # check for the "attacked version" of the gen table first
    gen_table_meta_path = f"{args.input_dir}/gen_table_attacked_meta.json"
    gen_table_path = f"{args.input_dir}/gen_table_attacked.jsonl"
    safe_gen_table_path = f"{args.input_dir}/gen_table_attacked_safe.jsonl"
    loaded_attacked = True

    attack_variants_exist = [
        os.path.exists(gen_table_meta_path),
        os.path.exists(gen_table_path),
    ]
    if not all(attack_variants_exist):
        loaded_attacked = False
        gen_table_meta_path = f"{args.input_dir}/gen_table_meta.json"
        gen_table_path = f"{args.input_dir}/gen_table.jsonl"
        safe_gen_table_path = f"{args.input_dir}/gen_table_safe.jsonl"

        assert os.path.exists(
            gen_table_meta_path
        ), f"failed file check for prev generations metadata json file: {gen_table_meta_path}"
        assert os.path.exists(
            gen_table_path
        ), f"failed file check for prev generations jsonl file: {gen_table_path}"

    assert not os.path.exists(safe_gen_table_path), (
        f"failed for safety bc there is a secondary 'safe' marked file",
        f" in this dir indicating a possible issue with the generation step. ",
    )

    cmdline_args = args.__dict__.copy()
    prev_gen_table_meta = read_json(gen_table_meta_path)
    joined_args = prev_gen_table_meta.copy()
    for k, v in cmdline_args.items():
        if v is not None:
            joined_args.update({k: v})
        else:
            print(
                f"cmdline arg {k} is None, leaving it as the value found in the input metadata: {prev_gen_table_meta[k]}"
            )

    # check that the args used to generate the prev generations are the same as
    # the current args, for the intersection of keys
    if not args.overwrite_args:
        # update the no check args based on the current state of args
        current_no_check_args = conditional_no_check_args(
            NO_CHECK_ARGS, args.evaluation_metrics, args
        )

        for key in prev_gen_table_meta.keys():
            if key in current_no_check_args:
                continue
            assert joined_args[key] == prev_gen_table_meta[key], (
                f"failed for safety bc after merging the prev metadata with "
                f"the current cmdline args, values for '{key}' are not the same. "
                f"in metadata: {prev_gen_table_meta[key]}, passed: {cmdline_args[key]}. "
                f"Pass the '--overwrite_args' flag to ignore this check."
            )

    args = argparse.Namespace(**joined_args)
    gen_table = [ex for ex in read_jsonlines(gen_table_path)]
    if args.debug:
        gen_table = gen_table[:50]
    if args.limit_rows == -1:
        gen_table_ds = Dataset.from_list(gen_table)
    else:
        gen_table_ds = Dataset.from_list(gen_table[: args.limit_rows])

    if getattr(args, "target_T", 0) > 0 and "baseline_completion" in gen_table_ds.column_names:
        from utils.evaluation import load_tokenizer
        tokenizer = load_tokenizer(args)
        target_T = args.target_T + args.upper_tolerance_T  # add upper tolerance to be safe

        def _truncate_baseline(batch):
            texts = batch["baseline_completion"]
            new_texts = []
            new_lens = []
            for t in texts:
                if not isinstance(t, str):
                    new_texts.append(t)
                    new_lens.append(0)
                    continue
                ids = tokenizer(t, add_special_tokens=False)["input_ids"]
                ids = ids[:target_T]
                new_texts.append(tokenizer.decode(ids, skip_special_tokens=True))
                new_lens.append(len(ids))
            batch["baseline_completion"] = new_texts
            batch["baseline_completion_length"] = new_lens
            return batch

        gen_table_ds = gen_table_ds.map(_truncate_baseline, batched=True)
    # check if newly added params are in the args namespace
    # when running old generations
    args_dict = vars(args)
    if not args_dict.get("use_position_prf"):
        args.use_position_prf = False
    if not args_dict.get("code_length"):
        args.code_length = args.message_length
    if not args_dict.get("use_fixed_position"):
        args.use_fixed_position = False
    if not args_dict.get("watermark_type"):
        args.watermark_type = "multibit"
    if not args_dict.get("chunk_capacity"):
        args.chunk_capacity = 3
    if not args_dict.get("use_gpu"):
        args.use_gpu = True
    if not args_dict.get("top_p"):
        args.top_p = 1.0
    if not args_dict.get("top_k"):
        args.top_k = 0
    ###########################################################################
    # Extract the seeding scheme fine grained parameters
    ###########################################################################
    from utils.evaluation import scheme_hparam_extractor

    args.__dict__.update(scheme_hparam_extractor(args.seeding_scheme))

    print(f"seeding_scheme: {args.seeding_scheme}")
    print(f"prf_type: {args.prf_type}")
    print(f"anchored: {args.anchored}")
    print(f"context_width: {args.context_width}")
    print(f"self_salt: {args.self_salt}")

    ###########################################################################
    # Early filtering: Filter by length before detection to reduce computation
    ###########################################################################
    if getattr(args, 'early_filtering', True):
        print(f"#" * 80)
        print("Performing early filtering before detection...")

        # If target_T is 0, use max_new_tokens
        if args.target_T == 0:
            effective_target_T = args.max_new_tokens
        else:
            effective_target_T = args.target_T

        # Ensure lower_tolerance_T doesn't exceed target_T
        effective_lower_tol = min(args.lower_tolerance_T, effective_target_T)

        print(f"Filtering range: [{effective_target_T - effective_lower_tol}, {effective_target_T + args.upper_tolerance_T}]")

        # Convert to pandas DataFrame for filtering
        df = gen_table_ds.to_pandas()
        orig_len = len(df)

        # Apply length filtering to all detection columns
        filter_columns = ["w_wm_output","no_wm_output"]

        # If attacked column exists, also filter it
        # if "w_wm_output_attacked" in df.columns:
        #     filter_columns.append("w_wm_output_attacked")

        for col in filter_columns:
            if col not in df.columns:
                continue

            length_col_name = infer_length_column(col, df, args=args)
            if length_col_name not in df.columns:
                print(f"Warning: length column '{length_col_name}' not found, skipping filtering for '{col}'")
                continue

            print(f"Filtering by {length_col_name}...")
            df = filter_text_col_length(
                df,
                text_col_name=length_col_name,
                count_suffix="",
                upper_T=effective_target_T + args.upper_tolerance_T,
                lower_T=effective_target_T - effective_lower_tol,
            )

        filtered_len = len(df)
        print(f"After length filtering: {filtered_len} / {orig_len} samples remain ({round(filtered_len/orig_len*100, 1)}%)")

        # Limit to min_generations if specified
        if hasattr(args, 'min_generations') and args.min_generations > 0:
            target_samples = args.min_generations
            target_samples = 500
            if filtered_len > target_samples:
                print(f"Selecting first {target_samples} samples for detection (as specified by min_generations)...")
                df = df.head(target_samples)
            elif filtered_len < target_samples:
                print(f"Warning: Only {filtered_len} samples after filtering, less than target {target_samples}.")
                print(f"Will use all {filtered_len} samples for detection.")

        # Convert back to Dataset
        gen_table_ds = Dataset.from_pandas(df, preserve_index=False)
        print(f"Final dataset for detection: {len(gen_table_ds)} samples")
        print(f"#" * 80)

    ###########################################################################
    # Concat logic for multiple generations
    ###########################################################################

    if args.concat_rows != 0:
        assert isinstance(args.concat_rows, int), f"Invalid concat_rows arg: {args.concat_rows}. "

        # set to all rows if -1
        if args.concat_rows == -1:
            args.concat_rows = len(gen_table_ds)

        if args.shuffle_before_concat:
            print(f"Shuffling the gen table before concatenating every {args.concat_rows} rows...")
            gen_table_ds = gen_table_ds.shuffle()

        print(f"Concatenating every {args.concat_rows} rows of the gen table...")

        # we concat all cols in OUTPUT_TEXT_COLUMN_NAMES
        # and update the length col to reflect the new length
        # which means we need to tokenize the new text temporarily
        # to get the new length

        tokenizer = load_tokenizer(args)

        concat_partial = partial(concat_rows, tokenizer=tokenizer, args=args)

        # manually write a batch loop bc hf doesnt support returning fewer rows than input
        concatenated_rows = []
        for i in tqdm(range(0, len(gen_table_ds), args.concat_rows)):
            batch = gen_table_ds[i : i + args.concat_rows]
            concatenated_rows.append(concat_partial(batch))
        gen_table_concated_ds = Dataset.from_list(concatenated_rows)

        # overwrite the args.max_new_tokens to reflect the implicit new target length T
        # which is concat_rows * max_new_tokens
        args.max_new_tokens = args.concat_rows * args.max_new_tokens

        # write the dataset out in the same filename as the original
        # but check that the input dir is different from the output dir
        assert (
            args.input_dir != args.output_dir
        ), f"Input dir and output dir must be different to write out the result of concat rows."

        if loaded_attacked:
            concat_meta_path = f"{args.output_dir}/gen_table_attacked_meta.json"
            concat_gen_table_path = f"{args.output_dir}/gen_table_attacked.jsonl"
        else:
            concat_meta_path = f"{args.output_dir}/gen_table_meta.json"
            concat_gen_table_path = f"{args.output_dir}/gen_table.jsonl"

        write_json(args.__dict__, concat_meta_path, indent=4)
        gen_table_concated_lst = [ex for ex in gen_table_concated_ds]
        write_jsonlines(gen_table_concated_lst, concat_gen_table_path)
    else:
        gen_table_concated_ds = gen_table_ds

    ###########################################################################
    # Additional args setup
    ###########################################################################
    # if target_T is not specified, use max_new_tokens (which will be in the reloaded gen metadata)
    # and potentially overwritten by the concat logic above
    if args.target_T == 0:
        args.target_T = args.max_new_tokens

    # storing slurm info to allow auditing logfiles
    # note this is set after the metadata check to ignore overwriting
    args.SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
    args.SLURM_ARRAY_JOB_ID = os.getenv("SLURM_ARRAY_JOB_ID")
    args.SLURM_ARRAY_TASK_ID = os.getenv("SLURM_ARRAY_TASK_ID")

    ###########################################################################
    # Start logging, we wait to do this until after loading the generations
    # so that we can log the args used to generate them unioned with the
    # cmdline args
    ###########################################################################
    if args.wandb:
        # start a new wandb run to track this experiment, will send data to it
        run = wandb.init(
            # set the wandb project where this run will be logged
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.run_name}",
            # track hyperparameters and run metadata
            config=args,
            tags=args.wandb_tags,
        )

    ###########################################################################
    # Perplexity (PPL) evaluation
    # NOTE: basically requires a model on gpu, or is extremely slow
    ###########################################################################
    if "ppl" in args.evaluation_metrics:
        assert args.oracle_model_name_or_path, "PPL metric requires oracle model."

        # Load the oracle model for PPL measurement
        oracle_model, oracle_tokenizer, _ = load_oracle_model(args)

        # construct the collator
        data_collator = DataCollatorWithPadding(
            tokenizer=oracle_tokenizer, padding=True, pad_to_multiple_of=8
        )

        # construct fluency/ppl partial
        evaluate_ppl_partial = partial(
            evaluate_ppl,
            oracle_model_name=args.oracle_model_name_or_path,
            oracle_model=oracle_model,
            oracle_tokenizer=oracle_tokenizer,
            data_collator=data_collator,
            empty_cache_between_batches=getattr(args, 'empty_cache_between_batches', True),
            include_prompt_in_ppl=getattr(args, 'include_prompt_in_ppl', True),
        )

        print(f"Computing metrics on model generations: {gen_table_concated_ds}")

        gen_table_w_ppl_ds = gen_table_concated_ds.map(
            evaluate_ppl_partial,
            batched=True,
            batch_size=args.ppl_batch_size,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

        # Free oracle model to release GPU/CPU memory before loading other models
        try:
            oracle_model = oracle_model.to(torch.device("cpu"))
        except Exception:
            pass
        try:
            del oracle_model
        except Exception:
            pass
        try:
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass
    else:
        gen_table_w_ppl_ds = gen_table_concated_ds

    ###########################################################################
    # Load watermark detector (type depends on watermark_type)
    ###########################################################################
    if args.watermark_type == "quantile":
        from quantile_watermark_processor import QuantileWatermarkDetector
        from utils.generation import load_model

        print("Loading model for quantile watermark detection...")
        model, tokenizer, device = load_model(args)
        model.eval()

        watermark_detector = QuantileWatermarkDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma,
            seeding_scheme=args.seeding_scheme,
            chunk_capacity=args.chunk_capacity,
            message_length=args.message_length,
            top_p=args.top_p if hasattr(args, 'top_p') else 1.0,
            top_k=args.top_k if hasattr(args, 'top_k') else 0,
            device=device,
            model=model,
            tokenizer=tokenizer,
            mapping_scheme=getattr(args, 'mapping_scheme', 'identity'),
            mapping_key=getattr(args, 'mapping_key', None),
            temperature=getattr(args, 'temperature', 1.0),
            glrt_mode=getattr(args, 'glrt_mode', 'lpo'),
            debug=getattr(args, 'debug', False),
            wrap_output_in_chat_template=getattr(args, 'wrap_output_in_chat_template', False),
            hash_from_topk=getattr(args, 'hash_from_topk', False),
            hash_topk=getattr(args, 'hash_topk', 16),
            hash_sort_ids=getattr(args, 'hash_sort_ids', True),
            skip_ratio=getattr(args, 'skip_ratio', 0.0),
        )
        print("Quantile watermark detector loaded.")
    elif args.watermark_type == "quantile_black":
        # 黑盒 quantile 检测：不加载模型，只需要 tokenizer
        from quantile_black_processor import QuantileBlackWatermarkDetector

        print("Loading tokenizer for quantile-black watermark detection (black-box)...")
        tokenizer = load_tokenizer(args)
        device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
        watermark_detector = QuantileBlackWatermarkDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma,
            seeding_scheme=args.seeding_scheme,
            chunk_capacity=args.chunk_capacity,
            message_length=args.message_length,
            device=device,
            tokenizer=tokenizer,
            z_threshold=args.detection_z_threshold,
        )
        print("Quantile-black watermark detector loaded (no oracle LM).")
    elif args.watermark_type == "stealthink":
        # StealthInk detector (does not require oracle LM, only tokenizer)
        from stealthink_watermark_processor import StealthInkDetector

        print("Loading tokenizer for StealthInk watermark detection...")
        tokenizer = load_tokenizer(args)
        device = "cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"
        R = args.gamma
        base = int(1.0 / R) if R > 0 else 2
        watermark_detector = StealthInkDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=R,
            seeding_scheme=args.seeding_scheme,
            base=base,
            message_length=args.message_length,
            device=torch.device(device),
            tokenizer=tokenizer,
            z_threshold=args.detection_z_threshold,
            normalizers=args.normalizers,
            ignore_repeated_ngrams=args.ignore_repeated_ngrams,
            R=R,
        )
        print("StealthInk watermark detector loaded.")
    elif args.watermark_type == "unbiased":
        # Unbiased (zero-bit) detector: requires access to the watermarked model
        from unbiased_watermark_processor import UnbiasedWatermarkDetector
        from utils.generation import load_model

        print("Loading model for Unbiased watermark detection...")
        model, tokenizer, device = load_model(args)
        model.eval()

        watermark_detector = UnbiasedWatermarkDetector(
            vocab=list(tokenizer.get_vocab().values()),
            seeding_scheme=args.seeding_scheme,
            wm_type=getattr(args, "unbiased_type", "gamma"),
            prefix_length=getattr(args, "unbiased_prefix_length", 0),
            n_grid=getattr(args, "unbiased_n_grid", 64),
            device=device,
            model=model,
            tokenizer=tokenizer,
            normalizers=args.normalizers,
            z_threshold=args.detection_z_threshold,
            ignore_history_detection=getattr(args, "unbiased_ignore_history_detection", False),
        )
        print("Unbiased watermark detector loaded.")
    else:
        # Original multibit detector (MPAC)
        watermark_detector = load_detector(args)

    # Map setup for all dataset operations:
    map_setup = dict(batched=False, load_from_cache_file=False)
    ###########################################################################
    # z-score evaluation
    # NOTE: requires a gpu because if original source of watermark randomness,
    # RNG, is gpu based, then detector should be on gpu as well
    ###########################################################################
    if "z-score" in args.evaluation_metrics:
        # Prefer batched detection for quantile watermark
        if args.watermark_type == "quantile" and args.detection_batch_size > 1:
            def compute_z_scores_quantile_batch(batch):
                # Prepare return dict initialized with passthrough batch
                out = dict(batch)
                # batch size for this slice
                some_key = next(iter(batch))
                bs = len(batch[some_key])

                # Initialize placeholders for all text columns so features are fixed at first batch.
                # IMPORTANT: use non-empty lists (e.g., [0] / [nan]) for list-valued
                # fields so that pyarrow infers a concrete element type (int/float)
                # instead of `null`. Otherwise later batches that contain real
                # integer/float lists would fail to cast to a `null`-typed feature.
                for col_name in ZSCORE_TEXT_COLUMN_NAMES:
                    prefix = f"{col_name}_"
                    # Scalar scores
                    out[prefix + "z_score"] = [float("nan")] * bs
                    # Detection time (seconds) returned by quantile detector (per-example average within batch).
                    out[prefix + "decoding_time"] = [float("nan")] * bs
                    # Internal score families from quantile detector (if present).
                    # These remain NaN when the detector does not populate them.
                    out[prefix + "wm_score_glrt"] = [float("nan")] * bs
                    out[prefix + "raw_log_likelihood"] = [float("nan")] * bs
                    out[prefix + "pred_message"] = [""] * bs
                    out[prefix + "token_count_scored"] = [float("nan")] * bs
                    out[prefix + "bit_acc"] = [float("nan")] * bs
                    out[prefix + "bit_match"] = [False] * bs
                    # Block-level diagnostics placeholders
                    out[prefix + "block_match_rate"] = [float("nan")] * bs
                    out[prefix + "block_match_vec"] = [[0] for _ in range(bs)]
                    out[prefix + "block_margins"] = [[float("nan")] for _ in range(bs)]
                    out[prefix + "block_token_counts"] = [[0] for _ in range(bs)]
                    out[prefix + "pred_digits"] = [""] * bs
                    out[prefix + "gold_digits"] = [""] * bs
                    # Optional multi-T statistics (per-prefix z-score / bit-accuracy).
                    # Only created when zscore_T_list is non-empty; values remain NaN
                    # if the detector does not populate z_score_at_T / bit_acc_at_T.
                    if getattr(args, "zscore_T_list", []) or []:
                        for T in args.zscore_T_list:
                            out[f"{prefix}z_T{T}"] = [float("nan")] * bs
                            out[f"{prefix}bit_acc_T{T}"] = [float("nan")] * bs

                # Helper to run detector over a column and merge outputs
                def run_for_column(col_name):
                    if col_name not in batch:
                        return
                    texts = batch[col_name]
                    # Include prompt if requested and present
                    prompt_lens = None
                    if args.include_prompt_in_detection and ("truncated_input" in batch) and ("prompt_length" in batch):
                        texts = [p + t for p, t in zip(batch["truncated_input"], texts)]
                        prompt_lens = batch["prompt_length"]
                    messages = batch.get("message", None)
                    det_outs = watermark_detector.detect_batch(
                        texts=texts,
                        messages=messages,
                        prompt_lens=prompt_lens,
                        return_prediction=False,
                        return_scores=True,
                        # Enable per-prefix statistics only when needed.
                        return_z_at_T=bool(getattr(args, "zscore_T_list", [])),
                    )
                    # det_outs = watermark_detector.detect_batch_multikey(
                    #     texts=texts,
                    #     messages=messages,
                    #     prompt_lens=prompt_lens,
                    #     return_prediction=True,
                    #     return_scores=True,
                    #     return_z_at_T=True,
                    #     num_fake_keys=4,
                    # )
                    # Merge
                    prefix = f"{col_name}_"
                    for i, d in enumerate(det_outs):
                        # Type-stable casts to avoid pyarrow 'null' inference
                        out[prefix + "z_score"][i] = float(d.get("z_score", float("nan")))
                        # Optional raw score families from quantile detector
                        if "wm_score_glrt" in d:
                            try:
                                out[prefix + "wm_score_glrt"][i] = float(
                                    d.get("wm_score_glrt", float("nan"))
                                )
                            except Exception:
                                out[prefix + "wm_score_glrt"][i] = float("nan")
                        out[prefix + "raw_log_likelihood"][i] = float(d.get("raw_log_likelihood", float("nan")))
                        out[prefix + "pred_message"][i] = str(d.get("pred_message", ""))
                        out[prefix + "decoding_time"][i] = float(d.get("decoding_time", float("nan")))
                        # store as float to keep dtype uniform across batches
                        tcs = d.get("token_count_scored", float("nan"))
                        try:
                            out[prefix + "token_count_scored"][i] = float("nan") if tcs is None else float(tcs)
                        except Exception:
                            out[prefix + "token_count_scored"][i] = float("nan")
                        out[prefix + "bit_acc"][i] = float(d.get("bit_acc", float("nan")))
                        bm = d.get("bit_match", False)
                        out[prefix + "bit_match"][i] = False if bm is None else bool(bm)
                        # Block-level diagnostics from detector (if present)
                        if "block_match_rate" in d:
                            try:
                                out[prefix + "block_match_rate"][i] = float(d.get("block_match_rate", float("nan")))
                            except Exception:
                                out[prefix + "block_match_rate"][i] = float("nan")
                        if "block_match_vec" in d:
                            out[prefix + "block_match_vec"][i] = d.get("block_match_vec", [])
                        if "block_margins" in d:
                            out[prefix + "block_margins"][i] = d.get("block_margins", [])
                        if "block_token_counts" in d:
                            out[prefix + "block_token_counts"][i] = d.get("block_token_counts", [])
                        if "pred_digits" in d:
                            out[prefix + "pred_digits"][i] = str(d.get("pred_digits", ""))
                        if "gold_digits" in d:
                            out[prefix + "gold_digits"][i] = str(d.get("gold_digits", ""))
                        # Optional per-prefix statistics: z_score_at_T / bit_acc_at_T.
                        # Detectors may expose these either as dict[T->value] or as a
                        # sequence indexed by (T-1). We map them into fixed columns
                        # w_wm_output_z_T50, w_wm_output_bit_acc_T50, etc.
                        zscore_T_list = getattr(args, "zscore_T_list", []) or []
                        if zscore_T_list:
                            def _get_at_T(seq, T):
                                if seq is None:
                                    return float("nan")
                                try:
                                    import torch
                                except Exception:
                                    torch = None  # type: ignore[assignment]
                                # Mapping form
                                if isinstance(seq, dict):
                                    val = seq.get(T, float("nan"))
                                    try:
                                        return float(val)
                                    except Exception:
                                        return float("nan")
                                # 1D tensor form
                                if torch is not None and isinstance(seq, torch.Tensor):
                                    if seq.ndim == 0 or T <= 0 or T > int(seq.shape[0]):
                                        return float("nan")
                                    try:
                                        return float(seq[T - 1].item())
                                    except Exception:
                                        return float("nan")
                                # List / ndarray form
                                try:
                                    n = len(seq)
                                    if T <= 0 or T > n:
                                        return float("nan")
                                    return float(seq[T - 1])
                                except Exception:
                                    return float("nan")

                            z_seq = d.get("z_score_at_T", None)
                            bit_seq = d.get("bit_acc_at_T", None)
                            for T in zscore_T_list:
                                if z_seq is not None:
                                    out[f"{prefix}z_T{T}"][i] = _get_at_T(z_seq, T)
                                if bit_seq is not None:
                                    out[f"{prefix}bit_acc_T{T}"][i] = _get_at_T(bit_seq, T)

                for col in ZSCORE_TEXT_COLUMN_NAMES:
                    run_for_column(col)
                return out

            desc_suffix = " (quantile, batched)"
            gen_table_w_zscore_ds = gen_table_w_ppl_ds.map(
                compute_z_scores_quantile_batch,
                batched=True,
                batch_size=args.detection_batch_size,
                load_from_cache_file=False,
                desc=f"Computing z-scores{desc_suffix}",
            )
        else:
            # Fallback to scalar detection
            compute_z_scores_partial = partial(
                compute_z_scores,
                watermark_detector=watermark_detector,
                args=args,
            )
            # Add a small suffix to distinguish different watermark detectors in progress bar
            if args.watermark_type == "quantile":
                desc_suffix = " (quantile)"
            elif args.watermark_type == "quantile_black":
                desc_suffix = " (quantile_black)"
            else:
                desc_suffix = ""
            gen_table_w_zscore_ds = gen_table_w_ppl_ds.map(
                compute_z_scores_partial, **map_setup, desc=f"Computing z-scores{desc_suffix}"
            )
    else:
        gen_table_w_zscore_ds = gen_table_w_ppl_ds

    # After z-score computation, if we loaded a quantile detector (which holds a full LM),
    # or an Unbiased detector, free it to reduce memory footprint before running other metrics.
    try:
        if args.watermark_type in ["quantile", "unbiased"]:
            # Best-effort cleanup: move model to CPU and delete references
            try:
                model = model.to(torch.device("cpu"))
            except Exception:
                pass
            try:
                del model
            except Exception:
                pass
            try:
                del watermark_detector
            except Exception:
                pass
            try:
                import gc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass
    except NameError:
        # No detector/model in scope, or different watermark type
        pass

    ###########################################################################
    # Windowed z-score evaluation
    ###########################################################################

    if "windowed-z-score" in args.evaluation_metrics:
        # Quantile, Unbiased and StealthInk watermarks don't support windowed detection
        if args.watermark_type in ["quantile", "stealthink", "unbiased"]:
            print(
                f"Skipping windowed-z-score metric for watermark_type={args.watermark_type} (not supported)"
            )
            gen_table_w_windowed_zscore_ds = gen_table_w_zscore_ds
        else:
            # set up the windowed partial
            compute_windowed_z_scores_partial = partial(
                compute_windowed_z_scores,
                watermark_detector=watermark_detector,
                args=args,
            )

            gen_table_w_windowed_zscore_ds = gen_table_w_zscore_ds.map(
                compute_windowed_z_scores_partial, **map_setup, desc="Computing windowed z-scores"
            )
    else:
        gen_table_w_windowed_zscore_ds = gen_table_w_zscore_ds

    ###########################################################################
    # run-len-chisqrd evaluation
    ###########################################################################
    if "run-len-chisqrd" in args.evaluation_metrics:
        # Quantile, Unbiased and StealthInk watermarks don't support run-len-chisqrd (no green token masks)
        if args.watermark_type in ["quantile", "stealthink", "unbiased"]:
            print(
                f"Skipping run-len-chisqrd metric for watermark_type={args.watermark_type} (not supported)"
            )
            gen_table_w_run_len_chisqrd_ds = gen_table_w_windowed_zscore_ds
        else:
            assert "w_wm_output_green_token_mask" in gen_table_w_windowed_zscore_ds.column_names, (
                f"Currently, run-len-chisqrd metric requires the green token masks to be computed previously "
                f"by one of the z-score metrics."
            )
            # this ^ is unused currently, but we will need it to remove the assert condition above

            # set up the run len chisqrd partial
            compute_run_len_chisqrd_partial = partial(
                compute_run_len_chsqrd_stats,
                watermark_detector=watermark_detector,
                args=args,
            )

            gen_table_w_run_len_chisqrd_ds = gen_table_w_windowed_zscore_ds.map(
                compute_run_len_chisqrd_partial, **map_setup, desc="Computing runlength tests"
            )
    else:
        gen_table_w_run_len_chisqrd_ds = gen_table_w_windowed_zscore_ds

    ###########################################################################
    # Diversity and Repetition evaluation
    ###########################################################################

    if "repetition" in args.evaluation_metrics or "diversity" in args.evaluation_metrics:
        # set up the partial
        compute_repetition_partial = partial(
            compute_repetition_diversity,
            include_repetition=("repetition" in args.evaluation_metrics),
            include_diversity=("diversity" in args.evaluation_metrics),
        )

        gen_table_w_repetition_ds = gen_table_w_run_len_chisqrd_ds.map(
            compute_repetition_partial, **map_setup, desc="Computing text repetition and diversity"
        )
    else:
        gen_table_w_repetition_ds = gen_table_w_run_len_chisqrd_ds

    ###########################################################################
    # P-SP evaluation
    ###########################################################################

    if "p-sp" in args.evaluation_metrics:
        print(f"Loading the P-SP model and computing P-SP")
        gen_table_w_p_sp_ds = compute_p_sp(gen_table_w_repetition_ds)
    else:
        gen_table_w_p_sp_ds = gen_table_w_repetition_ds

    ###########################################################################
    # Coherence evaluation
    ###########################################################################

    if "coherence" in args.evaluation_metrics:
        print(f"Computing coherence")
        gen_table_w_coherence_ds = compute_coherence(gen_table_w_p_sp_ds)
    else:
        gen_table_w_coherence_ds = gen_table_w_p_sp_ds

    ###########################################################################
    # Mauve evaluation
    ###########################################################################

    if "mauve" in args.evaluation_metrics:
        print(f"Computing mauve")
        gen_table_w_mauve_ds = compute_mauve(gen_table_w_coherence_ds)
    else:
        gen_table_w_mauve_ds = gen_table_w_coherence_ds

    ###########################################################################
    # Retrieval detection
    ###########################################################################

    if "detect-retrieval" in args.evaluation_metrics:
        print(f"Computing detect retrieval")
        gen_table_w_detect_retrieval_ds = compute_detect_retrieval(gen_table_w_mauve_ds, args=args)
    else:
        gen_table_w_detect_retrieval_ds = gen_table_w_mauve_ds

    ###########################################################################
    # External API judge (1–5) for text quality
    ###########################################################################

    if "api-judge-5" in args.evaluation_metrics:
        from utils.api_judge import (
            APIJudge5,
            APIJudge5Config,
            DEFAULT_SYSTEM_PROMPT,
            JSONLKeyValueCache,
            _coerce_text,
            make_cache_key,
            make_text_cache_key,
        )

        assert (
            hasattr(args, "api_judge_columns") and len(args.api_judge_columns) == 2
        ), (
            "api-judge-5 currently requires exactly two columns via "
            "--api_judge_columns, e.g. 'no_wm_output,w_wm_output'."
        )

        col_a, col_b = args.api_judge_columns
        system_prompt = (
            args.api_judge_system_prompt
            if getattr(args, "api_judge_system_prompt", "")
            else DEFAULT_SYSTEM_PROMPT
        )
        judge = APIJudge5(
            APIJudge5Config(
                model=args.api_judge_model,
                api_key_env=args.api_judge_api_key_env,
                base_url=args.api_judge_base_url if args.api_judge_base_url else None,
                system_prompt=system_prompt,
                temperature=args.api_judge_temperature,
                max_tokens=args.api_judge_max_tokens,
                timeout_s=args.api_judge_timeout_s,
                max_retries=args.api_judge_max_retries,
                retry_backoff_s=args.api_judge_retry_backoff_s,
                store_reason=args.api_judge_store_reason,
                force_json=args.api_judge_force_json,
            )
        )
        api_judge_cache_path = getattr(args, "api_judge_cache_path", "")
        if not api_judge_cache_path:
            api_judge_cache_path = f"{args.output_dir}/api_judge_cache.jsonl"
        api_judge_cache = JSONLKeyValueCache(api_judge_cache_path)

        def _preview_text(s: str, n: int = 360) -> str:
            s = _coerce_text(s)
            s = s.replace("\n", "\\n")
            return s if len(s) <= n else (s[:n] + "…")

        if args.api_judge_preview:
            row = gen_table_w_detect_retrieval_ds[int(args.api_judge_preview_row)]
            a = row.get(col_a, "")
            b = row.get(col_b, "")
            print("api-judge-5 preview mode (no dataset written):")
            print(f"- row: {int(args.api_judge_preview_row)}")
            print(f"- model: {args.api_judge_model}")
            print(f"- {col_a}: {_preview_text(a)}")
            print(f"- {col_b}: {_preview_text(b)}")
            if args.api_judge_separate_calls:
                res_a = judge.score_text(a)
                res_b = judge.score_text(b)
                res = {"raw": f"text_a: {res_a.get('raw','')}\n\ntext_b: {res_b.get('raw','')}"}
                res[f"{col_a}_score"] = res_a.get("score")
                res[f"{col_b}_score"] = res_b.get("score")
                res[f"{col_a}_eval"] = res_a.get("eval")
                res[f"{col_b}_eval"] = res_b.get("eval")
            else:
                res = judge.score_pair(col_a, a, col_b, b)
            print(f"- {col_a}_score: {res.get(col_a + '_score')}")
            print(f"- {col_b}_score: {res.get(col_b + '_score')}")
            if args.api_judge_store_dimensions:
                a_eval = res.get(f"{col_a}_eval") or {}
                b_eval = res.get(f"{col_b}_eval") or {}
                print(f"- {col_a}_dimension_scores: {a_eval.get('dimension_scores')}")
                print(f"- {col_b}_dimension_scores: {b_eval.get('dimension_scores')}")
            if args.api_judge_store_reason:
                print(f"- {col_a}_reason: {res.get(col_a + '_reason')}")
                print(f"- {col_b}_reason: {res.get(col_b + '_reason')}")
            raw = res.get("raw")
            if raw:
                raw_preview = raw if len(raw) <= 1200 else (raw[:1200] + "…")
                print(f"- raw_preview: {raw_preview}")
            exit(0)

        def compute_api_judge_5(example):
            a = example.get(col_a, "")
            b = example.get(col_b, "")
            missing = None
            short_dims = ["coherence", "clarity", "naturalness", "overall"]

            # Initialize all expected output keys to guarantee a stable schema
            out = {
                f"{col_a}_api_judge_quality_5": missing,
                f"{col_b}_api_judge_quality_5": missing,
            }
            if args.api_judge_store_dimensions:
                for dim in short_dims:
                    out[f"{col_a}_api_judge_{dim}_5"] = missing
                    out[f"{col_b}_api_judge_{dim}_5"] = missing
            if args.api_judge_store_reason:
                out[f"{col_a}_api_judge_quality_reason"] = None
                out[f"{col_b}_api_judge_quality_reason"] = None
            if args.api_judge_store_raw:
                out["api_judge_raw"] = None

            # If a column is missing, propagate None rather than crashing.
            if col_a not in example or col_b not in example:
                return out

            # Local fast path for empty outputs.
            if _coerce_text(a).strip() == "" and _coerce_text(b).strip() == "":
                out[f"{col_a}_api_judge_quality_5"] = 0
                out[f"{col_b}_api_judge_quality_5"] = 0
                if args.api_judge_store_dimensions:
                    for dim in short_dims:
                        out[f"{col_a}_api_judge_{dim}_5"] = 0
                        out[f"{col_b}_api_judge_{dim}_5"] = 0
                if args.api_judge_store_reason:
                    out[f"{col_a}_api_judge_quality_reason"] = "empty"
                    out[f"{col_b}_api_judge_quality_reason"] = "empty"
                if args.api_judge_store_raw:
                    out["api_judge_raw"] = ""
                return out

            try:
                key_a = make_text_cache_key(args.api_judge_model, system_prompt, a)
                key_b = make_text_cache_key(args.api_judge_model, system_prompt, b)
                cached_a = api_judge_cache.get(key_a)
                cached_b = api_judge_cache.get(key_b)

                if isinstance(cached_a, dict):
                    out[f"{col_a}_api_judge_quality_5"] = cached_a.get("overall", missing)
                    if args.api_judge_store_dimensions:
                        for dim in short_dims:
                            if dim in cached_a:
                                out[f"{col_a}_api_judge_{dim}_5"] = cached_a.get(dim, missing)
                if isinstance(cached_b, dict):
                    out[f"{col_b}_api_judge_quality_5"] = cached_b.get("overall", missing)
                    if args.api_judge_store_dimensions:
                        for dim in short_dims:
                            if dim in cached_b:
                                out[f"{col_b}_api_judge_{dim}_5"] = cached_b.get(dim, missing)

                need_a = not isinstance(cached_a, dict)
                need_b = not isinstance(cached_b, dict)

                if need_a or need_b:
                    if args.api_judge_separate_calls:
                        if need_a:
                            res_a = judge.score_text(a)
                            a_eval = res_a.get("eval") or {}
                            if isinstance(a_eval, dict):
                                api_judge_cache.put(key_a, a_eval)
                            out[f"{col_a}_api_judge_quality_5"] = res_a.get("score")
                            if args.api_judge_store_dimensions and isinstance(a_eval, dict):
                                for dim in short_dims:
                                    if dim in a_eval:
                                        out[f"{col_a}_api_judge_{dim}_5"] = a_eval.get(dim)
                        if need_b:
                            res_b = judge.score_text(b)
                            b_eval = res_b.get("eval") or {}
                            if isinstance(b_eval, dict):
                                api_judge_cache.put(key_b, b_eval)
                            out[f"{col_b}_api_judge_quality_5"] = res_b.get("score")
                            if args.api_judge_store_dimensions and isinstance(b_eval, dict):
                                for dim in short_dims:
                                    if dim in b_eval:
                                        out[f"{col_b}_api_judge_{dim}_5"] = b_eval.get(dim)
                    else:
                        res = judge.score_pair(col_a, a, col_b, b)
                        a_eval = res.get(f"{col_a}_eval") or {}
                        b_eval = res.get(f"{col_b}_eval") or {}
                        if isinstance(a_eval, dict):
                            api_judge_cache.put(key_a, a_eval)
                        if isinstance(b_eval, dict):
                            api_judge_cache.put(key_b, b_eval)
                        a_score = res.get(f"{col_a}_score")
                        b_score = res.get(f"{col_b}_score")
                        out[f"{col_a}_api_judge_quality_5"] = a_score
                        out[f"{col_b}_api_judge_quality_5"] = b_score
                        if args.api_judge_store_dimensions:
                            for dim in short_dims:
                                if isinstance(a_eval, dict) and dim in a_eval:
                                    out[f"{col_a}_api_judge_{dim}_5"] = a_eval.get(dim)
                                if isinstance(b_eval, dict) and dim in b_eval:
                                    out[f"{col_b}_api_judge_{dim}_5"] = b_eval.get(dim)
            except Exception as e:
                if not args.api_judge_fail_open:
                    raise
                print(f"WARNING: api-judge-5 failed for a row: {e!r}")
                return out
            return out

        gen_table_w_api_judge_ds = gen_table_w_detect_retrieval_ds.map(
            compute_api_judge_5, **map_setup, desc="Computing api-judge-5 (external)"
        )

        # Write an offline summary that does not rely on wandb. This is useful
        # when running with wandb disabled or when filters are turned off.
        try:
            score_col_a = f"{col_a}_api_judge_quality_5"
            score_col_b = f"{col_b}_api_judge_quality_5"

            def _agg_mean_std(ds, col):
                n = 0
                s = 0.0
                s2 = 0.0
                for row in ds:
                    v = row.get(col, None)
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if not np.isfinite(fv):
                        continue
                    n += 1
                    s += fv
                    s2 += fv * fv
                if n <= 0:
                    return {"mean": float("nan"), "std": float("nan"), "n": 0}
                mean = s / n
                var = (s2 / n) - (mean * mean)
                if var < 0.0:
                    var = 0.0
                return {"mean": float(mean), "std": float(np.sqrt(var)), "n": int(n)}

            api_judge_summary = {
                "api_judge_model": str(args.api_judge_model),
                "api_judge_columns": [str(col_a), str(col_b)],
                "no_wm_overall": _agg_mean_std(gen_table_w_api_judge_ds, score_col_a),
                "wm_overall": _agg_mean_std(gen_table_w_api_judge_ds, score_col_b),
            }
            api_judge_summary_path = f"{args.output_dir}/api_judge_summary.json"
            write_json(api_judge_summary, api_judge_summary_path, indent=2)
            print(f"Wrote api-judge-5 summary to {api_judge_summary_path}")
            print(json.dumps(api_judge_summary, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"WARNING: failed to write api-judge-5 summary: {e!r}")
    else:
        gen_table_w_api_judge_ds = gen_table_w_detect_retrieval_ds

    if "prefix_length" in gen_table_w_detect_retrieval_ds.features:
        if "no_wm_output_retrieval_score" in gen_table_w_detect_retrieval_ds.features:
            print("Avg scores at each prefix length for no_wm_output:")
            print(
                gen_table_w_detect_retrieval_ds.to_pandas()
                .groupby("prefix_length")["no_wm_output_retrieval_score"]
                .describe()
            )
        if "w_wm_output_retrieval_score" in gen_table_w_detect_retrieval_ds.features:
            print("Avg scores at each prefix length for w_wm_output:")
            print(
                gen_table_w_detect_retrieval_ds.to_pandas()
                .groupby("prefix_length")["w_wm_output_retrieval_score"]
                .describe()
            )
        if "w_wm_output_attacked_retrieval_score" in gen_table_w_detect_retrieval_ds.features:
            print("Avg scores at each prefix length for no_wm_output_attacked:")
            print(
                gen_table_w_detect_retrieval_ds.to_pandas()
                .groupby("prefix_length")["w_wm_output_attacked_retrieval_score"]
                .describe()
            )



    ###########################################################################
    # Detectgpt detection
    ###########################################################################
    if "detectgpt" in args.evaluation_metrics:
        assert args.evaluation_metrics == ["detectgpt"], (
            f"Detectgpt must be run separately from other metrics. "
            f"Found: {args.evaluation_metrics}. "
        )
        # check that the right score column exists
        assert any(
            ["detectgpt_score" in col for col in gen_table_w_detect_retrieval_ds.column_names]
        ), (
            f"Detectgpt metric requires the detectgpt_score column to be computed previously "
            f"but no such cols exist in this file."
        )
        print(
            f"Evaluating detectgpt by simply computing ROC-AUC metrics on the scores that already exist"
        )
        gen_table_w_metrics_ds = gen_table_w_detect_retrieval_ds

        # if we loaded an attack file, since detect gpt only outputs a baseline score col
        # and a no_wm_output score col (which is implcitly the attack col if the file was attacked)
        # we need to add the attacked score col to the dataset, and remove the no_wm score col
        if loaded_attacked:
            for suff in ["100_d", "100_z"]:
                gen_table_w_metrics_ds = gen_table_w_metrics_ds.add_column(
                    f"w_wm_output_attacked_detectgpt_score_{suff}",
                    gen_table_w_metrics_ds[f"no_wm_output_detectgpt_score_{suff}"],
                )
                gen_table_w_metrics_ds = gen_table_w_metrics_ds.remove_columns(
                    [f"no_wm_output_detectgpt_score_{suff}"]
                )
    else:
        ###########################################################################
        # Write the final dataset out to disk in jsonl format
        # with the metrics added
        ###########################################################################

        # last applied metric, NOTE which will of course change as more are added
        gen_table_w_metrics_ds = gen_table_w_api_judge_ds

        # write the metadata file, which is a union of the previous metadata
        # and the current cmdline args
        write_json(args.__dict__, metrics_meta_path, indent=4)

        # Stream write to avoid materializing the full dataset in RAM
        write_jsonlines(gen_table_w_metrics_ds, gen_table_w_metrics_path)

    ###########################################################################
    # Ensure attacked-length columns exist (e.g., Dipper attack pipeline)
    ###########################################################################
    # Some attack pipelines (e.g., Dipper paraphraser) only add the
    # `w_wm_output_attacked` text column but do not populate a corresponding
    # length column. Downstream utilities expect a `<col>_length` column for
    # any entry in FILTER_BY_COLUMNS. To remain compatible, we recompute the
    # attacked length here using the evaluation-time tokenizer.
    if (
        "w_wm_output_attacked" in gen_table_w_metrics_ds.column_names
        and "w_wm_output_attacked_length" not in gen_table_w_metrics_ds.column_names
    ):
        print(
            "w_wm_output_attacked_length not found in dataset; "
            "computing it in evaluation_pipeline using the tokenizer."
        )
        tokenizer = load_tokenizer(args)

        def _add_attacked_length(batch):
            texts = batch["w_wm_output_attacked"]
            lengths = []
            for t in texts:
                # Treat empty/whitespace-only strings as length 0 to match the
                # semantics used when an attack fails and yields an empty output.
                if t is None or (isinstance(t, str) and t.strip() == ""):
                    lengths.append(0)
                    continue
                if isinstance(t, list):
                    # Defensive: join list-of-string cases, though Dipper attack
                    # currently stores a single string per row.
                    if len(t) == 0:
                        lengths.append(0)
                        continue
                    t_str = " ".join(map(str, t))
                else:
                    t_str = str(t)

                encoded = tokenizer(t_str, add_special_tokens=False)["input_ids"]
                # HF returns a list of ints for a single string input; be robust
                # to list-of-list in case of tokenizer quirks.
                if isinstance(encoded, (list, tuple)) and encoded and isinstance(
                    encoded[0], (list, tuple)
                ):
                    length = len(encoded[0])
                else:
                    length = len(encoded)
                lengths.append(int(length))

            batch["w_wm_output_attacked_length"] = lengths
            return batch

        gen_table_w_metrics_ds = gen_table_w_metrics_ds.map(
            _add_attacked_length, batched=True, load_from_cache_file=False
        )

    ###########################################################################
    # Log the metric series to wandb
    ###########################################################################
    # log the metrics to wandb
    if args.wandb:
        # find cols that should be logged in a table
        tabular_column_types = ["string", "bool"]
        tabular_column_names = [
            name
            for name, feature in filter(
                lambda tup: hasattr(tup[1], 'dtype') and tup[1].dtype in tabular_column_types,
                gen_table_w_metrics_ds.features.items(),
            )
        ]
        # the rest should be logged as series
        series_column_names = [
            name
            for name, feature in filter(
                lambda tup: not hasattr(tup[1], 'dtype') or tup[1].dtype not in tabular_column_types,
                gen_table_w_metrics_ds.features.items(),
            )
        ]

        # Ensure boolean bit_match columns are treated as series for summary stats
        bit_match_cols = [
            name for name, feature in gen_table_w_metrics_ds.features.items() if 'bit_match' in name
        ]
        for col in bit_match_cols:
            if col not in series_column_names:
                series_column_names.append(col)
            if col in tabular_column_names:
                tabular_column_names.remove(col)

        for metric_name in series_column_names:
            # summarize series metrics as mean by default
            wandb.define_metric(metric_name, summary="mean")

        if args.log_raw_series:
            # log the raw series
            for example in tqdm(
                gen_table_w_metrics_ds.remove_columns(tabular_column_names),
                desc="Logging series metrics to wandb",
            ):
                run.log(example)

        if args.log_raw_tabular:
            # log the raw tabular data
            # but also include the dataset index as a column
            series_column_names.remove("idx")
            table = wandb.Table(
                dataframe=gen_table_w_metrics_ds.remove_columns(series_column_names).to_pandas()
            )
            run.log({"output_table": table})

        ###########################################################################
        # Filter rows, then log means to wandb
        ###########################################################################
        args.lower_tolerance_T = min(args.lower_tolerance_T, args.target_T)
        assert (
            args.target_T - args.lower_tolerance_T
        ) >= 0, "target_T - lower_tolerance_T must be >= 0"

        target_T = args.target_T
        lower_tolerance = args.lower_tolerance_T
        upper_tolerance = args.upper_tolerance_T
        filtered_table = gen_table_w_metrics_ds.to_pandas()  # explictly convert lists
        # Cast boolean bit_match columns to numeric so that pandas ops (mean/std) work and appear in summaries
        for col in bit_match_cols:
            if col in filtered_table.columns:
                try:
                    filtered_table[col] = filtered_table[col].astype(float)
                except Exception:
                    pass
        for col in args.filter_by_columns:
            length_col_name = infer_length_column(col, filtered_table, args=args)
            filtered_table = filter_text_col_length(
                filtered_table,
                text_col_name=length_col_name,
                count_suffix="",
                upper_T=target_T + upper_tolerance,
                lower_T=target_T - lower_tolerance,
            )

        # Save filtered mean values:
        for metric_name in series_column_names:
            # filtered_name = f"f_{target_T}p{upper_tolerance}m{lower_tolerance}_{metric_name}"
            filtered_name = f"filtered_{metric_name}"
            try:
                run.summary[f"{filtered_name}_mean"] = filtered_table[metric_name].mean()
                run.summary[f"{filtered_name}_std"] = filtered_table[metric_name].std()
            except (TypeError, ValueError) as e:
                # Handle list/array columns with varying lengths
                if metric_name == "w_wm_output_attacked_error_pos":
                    continue
                try:
                    two_dim_mean = filtered_table[metric_name].apply(np.mean).mean()
                    run.summary[f"{filtered_name}_mean"] = two_dim_mean
                except Exception:
                    # Skip columns that can't be averaged
                    continue

        # Special handling for PPL metrics - add filtered statistics
        # Note: filtered_table has already been filtered by length via filter_by_columns above
        ppl_metrics = [
            'baseline_completion_ppl',
            'no_wm_output_ppl',
            'w_wm_output_ppl',
            'w_wm_output_attacked_ppl',
        ]

        for ppl_metric in ppl_metrics:
            if ppl_metric in filtered_table.columns:
                # Remove NaN and Inf values before computing statistics
                valid_ppls = filtered_table[ppl_metric].replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid_ppls) > 0:
                    run.summary[f"filtered_{ppl_metric}_mean"] = valid_ppls.mean()
                    run.summary[f"filtered_{ppl_metric}_std"] = valid_ppls.std()
                    run.summary[f"filtered_{ppl_metric}_median"] = valid_ppls.median()
                    run.summary[f"filtered_{ppl_metric}_count"] = len(valid_ppls)

        ###########################################################################
        # Optional: Print block-level debug for low match-rate rows
        ###########################################################################
        try:
            # If block-level stats exist, print a small debug summary for non-matching rows
            for col in ["w_wm_output", "w_wm_output_attacked"]:
                bit_match_col = f"{col}_bit_match"
                block_rate_col = f"{col}_block_match_rate"
                block_vec_col = f"{col}_block_match_vec"
                block_margins_col = f"{col}_block_margins"
                block_tokens_col = f"{col}_block_token_counts"
                if bit_match_col in filtered_table.columns and block_rate_col in filtered_table.columns:
                    df_sub = filtered_table[[c for c in ["idx", bit_match_col, block_rate_col, block_vec_col, block_margins_col, block_tokens_col] if c in filtered_table.columns]]
                    if len(df_sub) == 0:
                        continue
                    # focus on mismatches
                    mism = df_sub[df_sub[bit_match_col] == False]
                    if len(mism) == 0:
                        continue
                    print(f"[Block Debug] Column {col}: {len(mism)} non-matching rows; avg block_match_rate={mism[block_rate_col].mean():.4f}")
                    # Print top-5 lowest block_match_rate rows
                    worst = mism.sort_values(block_rate_col).head(5)
                    for _, row in worst.iterrows():
                        ridx = int(row.get("idx", -1))
                        rate = float(row[block_rate_col]) if not np.isnan(row[block_rate_col]) else float("nan")
                        vec = row.get(block_vec_col, None)
                        margins = row.get(block_margins_col, None)
                        btoks = row.get(block_tokens_col, None)
                        print(f"  idx={ridx} rate={rate:.3f} vec={vec} margins={margins} tokens={btoks}")
        except Exception as e:
            print(f"[Block Debug] Skipped due to error: {e}")

        ###########################################################################
        # Compute ROC-AUC and send to wandb
        ###########################################################################
        try:
            test_stats = args.roc_test_stat
            if isinstance(test_stats, str):
                test_stats = [test_stats]
            for test_stat in test_stats:
                for attacked in [True, False]:
                    try:
                        roc_auc, fpr, tpr, thresholds, tpr_at_X_fpr = _roc_metrics_for_wandb(
                            filtered_table, test_stat, attacked=attacked
                        )
                        run.summary[
                            f"{'attacked_' if attacked else ''}{test_stat}_roc_auc"
                        ] = roc_auc
                        run.summary[
                            f"{'attacked_' if attacked else ''}{test_stat}_tpr_at_X_fpr"
                        ] = tpr_at_X_fpr

                        # for tp, fp, thr in tqdm(
                        #     zip(tpr, fpr, thresholds), desc="Logging ROC curve"
                        # ):
                        #     run.log(
                        #         {
                        #             f"{'attacked_' if attacked else ''}{test_stat}_fpr": fp,
                        #             f"{'attacked_' if attacked else ''}{test_stat}_tpr": tp,
                        #             f"{'attacked_' if attacked else ''}thr": thr,
                        #         }
                        #     )
                        data = [[x, y] for (x, y) in zip(fpr, tpr)]
                        table = wandb.Table(data=data, columns=["fpr", "tpr"])
                        run.log(
                            {
                                f"{'attacked_' if attacked else ''}{test_stat}": wandb.plot.line(
                                    table,
                                    "fpr",
                                    "tpr",
                                    title=f"ROC ({test_stat}{',attacked' if attacked else ',clean'})",
                                )
                            }
                        )
                        print(f"Successfully logged ROC-AUC metrics for {test_stat}.")

                    except Exception as e:
                        if args.verbose:
                            print(e)
                            print(
                                f"Failed to log ROC-AUC metrics for {'attacked output' if attacked else ''} {test_stat}."
                                f"Metric probably was not computed and or attack col not present."
                            )

                ###################################################################
                # Additional ROC variants:
                #   1) no_wm_output (positive) vs baseline_completion (negative)
                #   2) no_wm_output + w_wm_output (positives) vs baseline_completion
                ###################################################################
                try:
                    baseline_col = f"baseline_completion_{test_stat}"
                    no_wm_col = f"no_wm_output_{test_stat}"
                    w_wm_col = f"w_wm_output_{test_stat}"

                    # 1) no_wm_output as positive, baseline_completion as negative
                    if baseline_col in filtered_table.columns and no_wm_col in filtered_table.columns:
                        base_vals = filtered_table[baseline_col].to_numpy()
                        no_wm_vals = filtered_table[no_wm_col].to_numpy()
                        mask = np.isfinite(base_vals) & np.isfinite(no_wm_vals)
                        base_vals = base_vals[mask]
                        no_wm_vals = no_wm_vals[mask]
                        if (len(base_vals) > 0) and (len(no_wm_vals) > 0):
                            labels = np.concatenate(
                                [
                                    np.zeros_like(base_vals, dtype=int),
                                    np.ones_like(no_wm_vals, dtype=int),
                                ]
                            )
                            scores = np.concatenate([base_vals, no_wm_vals])
                            fpr2, tpr2, thresholds2 = metrics.roc_curve(labels, scores, pos_label=1)
                            roc_auc2 = metrics.auc(fpr2, tpr2)
                            try:
                                tpr_at_X_fpr2 = tpr2[np.where(fpr2 < 1e-2)[0][-1]]
                            except IndexError:
                                tpr_at_X_fpr2 = float("NaN")

                            prefix = f"no_wm_vs_baseline_{test_stat}"
                            run.summary[f"{prefix}_roc_auc"] = roc_auc2
                            run.summary[f"{prefix}_tpr_at_X_fpr"] = tpr_at_X_fpr2

                            data2 = [[x, y] for (x, y) in zip(fpr2, tpr2)]
                            table2 = wandb.Table(data=data2, columns=["fpr", "tpr"])
                            run.log(
                                {
                                    prefix: wandb.plot.line(
                                        table2,
                                        "fpr",
                                        "tpr",
                                        title=f"ROC (no_wm_vs_baseline,{test_stat})",
                                    )
                                }
                            )
                            print(f"Successfully logged ROC-AUC metrics for no_wm_vs_baseline ({test_stat}).")

                    # 2) Treat both no_wm_output and w_wm_output as positives vs baseline_completion
                    if (
                        baseline_col in filtered_table.columns
                        and no_wm_col in filtered_table.columns
                        and w_wm_col in filtered_table.columns
                    ):
                        base_vals = filtered_table[baseline_col].to_numpy()
                        no_wm_vals = filtered_table[no_wm_col].to_numpy()
                        w_wm_vals = filtered_table[w_wm_col].to_numpy()

                        neg_scores = base_vals[np.isfinite(base_vals)]
                        pos_scores = np.concatenate(
                            [
                                no_wm_vals[np.isfinite(no_wm_vals)],
                                w_wm_vals[np.isfinite(w_wm_vals)],
                            ]
                        )

                        if (len(neg_scores) > 0) and (len(pos_scores) > 0):
                            labels_c = np.concatenate(
                                [
                                    np.zeros_like(neg_scores, dtype=int),
                                    np.ones_like(pos_scores, dtype=int),
                                ]
                            )
                            scores_c = np.concatenate([neg_scores, pos_scores])
                            fpr_c, tpr_c, thresholds_c = metrics.roc_curve(labels_c, scores_c, pos_label=1)
                            roc_auc_c = metrics.auc(fpr_c, tpr_c)
                            try:
                                tpr_at_X_fpr_c = tpr_c[np.where(fpr_c < 1e-2)[0][-1]]
                            except IndexError:
                                tpr_at_X_fpr_c = float("NaN")

                            prefix_c = f"no_wm_and_w_wm_vs_baseline_{test_stat}"
                            run.summary[f"{prefix_c}_roc_auc"] = roc_auc_c
                            run.summary[f"{prefix_c}_tpr_at_X_fpr"] = tpr_at_X_fpr_c

                            data_c = [[x, y] for (x, y) in zip(fpr_c, tpr_c)]
                            table_c = wandb.Table(data=data_c, columns=["fpr", "tpr"])
                            run.log(
                                {
                                    prefix_c: wandb.plot.line(
                                        table_c,
                                        "fpr",
                                        "tpr",
                                        title=f"ROC (no_wm_and_w_wm_vs_baseline,{test_stat})",
                                    )
                                }
                            )
                            print(
                                f"Successfully logged ROC-AUC metrics for no_wm_and_w_wm_vs_baseline ({test_stat})."
                            )
                except Exception as e:
                    if args.verbose:
                        print(e)
                        print(
                            f"Failed to log ROC-AUC metrics for {'attacked output' if attacked else ''} {test_stat}."
                                f"Metric probably was not computed and or attack col not present."
                            )
        except Exception as e:
            if args.verbose:
                print(f"Exception: {e}")
                print(
                    f"Failed to log ROC-AUC metrics. ",
                    f"Make sure the test statistic required for detection ({test_stat}) has been computed!",
                )

        ###########################################################################
        # Per-T ROC-AUC and bit-accuracy (multi-T z-score analysis)
        ###########################################################################
        zscore_T_list = getattr(args, "zscore_T_list", []) or []
        if zscore_T_list:
            auc_T_values = []
            auc_scores = []
            tpr_scores = []
            bitacc_T_values = []
            bitacc_scores = []
            for T in zscore_T_list:
                # AUC for z-score at prefix length T:
                #   positives:  w_wm_output_z_T{T}
                #   negatives:  baseline_completion_z_T{T}
                pos_col = f"w_wm_output_z_T{T}"
                neg_col = f"baseline_completion_z_T{T}"
                if pos_col in filtered_table.columns and neg_col in filtered_table.columns:
                    pos_vals = filtered_table[pos_col].to_numpy()
                    neg_vals = filtered_table[neg_col].to_numpy()
                    # Drop NaNs
                    mask = np.isfinite(pos_vals) & np.isfinite(neg_vals)
                    pos_vals = pos_vals[mask]
                    neg_vals = neg_vals[mask]
                    if (len(pos_vals) > 0) and (len(neg_vals) > 0):
                        labels = np.concatenate(
                            [
                                np.zeros_like(neg_vals, dtype=int),
                                np.ones_like(pos_vals, dtype=int),
                            ]
                        )
                        scores = np.concatenate([neg_vals, pos_vals])
                        fpr_T, tpr_T, thr_T = metrics.roc_curve(labels, scores, pos_label=1)
                        auc_T = metrics.auc(fpr_T, tpr_T)
                        try:
                            tpr_at_X_fpr_T = tpr_T[np.where(fpr_T < 1e-2)[0][-1]]
                        except IndexError:
                            tpr_at_X_fpr_T = float("NaN")
                        auc_T_values.append(T)
                        auc_scores.append(float(auc_T))
                        tpr_scores.append(float(tpr_at_X_fpr_T))
                        run.summary[f"auc_z_T{T}"] = float(auc_T)
                        run.summary[f"tpr_at_1e-2_fpr_z_T{T}"] = float(tpr_at_X_fpr_T)

                # Mean bit-accuracy at prefix length T (multi-bit detectors only).
                acc_col = f"w_wm_output_bit_acc_T{T}"
                if acc_col in filtered_table.columns:
                    acc_series = filtered_table[acc_col]
                    acc_series = acc_series.replace([np.inf, -np.inf], np.nan)
                    mask_acc = acc_series.notna()
                    if mask_acc.any():
                        mean_acc_T = float(acc_series[mask_acc].mean())
                        bitacc_T_values.append(T)
                        bitacc_scores.append(mean_acc_T)
                        run.summary[f"bit_acc_T{T}"] = mean_acc_T

            # Log aggregated curves vs T to wandb for easier inspection:
            #   - AUC(z) vs prefix length
            #   - TPR@1e-2 FPR vs prefix length
            #   - Bit accuracy vs prefix length (when available)
            if auc_T_values:
                data_auc = [[t, a] for t, a in zip(auc_T_values, auc_scores)]
                table_auc = wandb.Table(data=data_auc, columns=["T", "auc_z"])
                run.log(
                    {
                        "auc_z_vs_T": wandb.plot.line(
                            table_auc,
                            "T",
                            "auc_z",
                            title="AUC(z) vs prefix length T",
                        )
                    }
                )
                data_tpr = [[t, v] for t, v in zip(auc_T_values, tpr_scores)]
                table_tpr = wandb.Table(data=data_tpr, columns=["T", "tpr_at_1e-2_fpr"])
                run.log(
                    {
                        "tpr_at_1e-2_fpr_vs_T": wandb.plot.line(
                            table_tpr,
                            "T",
                            "tpr_at_1e-2_fpr",
                            title="TPR@1e-2 FPR vs prefix length T",
                        )
                    }
                )
            if bitacc_T_values:
                data_bit = [[t, b] for t, b in zip(bitacc_T_values, bitacc_scores)]
                table_bit = wandb.Table(data=data_bit, columns=["T", "bit_acc"])
                run.log(
                    {
                        "bit_acc_vs_T": wandb.plot.line(
                            table_bit,
                            "T",
                            "bit_acc",
                            title="Bit accuracy vs prefix length T",
                        )
                    }
                )

        ################################################################################
        # NOTE we do that ^^^ basic ROC logic first because it's faster
        # as well as the manual prefix lengths at T logic bc that's also faster
        ################################################################################

        # Handle z @ T but for the retrieval and detectgpt scores that are evaluated
        # manually at each prefix length.  Use groupby to compute the mean and std
        # for each prefix length for any of the feats that have retrieval_score in them,
        # then log those pairs to wandb.
        at_T_df = gen_table_w_metrics_ds.to_pandas()

        for name, feat in gen_table_w_metrics_ds.features.items():
            if "retrieval_score" in name and "prefix_length" in at_T_df.columns:
                # compute the mean and std for each prefix length
                # and log those pairs to wandb
                df_view = at_T_df.groupby("prefix_length")[name].describe()[["mean", "std"]]
                T_indices = df_view.index

                # for idx, (mean, std) in df_view.iterrows():
                #     run.log(data={f"{name}_mean": mean, f"{name}_std": std, "idx_T": idx})
                # log this triple as a table instead like the ROC curve above
                # where the first two are plotted and the third is the x axis
                data = [[x, y, z] for x, (y, z) in df_view.iterrows()]
                table = wandb.Table(data=data, columns=["idx_T", "mean", "std"])
                # compute stderr from std
                table.add_column(
                    "stderr",
                    [
                        std / np.sqrt(len(at_T_df[at_T_df["prefix_length"] == idx]))
                        for idx, std in zip(T_indices, df_view["std"])
                    ],
                )
                # first log mean
                run.log({f"{name}": wandb.plot.line(table, "idx_T", "mean", title=f"{name} mean")})
                # then log std err
                run.log(
                    {
                        f"{name}_stderr": wandb.plot.line(
                            table, "idx_T", "stderr", title=f"{name} stderr"
                        )
                    }
                )

                # also compute an AUC at each prefix len idx by treating the name col as the positives
                # and the baseline_completion_retrieval_score as the negatives
                # then log those pairs to wandb
                if name != "baseline_completion_retrieval_score":
                    pos_negs_at_T = at_T_df.groupby("prefix_length")[
                        [name, "baseline_completion_retrieval_score"]
                    ]
                    # auc_at_T = []
                    # tpr_at_X_fpr = []
                    all_aucs, all_tpr_at_X_fpr = [], []
                    for idx, sub_df in pos_negs_at_T:
                        pos = sub_df[name]
                        neg = sub_df["baseline_completion_retrieval_score"]
                        # convert to arrays and remove nans
                        pos = pos.to_numpy()[~np.isnan(pos.to_numpy())]
                        neg = neg.to_numpy()[~np.isnan(neg.to_numpy())]

                        fpr, tpr, thresholds = metrics.roc_curve(
                            np.concatenate([np.ones_like(pos), np.zeros_like(neg)]),  # labels
                            np.concatenate([pos, neg]),  # scores
                            pos_label=1,
                        )
                        auc = metrics.auc(fpr, tpr)
                        try:
                            tpr_at_X_fpr = tpr[np.where(fpr < 1e-2)[0][-1]]
                        except IndexError:
                            tpr_at_X_fpr = float("NaN")
                        all_aucs.append(auc)
                        all_tpr_at_X_fpr.append(tpr_at_X_fpr)

                        # run.log(data={f"{name}_auc_at_T": auc, "idx_T": idx})
                    # log this triple as a table instead like the AUC and tpr at X fpr below
                    # where the first two are plotted and the third is the x axis
                    data = [
                        [x, y, z] for x, (y, z) in zip(T_indices, zip(all_aucs, all_tpr_at_X_fpr))
                    ]
                    table = wandb.Table(data=data, columns=["idx_T", "aucs", "tpr_at"])
                    run.log(
                        {
                            f"{name}_aucs": wandb.plot.line(
                                table, "idx_T", "aucs", title=f"{name} aucs"
                            )
                        }
                    )
                    run.log(
                        {
                            f"{name}_tpr_at": wandb.plot.line(
                                table, "idx_T", "tpr_at", title=f"{name} tpr_at"
                            )
                        }
                    )

            elif "detectgpt_score" in name and "prefix_length" in at_T_df.columns:
                # this covers detectgpt_score_100_d and variants
                # compute the mean and std for each prefix length
                # and log those pairs to wandb
                df_view = at_T_df.groupby("prefix_length")[name].describe()[["mean", "std"]]
                T_indices = df_view.index

                # for idx, (mean, std) in df_view.iterrows():
                #     run.log(data={f"{name}_mean": mean, f"{name}_std": std, "idx_T": idx})
                # log this triple as a table instead like the ROC curve above
                # where the first two are plotted and the third is the x axis
                data = [[x, y, z] for x, (y, z) in df_view.iterrows()]
                table = wandb.Table(data=data, columns=["idx_T", "mean", "std"])

                # compute stderr from std
                table.add_column(
                    "stderr",
                    [
                        std / np.sqrt(len(at_T_df[at_T_df["prefix_length"] == idx]))
                        for idx, std in zip(T_indices, df_view["std"])
                    ],
                )
                # first log mean
                run.log({f"{name}": wandb.plot.line(table, "idx_T", "mean", title=f"{name} mean")})
                # then log std err
                run.log(
                    {
                        f"{name}_stderr": wandb.plot.line(
                            table, "idx_T", "stderr", title=f"{name} stderr"
                        )
                    }
                )

                # also compute an AUC at each prefix len idx by treating the name col as the positives
                # and the baseline_completion_retrieval_score as the negatives
                # then log those pairs to wandb
                if "baseline_completion_detectgpt_score" not in name:
                    # check which suffix this is in ["_100_d", "_100_z"]
                    # and use that to set the baseline/falst col
                    if name.endswith("_100_d"):
                        baseline_col = "baseline_completion_detectgpt_score_100_d"
                    elif name.endswith("_100_z"):
                        baseline_col = "baseline_completion_detectgpt_score_100_z"
                    pos_negs_at_T = at_T_df.groupby("prefix_length")[[name, baseline_col]]
                    # auc_at_T = []
                    # tpr_at_X_fpr = []
                    all_aucs, all_tpr_at_X_fpr = [], []
                    for idx, sub_df in pos_negs_at_T:
                        pos = sub_df[name]
                        neg = sub_df[baseline_col]
                        # convert to arrays and remove nans
                        pos = pos.to_numpy()[~np.isnan(pos.to_numpy())]
                        neg = neg.to_numpy()[~np.isnan(neg.to_numpy())]

                        fpr, tpr, thresholds = metrics.roc_curve(
                            np.concatenate([np.ones_like(pos), np.zeros_like(neg)]),  # labels
                            np.concatenate([pos, neg]),  # scores
                            pos_label=1,
                        )
                        auc = metrics.auc(fpr, tpr)
                        try:
                            tpr_at_X_fpr = tpr[np.where(fpr < 1e-2)[0][-1]]
                        except IndexError:
                            tpr_at_X_fpr = float("NaN")
                        all_aucs.append(auc)
                        all_tpr_at_X_fpr.append(tpr_at_X_fpr)

                        # run.log(data={f"{name}_auc_at_T": auc, "idx_T": idx})
                    # log this triple as a table instead like the AUC and tpr at X fpr below
                    # where the first two are plotted and the third is the x axis
                    data = [
                        [x, y, z] for x, (y, z) in zip(T_indices, zip(all_aucs, all_tpr_at_X_fpr))
                    ]
                    table = wandb.Table(data=data, columns=["idx_T", "aucs", "tpr_at"])
                    run.log(
                        {
                            f"{name}_aucs": wandb.plot.line(
                                table, "idx_T", "aucs", title=f"{name} aucs"
                            )
                        }
                    )
                    run.log(
                        {
                            f"{name}_tpr_at": wandb.plot.line(
                                table, "idx_T", "tpr_at", title=f"{name} tpr_at"
                            )
                        }
                    )

        ###########################################################################
        # Compute our @ T detection metrics and send to wandb
        ###########################################################################

        # Merge z_at_T and other sequence metrics so they can be shown in wandb:
        for name, feat in gen_table_w_metrics_ds.features.items():
            if isinstance(feat, Sequence):
                max_feat_seq_len = max([len(l) for l in gen_table_w_metrics_ds[name]])
                merging_seq = np.zeros(max_feat_seq_len)
                counts = np.zeros(max_feat_seq_len)
                proto_variance = np.zeros(max_feat_seq_len)

                # Some Sequence features may be non-numeric (e.g., lists of
                # strings). Skip those when computing merged statistics.
                is_numeric_sequence = True
                for entry in gen_table_w_metrics_ds[name]:
                    entry = np.asarray(entry)
                    if not np.issubdtype(entry.dtype, np.number):
                        is_numeric_sequence = False
                        break

                    len_seq = len(entry)
                    delta = entry * counts[:len_seq] - merging_seq[:len_seq]
                    # Accumulate ragged sum over entries:
                    counts[:len_seq] += 1
                    merging_seq[:len_seq] += entry[: len(merging_seq)]
                    # Compute ragged, running variance via Welford:
                    gamma = entry * counts[:len_seq] - merging_seq[:len_seq]
                    proto_variance[:len_seq] += (delta / counts[:len_seq]) * (
                        gamma / counts[:len_seq]
                    )

                if not is_numeric_sequence:
                    print(
                        f"Skipping non-numeric Sequence feature '{name}' when "
                        "computing merged statistics."
                    )
                    continue

                mask = counts != 0
                averaged_seq = merging_seq.copy()
                averaged_seq[mask] /= counts
                averaged_seq[~mask] = float("NaN")

                seq_stderr = proto_variance.copy()
                seq_stderr[counts > 1] = np.sqrt(
                    proto_variance[counts > 1] / (counts[counts > 1] - 1)
                ) / np.sqrt(counts[counts > 1])
                seq_stderr[counts <= 1] = float("NaN")
                # for idx, (avg, stderr) in enumerate(zip(averaged_seq[mask], seq_stderr[mask])):
                #     run.log(data={f"{name}_avg": avg, f"{name}_stderr": stderr, "idx_T": idx})
                # log this triple as a table instead like the ROC curve above
                # where the first two are plotted and the third is the x axis
                data = [
                    [x, y, z]
                    for (x, y, z) in zip(
                        averaged_seq[mask], seq_stderr[mask], range(len(averaged_seq[mask]))
                    )
                ]
                table = wandb.Table(data=data, columns=["avg", "stderr", "idx_T"])

                # first plot avg
                run.log({f"{name}": wandb.plot.line(table, "idx_T", "avg", title=f"{name} avg")})
                # then plot stderr
                run.log(
                    {
                        f"{name}_stderr": wandb.plot.line(
                            table, "idx_T", "stderr", title=f"{name} stderr"
                        )
                    }
                )

        # Compute AUC_at_T
        # For now we'll just do a dumb loop over scipy.roc_curve, but this could be batched
        test_stats = args.roc_test_stat
        if isinstance(test_stats, str):
            test_stats = [test_stats]

        for test_stat in test_stats:
            for attacked in [True, False]:
                base_col = f"baseline_completion_{test_stat}_at_T"
                w_wm_col = f"w_wm_output{'_attacked' if attacked else ''}_{test_stat}_at_T"
                name = f"w_wm{'_attacked' if attacked else ''}_{test_stat}_at_T"

                if w_wm_col in gen_table_w_metrics_ds.features.keys():  # metric was computed
                    print(f"Computing AUC at T for {name}.")
                    max_length = min(
                        max([len(l) for l in gen_table_w_metrics_ds[base_col]]),
                        max([len(l) for l in gen_table_w_metrics_ds[w_wm_col]]),
                    )

                    all_aucs, all_tpr_at_X_fpr = [], []
                    for T in range(1, max_length):
                        w_wm_stats = np.array(
                            [t[T] for t in gen_table_w_metrics_ds[w_wm_col] if len(t) > T]
                        )

                        baseline_stats = np.array(
                            [t[T] for t in gen_table_w_metrics_ds[base_col] if len(t) > T]
                        )[: len(w_wm_stats)]
                        all_scores = np.concatenate([baseline_stats, w_wm_stats])

                        baseline_labels = np.zeros_like(baseline_stats)
                        attacked_labels = np.ones_like(w_wm_stats)
                        all_labels = np.concatenate([baseline_labels, attacked_labels])

                        if len(np.unique(all_labels)) < 2:
                            roc_auc = float("NaN")
                            tpr_at_X_fpr = float("NaN")
                        else:
                            fpr, tpr, thresholds = metrics.roc_curve(
                                all_labels, all_scores, pos_label=1
                            )
                            roc_auc = metrics.auc(fpr, tpr)
                            try:
                                tpr_at_X_fpr = tpr[np.where(fpr < 1e-2)[0][-1]]
                            except IndexError:
                                tpr_at_X_fpr = float("NaN")

                        all_aucs.append(roc_auc)
                        all_tpr_at_X_fpr.append(tpr_at_X_fpr)
                    # for idx, (aucs, tpr_at) in enumerate(zip(all_aucs, all_tpr_at_X_fpr)):
                    #     run.log(data={f"{name}_aucs": aucs, f"{name}_tpr_at": tpr_at, "idx_T": idx})
                    # log these two separately using a table
                    data = [
                        [x, y, z]
                        for (x, y, z) in zip(all_aucs, all_tpr_at_X_fpr, range(len(all_aucs)))
                    ]
                    table = wandb.Table(data=data, columns=["aucs", "tpr_at", "idx_T"])
                    run.log(
                        {
                            f"{name}_aucs": wandb.plot.line(
                                table, "idx_T", "aucs", title=f"{name} aucs"
                            )
                        }
                    )
                    run.log(
                        {
                            f"{name}_tpr_at": wandb.plot.line(
                                table, "idx_T", "tpr_at", title=f"{name} tpr_at"
                            )
                        }
                    )

        # finish the wandb run
        run.finish()

    return


def _roc_metrics_for_wandb(
    gen_table_ds, test_stat="z_score", prefix="", attacked=False, remove_nan=True
):
    # In theory, we actually should be filtering the attacked column too, but we know these
    # end up very short sometimes. So, to make sure the logic works, we just
    # filter for any rows where the test metrics are NaN and note the damage

    baseline_col_name = f"{prefix}baseline_completion_{test_stat}"
    # baseline_col_name = f"{prefix}no_wm_output_{test_stat}"
    if "retrieval" in test_stat:
        if attacked:
            w_wm_col_name = f"{prefix}w_wm_output_attacked_retrieval_score"
        else:
            w_wm_col_name = f"{prefix}{args.retrieval_db_column}_retrieval_score"
    elif "detectgpt" in test_stat:
        if attacked:
            w_wm_col_name = f"{prefix}w_wm_output_attacked_{test_stat}"
        else:
            w_wm_col_name = f"{prefix}no_wm_output_{test_stat}"
    else:
        w_wm_col_name = f"{prefix}w_wm_output{'_attacked' if attacked else ''}_{test_stat}"

    # drop nans in either column
    if remove_nan:
        orig_length = len(gen_table_ds)
        gen_table_ds = gen_table_ds.dropna(subset=[baseline_col_name, w_wm_col_name])
        if orig_length != len(gen_table_ds):
            print(
                f"NOTE: During ROC calculation, dropped {orig_length - len(gen_table_ds)} rows due to NaNs in {baseline_col_name} or {w_wm_col_name}"
            )

    baseline_stats = gen_table_ds[baseline_col_name].values
    w_wm_stats = gen_table_ds[w_wm_col_name].values
    all_scores = np.concatenate([baseline_stats, w_wm_stats])

    baseline_labels = np.zeros_like(baseline_stats)
    attacked_labels = np.ones_like(w_wm_stats)
    all_labels = np.concatenate([baseline_labels, attacked_labels])

    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    try:
        tpr_at_X_fpr = tpr[np.where(fpr < 1e-2)[0][-1]]
    except IndexError:
        tpr_at_X_fpr = float("NaN")
    return roc_auc, fpr, tpr, thresholds, tpr_at_X_fpr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation pipeline for watermark detection")
    parser.add_argument(
        "--watermark_type",
        type=str,
        default="multibit",
        choices=["multibit", "MPAC", "quantile", "quantile_black", "stealthink", "unbiased"],
        help=(
            "The type of watermark to detect: "
            "multibit/MPAC (original), quantile (interval-based), quantile_black (shared-shuffle, black-box-detectable), "
            "stealthink (reweight-based), or unbiased (zero-bit, LLR-scored)."
        ),
    )
    parser.add_argument(
        "--chunk_capacity",
        type=int,
        default=3,
        help="Bits per chunk for quantile watermark. Only used when watermark_type=quantile.",
    )
    parser.add_argument(
        "--mapping_scheme",
        type=str,
        default="identity",
        choices=["identity", "cyclic", "permute"],
        help="Bucket-to-interval mapping scheme for quantile watermark (both generation and detection).",
    )
    parser.add_argument(
        "--mapping_key",
        type=str,
        default="quantile-map-key-v1",
        help="Optional secret key to salt the mapping hash for quantile watermark (both generation and detection).",
    )
    parser.add_argument(
        "--unbiased_type",
        type=str,
        default="gamma",
        choices=["gamma", "delta"],
        help="Reweighting strategy for Unbiased watermark ('gamma' or 'delta').",
    )
    parser.add_argument(
        "--unbiased_prefix_length",
        type=int,
        default=3,
        help="Prefix length (in tokens) used by Unbiased watermark for context-code extraction and scoring.",
    )
    parser.add_argument(
        "--unbiased_n_grid",
        type=int,
        default=8,
        help="Grid size n used by Unbiased watermark detector (number of q-pairs is (n+1) in the q-axis).",
    )
    parser.add_argument(
        "--unbiased_ignore_history_detection",
        type=str2bool,
        default=True,
        help="If True, Unbiased detector ignores context history (always applies reweighting).",
    )
    parser.add_argument(
        "--include_prompt_in_ppl",
        type=str2bool,
        default=True,
        help="Whether to prepend truncated_input to outputs when computing PPL (independent of detection setting).",
    )
    parser.add_argument(
        "--empty_cache_between_batches",
        type=str2bool,
        default=True,
        help="If True, call torch.cuda.empty_cache() and gc.collect() after each batch (generation/PPL) to mitigate OOM.",
    )
    parser.add_argument(
        "--evaluation_metrics",
        type=str,
        default="all",
        help="Comma separated list of columns to remove from the dataset before generation.",
    )
    # External API judge (1–5) for text quality
    parser.add_argument(
        "--api_judge_columns",
        type=str,
        default="no_wm_output,w_wm_output",
        help="Comma-separated list of exactly two text columns to score with api-judge-5.",
    )
    parser.add_argument(
        "--api_judge_model",
        type=str,
        default="gpt-4o",
        help="Chat model name for api-judge-5 (OpenAI-compatible).",
    )
    parser.add_argument(
        "--api_judge_api_key_env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name holding the API key for api-judge-5.",
    )
    parser.add_argument(
        "--api_judge_base_url",
        type=str,
        default="",
        help="Optional base URL for OpenAI-compatible APIs (leave empty for OpenAI default).",
    )
    parser.add_argument(
        "--api_judge_system_prompt",
        type=str,
        default="",
        help="Optional system prompt override for api-judge-5 (inline string).",
    )
    parser.add_argument(
        "--api_judge_system_prompt_path",
        type=str,
        default="",
        help="Optional path to a text file containing the system prompt for api-judge-5.",
    )
    parser.add_argument(
        "--api_judge_temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for api-judge-5.",
    )
    parser.add_argument(
        "--api_judge_max_tokens",
        type=int,
        default=128,
        help="Max tokens to generate for api-judge-5 responses.",
    )
    parser.add_argument(
        "--api_judge_timeout_s",
        type=float,
        default=120.0,
        help="Request timeout (seconds) for api-judge-5.",
    )
    parser.add_argument(
        "--api_judge_max_retries",
        type=int,
        default=20,
        help="Max retries for api-judge-5 on transient failures.",
    )
    parser.add_argument(
        "--api_judge_retry_backoff_s",
        type=float,
        default=3.0,
        help="Base backoff (seconds) for api-judge-5 retries (exponential).",
    )
    parser.add_argument(
        "--api_judge_force_json",
        type=str2bool,
        default=True,
        help="If True, request strict JSON responses when supported by the model.",
    )
    parser.add_argument(
        "--api_judge_cache_path",
        type=str,
        default="",
        help="Optional JSONL cache path for api-judge-5 (appends per-text results). Defaults to <output_dir>/api_judge_cache.jsonl.",
    )
    parser.add_argument(
        "--api_judge_separate_calls",
        type=str2bool,
        default=True,
        help="If True, call the API separately for each column (2 calls per row). If False, score both texts in one call.",
    )
    parser.add_argument(
        "--api_judge_store_reason",
        type=str2bool,
        default=False,
        help="If True, store short reasons from api-judge-5 in the output dataset.",
    )
    parser.add_argument(
        "--api_judge_store_dimensions",
        type=str2bool,
        default=False,
        help="If True, store per-dimension 1–5 scores from api-judge-5 as separate scalar columns.",
    )
    parser.add_argument(
        "--api_judge_store_raw",
        type=str2bool,
        default=False,
        help="If True, store the raw api-judge-5 model output (can be large).",
    )
    parser.add_argument(
        "--api_judge_fail_open",
        type=str2bool,
        default=False,
        help="If True, write None scores on API failures instead of raising.",
    )
    parser.add_argument(
        "--api_judge_preview",
        type=str2bool,
        default=False,
        help="If True, score a single row via api-judge-5, print results, and exit without writing files.",
    )
    parser.add_argument(
        "--api_judge_preview_row",
        type=int,
        default=0,
        help="Row index to use with --api_judge_preview.",
    )
    parser.add_argument(
        "--compute_scores_at_T",
        type=str2bool,
        default=True,
        help="Whether to compute (applicable) metrics at each T index in the output/text columns.",
    )
    parser.add_argument(
        "--overwrite_args",
        type=str2bool,
        default=False,
        help="Whether to overwrite the shared args in the metadata file with the current, runtime args.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Main model for generation/detection (required for quantile watermark detection), path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--oracle_model_name_or_path",
        type=str,
        default="facebook/opt-6.7b",
        help="Oracle model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run model on GPU if available.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=None,
        help=(
            "Whether to run model (for ppl) in float16 precsion, note, will overwrite error as a reminder that "
            "generation was run in other mode, even though there's no hard requirement that these match."
        ),
    )
    parser.add_argument(
        "--ppl_batch_size",
        type=int,
        default=2,
        help="Batch size for ppl eval.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default=None,
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default=None,
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_ngrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--return_green_token_mask",
        type=str2bool,
        default=True,
        help="Whether to return the mask marking which tokens are green from the watermark detector.",
    )
    parser.add_argument(
        "--window_settings",
        type=str,
        default="20,40,max",  # can also be "20" or "20,40,max"
        help="Comma separated list of window sizes to use for watermark detection. Only used if 'windowed-z-score' is in the evaluation metrics list.",
    )
    parser.add_argument(
        "--zscore_T_list",
        type=str,
        default="",
        help=(
            "Comma separated list of prefix lengths T at which to record z-score "
            "and (when available) bit-accuracy statistics. "
            "Example: '50,100,150,200,300,400'. Empty string disables multi-T stats."
        ),
    )
    parser.add_argument(
        "--run_len_chisqrd_variant",
        type=str,
        default="F_succ_T_runs",
        choices=["F_succ_T_runs", "T_and_F_runs"],
        help="The variant of the run length test to use for watermark detection.",
    )
    parser.add_argument(
        "--run_len_chisqrd_bin_spec",
        type=str,
        default="max_plus_1",
        choices=["max", "max_plus_1"],
        help="The binning specification to use for the run length test.",
    )
    parser.add_argument(
        "--run_len_chisqrd_mask_zeros",
        type=str2bool,
        default=True,
        help="Whether to mask zeros in the run length test.",
    )
    parser.add_argument(
        "--run_len_chisqrd_mask_leading_bins",
        type=int,
        default=0,
        help="The number of leading bins to mask in the run length test.",
    )
    parser.add_argument(
        "--run_len_chisqrd_lambda",
        type=str,
        default="pearson",
        choices=["pearson", "g_test", "cressie_read"],
        help="The lambda_ param to use for the run length test.",
    )
    parser.add_argument(
        "--retrieval_technique",
        type=str,
        default="bm25",
        choices=["bm25", "sim"],
        help="The retrieval technique to use for retrieval detection.",
    )
    parser.add_argument(
        "--retrieval_db_column",
        type=str,
        default="no_wm_output",
        choices=["w_wm_output", "no_wm_output"],
        help="The column to populate the db/index with use for retrieval detection.",
    )
    parser.add_argument(
        "--retrieval_db_load_all_prefixes",
        type=str2bool,
        default=False,
        help="Whether to load all prefixes into the retrieval db, or just the longest for each unique entry.",
    )
    parser.add_argument(
        "--roc_test_stat",
        type=str,
        default="all",
        help="The comma separated list of test statistics to use for the ROC-AUC metric.",
    )
    parser.add_argument(
        "--target_T",
        type=int,
        default=0,
        help="The target generation length to use when dropping rows before ROC-AUC evaluation.",
    )
    parser.add_argument(
        "--lower_tolerance_T",
        type=int,
        default=100,
        help="The lower tolerance to use when dropping rows before ROC-AUC evaluation.",
    )
    parser.add_argument(
        "--upper_tolerance_T",
        type=int,
        default=100,
        help="The upper tolerance to use when dropping rows before ROC-AUC evaluation.",
    )
    parser.add_argument(
        "--filter_by_columns",
        type=str,
        default="all",
        help="The comma separated list of columns to filter by before ROC-AUC evaluation.",
    )
    parser.add_argument(
        "--wandb",
        type=str2bool,
        default=False,
        help="Whether to log to wandb.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="lm-watermarking",
        help="The name of the wandb project.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="banga",
        help="The wandb entity/user for the project.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="The comma separated list of tags to add to the wandb run.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="The unique name for the run.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./input",
        help="The directory containing the input files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help=(
            "The directory in which to write out the dataset after adding the metrics. "
            "If not specified, will use the input_dir. Note, if the output_dir already "
            "contains the metric-enriched file, it will be overwritten :/"
        ),
    )
    parser.add_argument(
        "--overwrite_output_file",
        type=str2bool,
        default=False,
        help="Whether to overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--limit_rows",
        type=int,
        default=-1,
        help="The number of rows to limit the dataset to. Useful for debugging.",
    )
    parser.add_argument(
        "--concat_rows",
        type=int,
        default=0,
        help="The number of rows to concatenate into a single row. Result is a mangled dataset, be careful",
    )
    parser.add_argument(
        "--shuffle_before_concat",
        type=str2bool,
        default=False,
        help="Whether to shuffle the dataset before concatenating rows.",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=None,
        help="Whether to verbosely print things here and there.",
    )
    parser.add_argument(
        "--log_raw_series",
        type=str2bool,
        default=True,
        help="Whether to log the raw series metric data to wandb.",
    )
    parser.add_argument(
        "--log_raw_tabular",
        type=str2bool,
        default=True,
        help="Whether to log the raw tabular metric data to wandb.",
    )
    parser.add_argument(
        "--debug",
        type=str2bool,
        default=False
    )
    parser.add_argument(
        "--early_filtering",
        type=str2bool,
        default=True,
        help="Whether to perform early filtering before detection to reduce computation. Filters by length and selects up to min_generations samples.",
    )
    # watermarking related
    parser.add_argument(
        "--glrt_mode",
        type=str,
        default="lpo",
        choices=["lpo", "strict"],
        help="GLRT score mode for quantile detector: 'lpo' uses token-level log-posterior-odds average; 'strict' uses position-level GLRT T (bounded, length/M comparable).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Logit temperature to apply in quantile watermark detection; T>1 smooths, T<1 sharpens.",
    )
    parser.add_argument(
        "--wrap_output_in_chat_template",
        type=str2bool,
        default=False,
        help="When True (and tokenizer has chat_template), wrap bare outputs with an empty user chat prefix for detection, so teacher-forced logits align with chat models even without the original prompt.",
    )
    parser.add_argument(
        "--message_length",
        type=int,
        default=4,
        help="Number of bits of message to watermark",
    )
    parser.add_argument(
        "--base",
        type=int,
        default=4,
        help="Base (radix) of message. Defaults to bit message.",
    )
    parser.add_argument(
        "--include_prompt_in_detection",
        type=str2bool,
        default=True,
        help=(
            "Whether to include the prompt (truncated_input) when detecting watermarks. "
            "When True, passes full text (prompt + generation) and prompt_len to detector. "
            "Important for quantile watermark which needs full context for logit computation. "
            "For multibit watermark, this parameter is ignored as it only analyzes the generation part."
        ),
    )
    parser.add_argument(
        "--detection_batch_size",
        type=int,
        default=4,
        help="Batch size for detection (used when watermark_type=quantile). Set to 1 to disable batched detection.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling parameter.",
    )
    args = parser.parse_args()

    ###########################################################################
    # Argument validation and conditional setting
    ###########################################################################

    # convert evaluation metrics to list
    assert args.evaluation_metrics, "evaluation_metrics list must be specified"
    args.evaluation_metrics = args.evaluation_metrics.split(",")

    if args.evaluation_metrics == ["all"]:
        # Copy to avoid mutating the imported SUPPORTED_METRICS list.
        all_metrics = list(SUPPORTED_METRICS)
        # By default, avoid expensive/external metrics.
        for m in ["ppl", "detectgpt", "api-judge-5"]:
            if m in all_metrics:
                all_metrics.remove(m)
        args.evaluation_metrics = all_metrics
    if args.evaluation_metrics == ["all_w_ppl"]:
        all_metrics = list(SUPPORTED_METRICS)
        all_metrics.remove("api-judge-5")
        args.evaluation_metrics = all_metrics

    # Parse api-judge-5 columns (kept even when metric not selected).
    args.api_judge_columns = [
        c.strip() for c in str(getattr(args, "api_judge_columns", "")).split(",") if c.strip()
    ]
    # Load api-judge-5 system prompt from file if provided.
    if getattr(args, "api_judge_system_prompt_path", ""):
        with open(args.api_judge_system_prompt_path, "r", encoding="utf-8") as f:
            args.api_judge_system_prompt = f.read()

    # if no output dir specified, use the input dir
    if args.output_dir == "":
        args.output_dir = args.input_dir

    # check limit_rows
    assert (args.limit_rows == -1) or (
        (args.limit_rows > 0) and isinstance(args.limit_rows, int)
    ), "limit_rows must be -1 or > 0"

    # convert normalizers to list
    if args.normalizers:
        args.normalizers = args.normalizers.split(",")
    else:
        args.normalizers = []

    # convert roc_test_stat to list
    args.roc_test_stat = args.roc_test_stat.split(",")

    if args.roc_test_stat == ["all"]:
        args.roc_test_stat = ROC_TEST_STAT_SUFFIXES

    # convert filter_by_columns to list
    args.filter_by_columns = args.filter_by_columns.split(",")

    # exclude filtering baseline_completion for those datasets that have longer tokens
    # if "essays" in args.input_dir or "lfqa" in args.input_dir:
    #     FILTER_BY_COLUMNS.remove("baseline_completion")
    #     FILTER_BY_COLUMNS.remove("no_wm_output")

    if args.filter_by_columns == ["all"]:
        args.filter_by_columns = FILTER_BY_COLUMNS

    # split wandb tags
    if args.wandb_tags != "":
        args.wandb_tags = args.wandb_tags.split(",")
    else:
        args.wandb_tags = []

    # split window settings
    args.window_settings = args.window_settings.split(",")

    # Parse z-score prefix list; store as a list[int] for downstream code.
    # Keep the empty-string default as "no multi-T stats".
    if getattr(args, "zscore_T_list", ""):
        try:
            args.zscore_T_list = [
                int(x) for x in str(args.zscore_T_list).split(",") if x.strip() != ""
            ]
        except ValueError:
            raise ValueError(
                f"Failed to parse --zscore_T_list={args.zscore_T_list!r}; "
                f"expected a comma-separated list of integers, e.g. '50,100,150'."
            )
    else:
        args.zscore_T_list = []


    main(args)
