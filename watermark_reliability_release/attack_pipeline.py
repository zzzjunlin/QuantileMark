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
import argparse
from functools import partial

from tqdm import tqdm
import wandb

from datasets import Dataset
from utils.submitit import str2bool  # better bool flag type for argparse
from utils.io import read_jsonlines, read_json, write_json, write_jsonlines

from utils.evaluation import NO_CHECK_ARGS, load_tokenizer
from utils.notebooks import filter_text_col_length, infer_length_column

from utils.attack import (
    SUPPORTED_ATTACK_METHODS,
    gpt_attack,
    dipper_attack,
    tokenize_for_copy_paste,
    copy_paste_attack,
    scramble_attack,
    # New Appendix E attacks
    word_deletion_attack,
    basic_synonym_substitution_attack,
    context_aware_synonym_substitution_attack,
)

print(f"Current huggingface cache dir: {os.environ['HF_HOME']}")


def main(args):
    ###########################################################################
    # Create output dir if it doesn't exist, and warn if it contains an
    # attacked generations file
    ###########################################################################
    gen_table_attacked_path = f"{args.output_dir}/gen_table_attacked.jsonl"
    attacked_meta_path = f"{args.output_dir}/gen_table_attacked_meta.json"

    print(f"Output dir for this run: {args.output_dir}")
    # notify if exists
    if os.path.exists(args.output_dir):
        print(f"Output dir for this run already exists!")
        print(f"Contents: {sorted(os.listdir(args.output_dir))}")
        # warn if metrics file exists
        if os.path.exists(gen_table_attacked_path):
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
    # Parse attack_method arg
    ###########################################################################
    # check that attack method is supported
    assert (
        args.attack_method in SUPPORTED_ATTACK_METHODS
    ), f"Unsupported attack '{args.attack_method}'"
    print(f"Attack method: {args.attack_method}")

    ###########################################################################
    # Load generations
    ###########################################################################
    print(f"Input dir for this run: {args.input_dir}")
    print(f"Loading previously generated outputs for attacking ...")
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
    joined_args.update(cmdline_args)

    # check that the args used to generate the prev generations are the same as
    # the current args, for the intersection of keys
    if not args.overwrite_args:
        for key in prev_gen_table_meta.keys():
            if key in NO_CHECK_ARGS:
                continue
            assert joined_args[key] == prev_gen_table_meta[key], (
                f"failed for safety bc after merging the prev metadata with "
                f"the current cmdline args, values for '{key}' are not the same. "
                f"in metadata: {prev_gen_table_meta[key]}, passed: {cmdline_args[key]}. "
                f"Pass the '--overwrite_args' flag to ignore this check."
            )

    args = argparse.Namespace(**joined_args)
    gen_table = [ex for ex in read_jsonlines(gen_table_path)]
    gen_table_ds = Dataset.from_list(gen_table[: args.limit_rows])

    ###########################################################################
    # Early length-based filtering before expensive attacks (e.g., DIPPER)
    ###########################################################################
    if args.attack_method == "dipper" and getattr(args, "early_filtering", True):
        print("#" * 80)
        print("Performing early length filtering before DIPPER attack...")

        # Ensure tolerance hyperparameters exist with sensible defaults
        if not hasattr(args, "target_T"):
            args.target_T = 0
        if not hasattr(args, "lower_tolerance_T"):
            args.lower_tolerance_T = 0
        if not hasattr(args, "upper_tolerance_T"):
            args.upper_tolerance_T = 0

        # If target_T is 0, fall back to generation length
        if args.target_T == 0:
            effective_target_T = args.max_new_tokens
        else:
            effective_target_T = args.target_T

        # Clamp lower tolerance to [0, target_T]
        effective_lower_tol = min(max(args.lower_tolerance_T, 0), effective_target_T)

        print(
            f"Filtering range: "
            f"[{effective_target_T - effective_lower_tol}, "
            f"{effective_target_T + args.upper_tolerance_T}]"
        )

        df = gen_table_ds.to_pandas()
        orig_len = len(df)

        # Mirror evaluation's behavior: filter by detection columns' lengths
        filter_columns = ["w_wm_output","no_wm_output"]

        for col in filter_columns:
            if col not in df.columns:
                continue

            length_col_name = infer_length_column(col, df, args=args)
            if length_col_name not in df.columns:
                print(
                    f"Warning: length column '{length_col_name}' not found, "
                    f"skipping filtering for '{col}'"
                )
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
        ratio = round(filtered_len / orig_len * 100, 1) if orig_len > 0 else 0.0
        print(
            f"After length filtering: {filtered_len} / {orig_len} samples remain "
            f"({ratio}%)"
        )

        # Optionally cap number of samples attacked (mirror evaluation)
        if hasattr(args, "min_generations") and args.min_generations > 0:
            target_samples = args.min_generations
            # Keep in sync with evaluation_pipeline early filtering behavior
            target_samples = 500
            if filtered_len > target_samples:
                print(
                    f"Selecting first {target_samples} samples for DIPPER attack "
                    f"(as specified by min_generations)..."
                )
                df = df.head(target_samples)
            elif filtered_len < target_samples:
                print(
                    f"Warning: Only {filtered_len} samples after filtering, "
                    f"less than target {target_samples}."
                )
                print(f"Will use all {filtered_len} samples for DIPPER attack.")

        gen_table_ds = Dataset.from_pandas(df, preserve_index=False)
        print(
            f"Final dataset for DIPPER attack after filtering: "
            f"{len(gen_table_ds)} samples"
        )
        print("#" * 80)

    ###########################################################################
    # Start logging, we wait to do this until after loading the generations
    # so that we can log the args used to generate them unioned with the
    # cmdline args
    ###########################################################################
    # storing slurm info to allow auditing logfiles
    # note this is set after the metadata check to ignore overwriting
    args.SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
    args.SLURM_ARRAY_JOB_ID = os.getenv("SLURM_ARRAY_JOB_ID")
    args.SLURM_ARRAY_TASK_ID = os.getenv("SLURM_ARRAY_TASK_ID")

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
    # GPT attack
    ###########################################################################

    if args.attack_method == "gpt":
        print("Running GPT attack")
        import openai

        openai.api_key = os.environ["OPENAI_API_KEY"]
        prompt_pool = read_json("utils/prompts.json")["prompt_pool"]
        prompt_pool = {int(k): v for k, v in prompt_pool.items()}

        if args.attack_prompt is None:
            attack_prompt = prompt_pool[args.attack_prompt_id]
            args.attack_prompt = attack_prompt

        print(f"Using attack prompt: {attack_prompt}")

        gpt_attack_partial = partial(
            gpt_attack,
            attack_prompt=attack_prompt,
            args=args,
        )
        # gen_table_attacked_ds = gen_table_ds.map(
        #     gpt_attack_partial, batched=False, num_proc=min(len(gen_table_ds), 16)
        # )
        gen_table_attacked_ds = gen_table_ds.map(gpt_attack_partial, batched=False)

    ###########################################################################
    # DIPPER attack
    ###########################################################################

    elif args.attack_method == "dipper":
        print("Running DIPPER attack")
        print(f"Using lexical diversity: {args.lex}, order diversity: {args.order}")
        gen_table_attacked_ds = dipper_attack(
            gen_table_ds, lex=args.lex, order=args.order, args=args
        )

    ###########################################################################
    # Scramble attack
    ###########################################################################
    elif args.attack_method == "scramble":
        #  if no cp_attack_min_len specified, use args.max_new_tokens
        if args.cp_attack_min_len == 0:
            args.cp_attack_min_len = args.max_new_tokens
        tokenizer = load_tokenizer(args)
        scramble_attack_partial = partial(
            scramble_attack,
            tokenizer=tokenizer,
            args=args,
        )
        gen_table_attacked_ds = gen_table_ds.map(scramble_attack_partial, batched=False)
    ###########################################################################
    # Word Deletion attack (Appendix E)
    ###########################################################################
    elif args.attack_method == "word-deletion":
        if args.cp_attack_min_len == 0:
            args.cp_attack_min_len = args.max_new_tokens
        tokenizer = load_tokenizer(args)
        # normalize deletion rate (accept percent strings or floats)
        del_rate = args.deletion_rate
        print(f"Running Word Deletion attack: rate={del_rate}, keep_structure={args.keep_structure}")
        deletion_partial = partial(
            word_deletion_attack,
            tokenizer=tokenizer,
            deletion_rate=del_rate,
            keep_structure=args.keep_structure,
            args=args,
        )
        gen_table_attacked_ds = gen_table_ds.map(deletion_partial, batched=False)
    ###########################################################################
    # Synonym substitution (basic)
    ###########################################################################
    elif args.attack_method == "synonym-basic":
        if args.cp_attack_min_len == 0:
            args.cp_attack_min_len = args.max_new_tokens
        tokenizer = load_tokenizer(args)
        print(f"Running Basic Synonym Substitution attack: rate={args.syn_sub_rate}")
        basic_syn_partial = partial(
            basic_synonym_substitution_attack,
            tokenizer=tokenizer,
            sub_rate=args.syn_sub_rate,
            args=args,
        )
        gen_table_attacked_ds = gen_table_ds.map(basic_syn_partial, batched=False)
    ###########################################################################
    # Synonym substitution (context-aware)
    ###########################################################################
    elif args.attack_method == "synonym-context":
        if args.cp_attack_min_len == 0:
            args.cp_attack_min_len = args.max_new_tokens
        tokenizer = load_tokenizer(args)
        print(f"Running Context-aware Synonym Substitution attack: rate={args.syn_sub_rate}")
        ctx_syn_partial = partial(
            context_aware_synonym_substitution_attack,
            tokenizer=tokenizer,
            sub_rate=args.syn_sub_rate,
            args=args,
        )
        gen_table_attacked_ds = gen_table_ds.map(ctx_syn_partial, batched=False)
    ###########################################################################
    # Copy-paste attack
    ###########################################################################
    elif args.attack_method == "copy-paste":
        #  if no cp_attack_min_len specified, use args.max_new_tokens
        if args.cp_attack_min_len == 0:
            args.cp_attack_min_len = args.max_new_tokens

        # NOTE FIXME: the above arg indicates the filter condition by which
        # some rows are skipped/not attacked/NOOP. Since the attacked col
        # is set to the empty string, and length 0, the detection code
        # including the baselines 🤞🏼 will ignore these rows one way or another
        if args.cp_attack_type == "triple-single":
            args.cp_attack_num_insertions = 3
        elif args.cp_attack_type == "single-single":
            args.cp_attack_num_insertions = 1

        # convert cp_attack_insertion_len to int
        if "%" in args.cp_attack_insertion_len:
            original_len_str = args.cp_attack_insertion_len
            # treat as a percent of 1 minus the length of the source col
            # effectively how much of the source col "remains", accounting for
            # the number of insertions that will be made to total this length
            args.cp_attack_insertion_len = (
                int((int(args.cp_attack_insertion_len[:-1]) / 100) * args.max_new_tokens)
                // args.cp_attack_num_insertions
            )

            # check that this is not more than args.max_new_tokens total
            assert (
                args.cp_attack_insertion_len * args.cp_attack_num_insertions <= args.max_new_tokens
            ) and (
                args.cp_attack_insertion_len * args.cp_attack_num_insertions > 0
            ), f"Invalid attack strength: {original_len_str} for {args.cp_attack_num_insertions} insertions."

            args.cp_attack_effective_attack_percentage = (
                1 - (int(original_len_str[:-1]) / 100)
            ) * 100
            print(
                f"Effective attack percentage is 1-{original_len_str}={args.cp_attack_effective_attack_percentage}% by "
                f"copying {args.cp_attack_num_insertions} x {args.cp_attack_insertion_len} = {args.cp_attack_num_insertions * args.cp_attack_insertion_len} tokens "
                f"from {args.cp_attack_src_col} to {args.cp_attack_dst_col} where T={args.max_new_tokens}"
            )
        else:
            args.cp_attack_insertion_len = int(args.cp_attack_insertion_len)
            args.cp_attack_effective_attack_percentage = (
                1
                - (
                    (args.cp_attack_insertion_len * args.cp_attack_num_insertions)
                    / args.max_new_tokens
                )
            ) * 100
            print(
                f"Effective attack percentage is {args.cp_attack_effective_attack_percentage}% by "
                f"copying {args.cp_attack_num_insertions} x {args.cp_attack_insertion_len} = {args.cp_attack_num_insertions * args.cp_attack_insertion_len} tokens "
                f"from {args.cp_attack_src_col} to {args.cp_attack_dst_col} where T={args.max_new_tokens}"
            )

        tokenizer = load_tokenizer(args)
        tokenize_for_copy_paste_partial = partial(tokenize_for_copy_paste, tokenizer=tokenizer)
        gen_table_tokd_ds = gen_table_ds.map(tokenize_for_copy_paste_partial, batched=False)

        copy_paste_attack_partial = partial(copy_paste_attack, tokenizer=tokenizer, args=args)
        gen_table_attacked_ds = gen_table_tokd_ds.map(copy_paste_attack_partial, batched=False)
    ###########################################################################
    # Write the final dataset out to disk in jsonl format
    # with the metrics added
    ###########################################################################
    else:
        raise ValueError(f"Invalid attack method: {args.attack_method}")

    # write the metadata file, which is a union of the previous metadata
    # and the current cmdline args
    write_json(args.__dict__, attacked_meta_path, indent=4)

    gen_table_attacked_lst = [ex for ex in gen_table_attacked_ds]
    write_jsonlines(gen_table_attacked_lst, gen_table_attacked_path)

    ###########################################################################
    # Log the data/series to wandb
    ###########################################################################
    # log the metrics to wandb
    if args.wandb:
        # find cols that should be logged in a table
        tabular_column_types = ["string", "bool"]
        # find cols that should be logged as series (numeric types)
        series_column_types = ["int64", "int32", "float64", "float32", "double"]

        tabular_column_names = [
            name
            for name, feature in gen_table_attacked_ds.features.items()
            if hasattr(feature, 'dtype') and feature.dtype in tabular_column_types
        ]
        # numeric columns should be logged as series
        series_column_names = [
            name
            for name, feature in gen_table_attacked_ds.features.items()
            if hasattr(feature, 'dtype') and feature.dtype in series_column_types
        ]

        for metric_name in series_column_names:
            # summarize series metrics as mean by default
            wandb.define_metric(metric_name, summary="mean")
        # log the raw series
        for example in tqdm(
            gen_table_attacked_ds.remove_columns(tabular_column_names),
            desc="Logging series metrics to wandb",
        ):
            run.log(example)
        # log the raw tabular data
        # but also include the dataset index as a column
        # Only include idx and tabular columns (string/bool) in the table
        table_columns = ["idx"] + tabular_column_names if "idx" in gen_table_attacked_ds.column_names else tabular_column_names
        # Filter to only include columns that actually exist in the dataset
        table_columns = [col for col in table_columns if col in gen_table_attacked_ds.column_names]
        if table_columns:
            table = wandb.Table(
                dataframe=gen_table_attacked_ds.select_columns(table_columns).to_pandas()
            )
            run.log({"output_table": table})

        # finish the wandb run
        run.finish()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attack pipeline for watermark robustness")
    parser.add_argument(
        "--attack_method",
        type=str,
        choices=SUPPORTED_ATTACK_METHODS,
        default="gpt",
        help="The attack method to use.",
    )
    parser.add_argument(
        "--attack_model_name",
        type=str,
        default="gpt-3.5-turbo",
    )
    parser.add_argument(
        "--attack_temperature",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--attack_max_tokens",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--attack_prompt_id",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--attack_prompt",
        type=str,
        default=None,
        help="Pass in the prompt to use for the attack. Is loaded by id from utils/prompts.json by default.",
    )
    parser.add_argument(
        "--no_wm_attack",
        type=str2bool,
        default=False,
        help="Whether to attack the no_wm_output column when running gpt or dipper.",
    )
    parser.add_argument(
        "--overwrite_args",
        type=str2bool,
        default=False,
        help="Whether to overwrite the shared args in the metadata file with the current, runtime args.",
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
        default=None,
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
        default=None,
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
        default=None,
        help="The number of rows to limit the dataset to. Useful for debugging.",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=False,
        help="Whether to print verbose output of every attack.",
    )
    parser.add_argument(
        "--lex",
        type=int,
        default=20,
        help="Lexical diversity knob for the paraphrase attack.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=0,
        help="Order diversity knob for the paraphrase attack.",
    )
    parser.add_argument(
        "--cp_attack_type",
        type=str,
        default="triple-single",
        choices=["single-single", "triple-single", "k-t"],
        help="Type of copy-paste attack to be run.",
    )
    parser.add_argument(
        "--cp_attack_min_len",
        type=int,
        default=0,
        help="Minimum length of cols for the copy-paste attack to be run.",
    )
    parser.add_argument(
        "--cp_attack_num_insertions",
        type=int,
        default=3,
        help="Length of the insertion for the copy-paste attack.",
    )
    parser.add_argument(
        "--cp_attack_insertion_len",
        type=str,
        default="20",
        help=(
            f"Length of the insertion for the copy-paste attack. "
            f"Converts to int. Unless expressed as a percentage, "
            f"in which case it refers to what percent of src is copied to dst, "
            f"which is 1-attack strength as a percentage."
        ),
    )
    parser.add_argument(
        "--cp_attack_src_col",
        type=str,
        default="w_wm_output",
        help="Source column for the copy-paste attack.",
    )
    parser.add_argument(
        "--cp_attack_dst_col",
        type=str,
        default="no_wm_output",
        help="Destination column for the copy-paste attack.",
    )
    parser.add_argument(
        "--target_T",
        type=int,
        default=0,
        help=(
            "Target generation length used when optionally filtering rows before "
            "running certain attacks (e.g., DIPPER). If 0, falls back to max_new_tokens."
        ),
    )
    parser.add_argument(
        "--lower_tolerance_T",
        type=int,
        default=0,
        help=(
            "Lower tolerance for length-based filtering before attacks; valid samples "
            "have length >= target_T - lower_tolerance_T."
        ),
    )
    parser.add_argument(
        "--upper_tolerance_T",
        type=int,
        default=0,
        help=(
            "Upper tolerance for length-based filtering before attacks; valid samples "
            "have length <= target_T + upper_tolerance_T."
        ),
    )
    parser.add_argument(
        "--early_filtering",
        type=str2bool,
        default=True,
        help=(
            "Whether to perform early length-based filtering before expensive attacks "
            "such as DIPPER, to avoid attacking samples that would be dropped during "
            "evaluation."
        ),
    )
    # Word deletion (Appendix E)
    parser.add_argument(
        "--deletion_rate",
        type=str,
        default="10%",
        help=(
            "Fraction of words to delete; accepts float fraction (e.g., 0.1) or percentage string (e.g., '10%')."
        ),
    )
    parser.add_argument(
        "--keep_structure",
        type=str2bool,
        default=True,
        help="If True, delete per-sentence and avoid adjacent/edge words to preserve structure.",
    )
    # Synonym substitution (Appendix E)
    parser.add_argument(
        "--syn_sub_rate",
        type=str,
        default="20%",
        help=(
            "Fraction of words to replace with synonyms; accepts float fraction (e.g., 0.2) or percentage (e.g., '20%')."
        ),
    )
    args = parser.parse_args()

    ###########################################################################
    # Argument validation and conditional setting
    ###########################################################################

    assert args.attack_method, "attack_method must be specified"

    # if no output dir specified, use the input dir
    if args.output_dir is None:
        args.output_dir = args.input_dir

    # check limit_rows
    assert (args.limit_rows is None) or (args.limit_rows == -1) or (
        (args.limit_rows > 0) and isinstance(args.limit_rows, int)
    ), "limit_rows must be > 0, -1, or None"

    # split wandb tags
    if args.wandb_tags != "":
        args.wandb_tags = args.wandb_tags.split(",")
    else:
        args.wandb_tags = []

    main(args)
