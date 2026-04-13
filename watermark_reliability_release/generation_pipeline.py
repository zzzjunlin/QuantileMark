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
import random
from tqdm import tqdm
import torch
import wandb

print(f"Current huggingface cache dir: {os.environ['HF_HOME']}")

# HF classses
from transformers import LogitsProcessorList, DataCollatorWithPadding

# better bool flag type for argparse
from utils.submitit import str2bool

# some file i/o helpers
from utils.io import write_jsonlines, write_json

# watermarking functionality
from mb_watermark_processor import WatermarkLogitsProcessor
# from watermark_processor import WatermarkLogitsProcessor
from stealthink_watermark_processor import ReweightProcessor, ReweightLogitsProcessor

# generation pipeline helpers
from utils.generation import (
    MAX_GENERATIONS,
    load_model,
    load_hf_dataset,
    check_input_lengths,
    check_output_lengths,
    tokenize_for_generation,
    generate,
)


def main(args):
    ###########################################################################
    # Start logging
    ###########################################################################
    # storing slurm info to allow auditing logfiles later
    args.SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
    args.SLURM_ARRAY_JOB_ID = os.getenv("SLURM_ARRAY_JOB_ID")
    args.SLURM_ARRAY_TASK_ID = os.getenv("SLURM_ARRAY_TASK_ID")

    if args.wandb:
        # start a new wandb run to track this experiment, will send data to it later
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
    # Create the output dir
    ###########################################################################
    print(f"Output dir for this run: {args.output_dir}")
    # notify if exists
    if os.path.exists(args.output_dir):
        print(f"Output dir for this run already exists!")
        print(f"Contents: {sorted(os.listdir(args.output_dir))}")
    else:
        # create the output dir where run artifacts are stored
        os.makedirs(args.output_dir)

    ###########################################################################
    # Load the dataset
    ###########################################################################
    # basic ops like shuffling and select are done in load fn
    dataset = load_hf_dataset(args)

    ###########################################################################
    # Instantiate model and tokenizer
    ###########################################################################

    model, tokenizer, device = load_model(args)

    ###########################################################################
    # Configure the prompt construction partial
    ###########################################################################

    # Construct the data filtering/sampling scheme partials
    token_kwargs = dict(
        hf_model_name=args.model_name_or_path,
        tokenizer=tokenizer,
        args=args,
    )
    if args.input_truncation_strategy == "prompt_length":
        token_kwargs.update(dict(min_prompt_tokens=args.min_prompt_tokens))
    elif args.input_truncation_strategy == "completion_length":
        token_kwargs.update(dict(max_new_tokens=args.max_new_tokens))
    elif args.input_truncation_strategy == "no_truncation":
        # truncate_input_for_prompt is a bool flag, that is set by
        # the dataset loading function, semi-redundant, to make sure
        # people are very aware of which input data style they are using
        assert (
            args.truncate_input_for_prompt == False
        ), "Cannot truncate input for prompt if 'no_truncation' strategy is specified"
        pass
    else:
        ValueError(f"Unknown input truncation strategy {args.input_truncation_strategy}")
    tokenize_prompts = partial(tokenize_for_generation, **token_kwargs)

    ###########################################################################
    # Configure the I/O data validation partials
    ###########################################################################

    input_check_kwargs = dict(
        min_sample_len=args.min_sample_tokens,
        max_input_len=model.config.max_position_embeddings,
        max_new_tokens=args.max_new_tokens,
    )
    if args.input_filtering_strategy == "prompt_length":
        input_check_kwargs.update(dict(min_prompt_len=args.min_prompt_tokens, min_completion_len=0))
    elif args.input_filtering_strategy == "completion_length":
        input_check_kwargs.update(dict(min_prompt_len=0, min_completion_len=args.max_new_tokens))
    elif args.input_filtering_strategy == "prompt_and_completion_length":
        input_check_kwargs.update(
            dict(min_prompt_len=args.min_prompt_tokens, min_completion_len=args.max_new_tokens)
        )
    elif args.input_filtering_strategy == "no_filter":
        input_check_kwargs.update(dict(min_prompt_len=0, min_completion_len=0))
    else:
        ValueError(f"Unknown input filtering strategy {args.input_filtering_strategy}")
    input_check = partial(check_input_lengths, **input_check_kwargs)

    if args.output_filtering_strategy == "max_new_tokens":
        # Require outputs to be at least max_new_tokens long
        output_kwargs = dict(min_output_len=args.max_new_tokens)
    elif args.output_filtering_strategy == "length_window":
        # Use a length window consistent with evaluation: keep any sample
        # whose continuation length is at least (target_T - lower_tolerance_T).
        # When target_T is 0, fall back to max_new_tokens.
        effective_target_T = args.target_T if getattr(args, "target_T", 0) > 0 else args.max_new_tokens
        effective_lower_tol = min(getattr(args, "lower_tolerance_T", 0), effective_target_T)
        min_len = max(0, effective_target_T - effective_lower_tol)
        output_kwargs = dict(min_output_len=min_len)
    elif args.output_filtering_strategy == "no_filter":
        output_kwargs = dict(min_output_len=0)
    else:
        ValueError(f"Unknown output filtering strategy {args.output_filtering_strategy}")
    output_check = partial(check_output_lengths, **output_kwargs)

    ###########################################################################
    # Construct the watermark processor
    ###########################################################################
    if args.watermark_type == "quantile":
        from quantile_watermark_processor import QuantileWatermarkLogitsProcessor
        import random

        watermark_processor = QuantileWatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma,
            seeding_scheme=args.seeding_scheme,
            chunk_capacity=args.chunk_capacity,
            message_length=args.message_length,
            top_p=args.top_p if args.use_sampling else 1.0,
            top_k=args.top_k if args.use_sampling else 0,
            device="cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu",
            mapping_scheme=getattr(args, 'mapping_scheme', 'identity'),
            mapping_key=getattr(args, 'mapping_key', None),
            epsilon=getattr(args, 'epsilon', 0.0),
            tokenizer=tokenizer,
        )

        # Set per-run message for quantile watermark (fixed if provided)
        if getattr(args, "fixed_message", None):
            binary_msg = args.fixed_message
        else:
            random.seed(args.generation_seed if args.generation_seed else 0)
            binary_msg = ''.join([str(random.randint(0, 1)) for _ in range(args.message_length)])
        watermark_processor.set_message(binary_msg)
        args.embedded_message = binary_msg  # Store for later use
        print(f"Quantile watermark embedded message: {binary_msg}")
    elif args.watermark_type == "unbiased":
        # Unbiased (zero-bit) watermark: delta/gamma-style reweighting.
        from unbiased_watermark_processor import UnbiasedWatermarkLogitsProcessor

        watermark_processor = UnbiasedWatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            seeding_scheme=args.seeding_scheme,
            wm_type=getattr(args, "unbiased_type", "gamma"),
            prefix_length=getattr(args, "unbiased_prefix_length", 0),
            ignore_history_generation=getattr(
                args, "unbiased_ignore_history_generation", False
            ),
        )

        # For compatibility with utils/generation.generate(), a dummy message
        # is still sampled and stored in the metadata, but it is ignored by
        # the Unbiased processor (zero-bit watermark).
    elif args.watermark_type in ["multibit", "MPAC"]:
        # Original multi-bit watermark (MPAC)
        wm_kwargs = {
                'use_position_prf': args.use_position_prf,
                'use_fixed_position': args.use_fixed_position,
                'code_length': args.message_length,
                'use_feedback': args.use_feedback,
                'feedback_args': {'eta': args.feedback_eta,
                                  'tau': args.feedback_tau,
                                  'feedback_bias': args.feedback_bias
                                  }
                     }
        watermark_processor = WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma,
            delta=args.delta,
            base=args.base,
            seeding_scheme=args.seeding_scheme,
            store_spike_ents=args.store_spike_ents,
            select_green_tokens=True,
            message_length=args.message_length,
            device="cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu",
            **wm_kwargs
        )
    elif args.watermark_type == "stealthink":
        # StealthInk watermark (reweight-based, multi-bit and stealthy)
        # Use gamma as the red-list mass fraction R, and base = floor(1/gamma)
        R = args.gamma
        base = int(1.0 / R) if R > 0 else 2

        reweight_processor = ReweightProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=R,
            delta=args.delta,
            seeding_scheme=args.seeding_scheme,
            select_green_tokens=True,
            base=base,
            message_length=args.message_length,
            code_length=args.code_length,
            use_position_prf=args.use_position_prf,
            use_fixed_position=args.use_fixed_position,
            device="cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu",
        )

        watermark_processor = ReweightLogitsProcessor(
            reweight_processor=reweight_processor,
            R=R,
        )
    else:
        raise ValueError(f"Unknown watermark_type: {args.watermark_type}")

    ###########################################################################
    # Configure the generation partials
    ###########################################################################
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
    logit_processors = []
    # FIXME can add typica
    if args.use_sampling:
        gen_kwargs.update(
            dict(
                do_sample=True,
                top_k=args.top_k,
                top_p=args.top_p,
                typical_p=args.typical_p,
                temperature=args.sampling_temp,
            )
        )
    else:
        from utils.custom_logit_processor import RepetitionPenaltyLogitsProcessor
        if args.repeat_penalty > 1:
            rep_processor = RepetitionPenaltyLogitsProcessor(penalty=args.repeat_penalty)
            logit_processors.append(rep_processor)
        gen_kwargs.update(dict(num_beams=args.num_beams))

    generate_without_watermark = partial(model.generate,
                                         logits_processor=LogitsProcessorList(logit_processors),
                                         **gen_kwargs)
    w_watermark_logit_processors = logit_processors.copy()
    w_watermark_logit_processors.append(watermark_processor)
    generate_with_watermark = partial(
        model.generate, logits_processor=w_watermark_logit_processors, **gen_kwargs
    )

    # construct the collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8)

    generation_partial = partial(
        generate,
        data_collator=data_collator,
        generate_without_watermark=generate_without_watermark,
        generate_with_watermark=generate_with_watermark,
        watermark_processor=watermark_processor,
        tokenizer=tokenizer,
        device=device,
        args=args,
    )

    ###########################################################################
    # Compose the partials to create the pipeline
    ###########################################################################

    print(f"#" * 80)
    print("Setting up data pipeline (lazy evaluation)...")
    print(f"Dataset streaming mode: {args.stream_dataset}")
    print(f"#" * 80)

    # tokenize and truncate the row inputs to create prompts according to the strategy spec'd above
    print("Step 1: Creating tokenization map (lazy)...")
    dataset_w_prompts = dataset.map(tokenize_prompts, batched=False)

    # filter the rows of the dataset based on length checks for the tokenized prompts and baseline completions
    print("Step 2: Creating input length filter (lazy)...")
    dataset_input_len_filtered = dataset_w_prompts.filter(input_check, batched=False)
    # need to remove the input tensor column after this map
    # bc it persists between the prompt creation and generation maps
    columns_to_remove = args.columns_to_remove + ["input_ids"]

    # call the generation partial on each prompt in the dataset
    print("Step 3: Creating generation map (lazy)...")
    dataset_w_generations = dataset_input_len_filtered.map(
        generation_partial,
        batched=True,
        batch_size=args.generation_batch_size,
        remove_columns=columns_to_remove,
    )
    print("Data pipeline setup complete. Actual execution starts when iterating...")
    print(f"#" * 80)

    ###########################################################################
    # Main loop - actually executes the generation pipeline.
    # and accumulates the result rows in a list, assumes list is "small"-ish
    # and we aren't accumulating any tensors or other memory hogging artifacts
    ###########################################################################

    # Calculate target total samples based on generation_multiplier
    target_total_samples = int(args.min_generations * args.generation_multiplier)
    print(
        f"Generation target: {args.min_generations} valid samples (or {target_total_samples} total samples with multiplier={args.generation_multiplier})"
    )

    print(f"#" * 80)
    print("Creating iterator from dataset_w_generations...")
    print("Note: First iteration may be slow as it triggers data loading and first batch generation")
    print(f"Batch size: {args.generation_batch_size}")
    print(f"#" * 80)

    processed_examples = []
    ds_iterator = iter(dataset_w_generations)
    i = 0  # Valid samples (passed output_check)
    total_steps = 0  # Total samples processed
    pbar = tqdm(total=args.min_generations, desc="Valid samples")

    print("Starting main generation loop...")
    print(f"Fetching first batch (this may take a while)...")

    # Dual condition: stop when we have enough valid samples OR reach total sample limit
    while i < args.min_generations and total_steps < target_total_samples:
        try:
            ex = next(ds_iterator)
            total_steps += 1

            # Print confirmation after first successful fetch
            if total_steps == 1:
                print(f"✓ Successfully fetched first example! Generation is working.")
                print(f"Continuing with remaining samples...")
        except StopIteration:
            print(f"\nDataset iterator exhausted after {total_steps} samples.")
            break

        if args.verbose:
            # log basics to stdout
            print(f"#" * 80)
            print(f"dataset index: {ex['idx']}")
            print(f"orig_sample_length: {ex['orig_sample_length']}")
            print(f"prompt_length: {ex['prompt_length']}")
            print(f"real_completion_length: {ex['baseline_completion_length']}")
            print(f"no_wm_output_length: {ex['no_wm_output_length']}")
            print(f"w_wm_output_length: {ex['w_wm_output_length']}")

            print(f"\ntruncated_input: ")
            print(ex["truncated_input"])
            print(f"\nbaseline_completion: ")
            print(ex["baseline_completion"])
            print(f"\nno_wm_output: ")
            print(ex["no_wm_output"])
            print(f"\nw_wm_output: ")
            print(ex["w_wm_output"])

        processed_examples.append(ex)

        if output_check(ex):
            i += 1
            pbar.update(1)
        else:
            # Print progress every 10 samples to avoid spamming
            if total_steps % 10 == 0:
                print(
                    f"\n{i} valid / {total_steps} total samples processed so far.",
                    f"\nValid ratio: {round(i/total_steps if total_steps > 0 else 0, 3)}",
                    f"\nTarget: {args.min_generations} valid samples (or {target_total_samples} total samples)",
                )
        # if using wandb, log progress to wandb
        if args.wandb:
            run.log(
                {
                    "num_valid_samples": i,
                    "num_total_samples": total_steps,
                    "valid_ratio": i / total_steps if total_steps > 0 else 0,
                    "progress_ratio": i / args.min_generations,
                    "generation_overhead_ratio": total_steps / (i + 1),
                },
                step=total_steps,
            )
    pbar.close()

    print(
        f"#" * 80,
        f"\nGeneration completed: {i} valid samples from {total_steps} total samples.",
        f"\nValid ratio: {round(i/total_steps if total_steps > 0 else 0, 3)}",
        f"\nTarget was: {args.min_generations} valid samples.",
    )
    if i < args.min_generations:
        print(
            f"#" * 80,
            f"\nWarning, may have run out of data before {args.min_generations} satisfactory samples were generated. ",
            f"\nNote, raw dataset limit was {args.limit_indices} rows.",
            f"\n{len(processed_examples)} prompt passed input checks and yielded generations, and {i} passed output checks,",
            f"\nProgress made: {round(i/args.min_generations, 2)}",
        )

    ###########################################################################
    # Generation jsonl dumping
    ###########################################################################

    gen_table_meta_path = f"{args.output_dir}/gen_table_meta.json"
    gen_table_path = f"{args.output_dir}/gen_table.jsonl"
    safe_gen_table_path = f"{args.output_dir}/gen_table_safe.jsonl"

    args.gen_table_already_existed = False

    if os.path.exists(gen_table_path):
        args.gen_table_already_existed = True
        print(f"Found existing generation files at this output dir: {args.output_dir}")
        if args.overwrite:
            print("Overwriting old generation files.")
            gen_table_path = gen_table_path
        else:
            print(
                f"Writing generations at alternate, safe path and exiting. Note! this only works once. "
                f"Safe version will get overwritten next time ... "
            )
            gen_table_path = safe_gen_table_path

    gen_table_meta = args.__dict__
    gen_table = processed_examples

    write_jsonlines(gen_table, gen_table_path)
    write_json(gen_table_meta, gen_table_meta_path, indent=4)

    # finish the wandb run
    if args.wandb:
        run.finish()
    return  # reload in separate script for metric measurement


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run watermarked huggingface LM generation pipeline"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-1.3b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=True,
        help="Whether to run model in float16 precsion.",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="c4",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--lfqa_source",
        type=str,
        default="hf",
        choices=["local", "hf"],
        help="For dataset_name=lfqa, choose between the original local JSONL ('local') "
             "and a full HF-based LFQA dataset ('hf'). Ignored for other datasets.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="realnewslike",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="The split of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--stream_dataset",
        type=str2bool,
        default=True,
        help="Whether to stream the dataset from the web or download it locally.",
    )
    parser.add_argument(
        "--columns_to_remove",
        type=str,
        default=None,
        help="Comma separated list of columns to remove from the dataset before generation.",
    )
    parser.add_argument(
        "--shuffle_dataset",
        type=str2bool,
        default=False,
        help="Whether to shuffle the dataset before sampling.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=1234,
        help="The seed to use for dataset shuffle op.",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=10_000,
        help="The buffer size to use for dataset shuffle op - takes n rows first, then shuffles those indices",
    )
    parser.add_argument(
        "--prompt_id",
        type=int,
        default=0,
        help="If the dataset supports multiple instruction prompts, denotes which one to use. 0 is default/no prompt.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="The number of tokens to generate using the model, and the num tokens removed from real text sample",
    )
    parser.add_argument(
        "--min_prompt_tokens",
        type=int,
        default=50,  # 500
        help="The number of examples (first N) to process from the dataset.",
    )
    parser.add_argument(
        "--min_sample_tokens",
        type=int,
        default=0,
        help="The the minimum length of raw prompt samples to consider.",
    )
    parser.add_argument(
        "--limit_indices",
        type=int,
        default=None,
        help="The number of examples (first N) to pull from the dataset, if None, pull all, and then set this arg to the number of rows in the dataset.",
    )
    parser.add_argument(
        "--min_generations",
        type=int,
        default=500,
        help="The minimum number of valid generations according to the output check strat to sample.",
    )
    parser.add_argument(
        "--input_truncation_strategy",
        type=str,
        default="completion_length",
        choices=["no_truncation", "completion_length", "prompt_length"],
        help="The strategy to use when tokenizing and truncating raw inputs to make prompts.",
    )
    parser.add_argument(
        "--apply_chat_template",
        type=str2bool,
        default=True,
        help=(
            "Whether to wrap raw text with the tokenizer's chat_template (if available) "
            "before generation. Set to False to feed plain text directly."
        ),
    )
    parser.add_argument(
        "--input_filtering_strategy",
        type=str,
        default="completion_length",
        choices=["no_filter", "completion_length", "prompt_length", "prompt_and_completion_length"],
        help="The strategy to use when tokenizing and truncating raw inputs to make prompts.",
    )
    parser.add_argument(
        "--output_filtering_strategy",
        type=str,
        default="no_filter",
        choices=["no_filter", "max_new_tokens", "length_window"],
        help=(
            f"The strategy to use when filtering/skipping rows if the model didn't ",
            f"generate enough tokens to facilitate analysis.",
        ),
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=False,
        help=("Whether to perform sampling during generation. (non-greedy decoding)"),
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="The temperature to use when generating using multinom sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="The top k to use when generating using top_k version of multinom sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="The top p to use when generating using top_p version of sampling",
    )
    parser.add_argument(
        "--typical_p",
        type=float,
        default=1.0,
        help="The typical p to use when generating using typical decoding version of multinom sampling",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="The number of beams to use where '1' is no beam search.",
    )
    parser.add_argument(
        "--repeat_penalty",
        type=float,
        default=1.0,
        help="Penalty term for greedy decoding. 1 means no penalty",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=1,
        help="Seed for setting the torch rng prior to generation using any decoding scheme with randomness.",
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=4,
        help="The batch size to use for generation.",
    )
    parser.add_argument(
        "--empty_cache_between_batches",
        type=str2bool,
        default=True,
        help="If True, call torch.cuda.empty_cache() and gc.collect() after each generation batch to mitigate OOM.",
    )
    parser.add_argument(
        "--watermark_type",
        type=str,
        default="multibit",
        choices=["multibit", "MPAC", "quantile", "quantile_black", "stealthink", "unbiased"],
        help=(
            "The type of watermark to use: "
            "multibit/MPAC (original), quantile (interval-based), quantile_black (shared-shuffle, black-box-detectable), "
            "stealthink (reweight-based), or unbiased (zero-bit, LLR-scored)."
        ),
    )
    parser.add_argument(
        "--chunk_capacity",
        type=int,
        default=3,
        help="Bits per chunk for quantile watermark, determines number of buckets M=2^chunk_capacity. Only used when watermark_type=quantile.",
    )
    parser.add_argument(
        "--mapping_scheme",
        type=str,
        default="identity",
        choices=["identity", "cyclic", "permute"],
        help="Bucket-to-interval mapping scheme for quantile watermark.",
    )
    parser.add_argument(
        "--mapping_key",
        type=str,
        default="quantile-map-key-v1",
        help="Optional secret key to salt the mapping hash for quantile watermark.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Epsilon shrink for CDF interval boundaries in quantile watermark (0<=eps<0.5).",
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
        "--unbiased_ignore_history_generation",
        type=str2bool,
        default=True,
        help="If True, Unbiased watermark ignores history when generating (always applies reweighting).",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="The seeding procedure to use for the watermark.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="The ratio of tokens to put in the greenlist when splitting the vocabulary",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="The amount of bias (absolute) to add to the logits in the whitelist half of the vocabulary "
             "at every step",
    )
    parser.add_argument(
        "--store_spike_ents",
        type=str2bool,
        default=True,
        help=("Whether to store the spike entropies while generating with watermark processor. "),
    )
    # multi-bit configuration
    parser.add_argument(
        "--message_length",
        type=int,
        default=4,
        help="Number of bits of message to watermark",
    )
    parser.add_argument(
        "--fixed_message",
        type=str,
        default=None,
        help=(
            "Optional fixed binary message to embed (e.g., '01'). "
            "If shorter than message_length it will be left-padded with zeros. "
            "Overrides random per-batch message sampling."
        ),
    )
    parser.add_argument(
        "--code_length",
        type=int,
        default=4,
        help="Length of the actual code to watermark when using error correcting algoritm",
    )
    parser.add_argument(
        "--base",
        type=int,
        default=4,
        help="Base (radix) of message. Defaults to bit message.",
    )
    parser.add_argument(
        "--zero_bit",
        type=str2bool,
        default=False,
        help="When true, this is a special case of zero-bit; all messages are set to 0.",
    )
    parser.add_argument(
        "--use_position_prf",
        type=str2bool,
        default=False,
        help="When true, the position seed will be determined by a different prf scheme"
    )
    parser.add_argument(
        "--use_fixed_position",
        type=str2bool,
        default=False,
        help="When true, the position seed will be sampled with a fixed seed (rotation)"
    )
    parser.add_argument(
        "--use_feedback",
        type=str2bool,
        default=False,
        help="When true, encoding will do error correcting using feedbacks"
    )
    parser.add_argument(
        "--feedback_bias",
        type=float,
        default=2,
        help="magnitude of bias when using feedback"
    )
    parser.add_argument(
        "--feedback_eta",
        type=int,
        default=2,
        help="number of tokens per position to observe after staring to correct"
    )
    parser.add_argument(
        "--feedback_tau",
        type=int,
        default=2,
        help="Parameter of condition 1"
    )
    # logging
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=False,
        help="Whether to log the generations to stdout.",
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
        "--output_dir",
        type=str,
        default="./output",
        help="The unique name for the run.",
    )
    parser.add_argument(
        "--overwrite",
        type=str2bool,
        default=False,
        help="Allow overwriting of old generation files at the same output location.",
    )
    parser.add_argument(
        "--generation_multiplier",
        type=float,
        default=2.0,
        help="Multiplier for generation: generate up to min_generations * generation_multiplier samples to have more data for filtering.",
    )
    parser.add_argument(
        "--target_T",
        type=int,
        default=0,
        help="Target generation length T used for output filtering windows (0 => use max_new_tokens).",
    )
    parser.add_argument(
        "--lower_tolerance_T",
        type=int,
        default=0,
        help="Lower tolerance for output length window; valid samples have length >= target_T - lower_tolerance_T when using length_window filtering.",
    )

    args = parser.parse_args()
    assert not args.zero_bit or (args.zero_bit and args.message_length == 1), "If conducting zero-bit experiment, " \
                                                                              "message length should be 1."
    # Validate generation_multiplier
    assert args.generation_multiplier >= 1.0, "generation_multiplier must be >= 1.0 to ensure enough samples are generated."
    if args.fixed_message is not None:
        msg = args.fixed_message.strip()
        if msg.startswith("0b"):
            msg = msg[2:]
        msg = msg.replace(" ", "").replace("_", "")
        if any(c not in "01" for c in msg):
            raise ValueError("--fixed_message must be a binary string containing only 0/1.")
        if len(msg) > args.message_length:
            raise ValueError(
                f"--fixed_message length ({len(msg)}) exceeds --message_length ({args.message_length})."
            )
        msg = msg.zfill(args.message_length)
        if args.zero_bit and set(msg) != {"0"}:
            raise ValueError("--zero_bit requires the embedded message to be all zeros.")
        args.fixed_message = msg
    ###########################################################################
    # Argument validation and conditional setting
    ###########################################################################
    # for removing some columns to save space
    args.columns_to_remove = args.columns_to_remove.split(",") if args.columns_to_remove else []

    # if decoding scheme is not sampling, then set generation seed to None
    # to avoid confusion and calling the torch rng unnecessarily
    args.generation_seed = args.generation_seed if args.use_sampling else None

    # -1 value for min_generations means no specified minimum
    # with the assumption that the
    if args.min_generations <= 0:
        args.min_generations = MAX_GENERATIONS
        print(
            f"Warning: min_generations is -1. A hardcoded value of {MAX_GENERATIONS} will be used to limit the generation loop."
        )

    # Set a sensible default for target_T used in length_window filtering.
    if args.target_T <= 0:
        args.target_T = args.max_new_tokens
    # Clamp lower_tolerance_T to [0, target_T]
    if args.lower_tolerance_T < 0:
        args.lower_tolerance_T = 0
    if args.lower_tolerance_T > args.target_T:
        args.lower_tolerance_T = args.target_T

    if args.limit_indices is None:
        print("No limit_indices specified, pulling all examples from the dataset.")
    else:
        print(f"Limiting iteration to {args.limit_indices} examples from the dataset.")

    # split wandb tags
    if args.wandb_tags != "":
        args.wandb_tags = args.wandb_tags.split(",")
    else:
        args.wandb_tags = []

    # seed for randomly sampling message
    random.seed(0)
    main(args)
