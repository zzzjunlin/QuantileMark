#!/bin/bash
# Quantile Watermark Experiment Script
if [ -z "${_RUN_QWM_REENTRANT:-}" ]; then
  # Default sweep when WATERMARK_TYPE is not provided.
  DEFAULT_TYPES="quantile stealthink MPAC"
  # DEFAULT_TYPES="quantile"
  export CUDA_VISIBLE_DEVICES=3
  if [ -n "${WATERMARK_TYPE:-}" ]; then
    IFS=',' read -r -a _types <<< "${WATERMARK_TYPE}"
  else
    _types=(${DEFAULT_TYPES})
  fi

  for _wt in "${_types[@]}"; do
    export WATERMARK_TYPE="${_wt}"
    echo "==== Running watermark pipeline with WATERMARK_TYPE=${WATERMARK_TYPE} ===="
    _RUN_QWM_REENTRANT=1 bash "$0" "$@"
  done
  exit 0
fi

# wandb offline
export OPENAI_API_KEY=""
export WANDB_API_KEY=""
# export CUDA_VISIBLE_DEVICES=1
export HF_HOME="~/.cache/huggingface"
export HF_ACCESS_TOKEN=""
export WANDB=T
export HF_DATASETS_OFFLINE=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export WANDB_MODE=offline
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=~/lib:$LD_LIBRARY_PATH
export HF_TOKEN=""
RANDOM_PORT=$(shuf -i 10000-65535 -n 1)
export HF_ENDPOINT=https://huggingface.co
export MASTER_PORT=$RANDOM_PORT

### Experiment type ###
export RUN_GEN=T; export RUN_ATT=T; export RUN_EVAL=T; export DEBUG=T;
export TEMPLATE=T
export LIMIT_ROWS=30000

### Model
export llama_instruct="meta-llama/Meta-Llama-3.1-8B-Instruct"
export qwen_instruct="Qwen/Qwen2.5-7B-Instruct"

### Generation ###
export MODEL_PATH=$qwen_instruct; export BS=32; export TOKEN_LEN=300;
export D_NAME="c4"; export D_CONFIG="realnewslike"; export INPUT_FILTER="prompt_and_completion_length"
export D_NAME="lfqa"
# export D_NAME="deepmind/code_contests"
if [ $D_NAME == "lfqa" ]
then
  export INPUT_FILTER="completion_length"
    # export INPUT_FILTER="no_filter"
fi
# export INPUT_FILTER="no_filter"
export NUM_BEAMS=1; export SAMPLING=T
export FP16=T; export MIN_GEN=500  # Target number of generations for generation (after filtering) and detection. Use a smaller value for a quick run.
export GENERATION_MULTIPLIER=11.0 # Sample at most $GENERATION_MULTIPLIER times the required number of generations to obtain enough outputs with valid $TOKEN_LEN.
export EARLY_FILTERING=T

### Watermark Configuration ###
# export WATERMARK_TYPE="quantile"  # Use quantile watermark
export MSG_LEN=24                 # Total embedded bits
export CHUNK_CAPACITY=2          # m = 2 bits / token. M = 2^2 = 4 buckets. 
export SEED_SCH="lefthash_3"         # hash seeding scheme
# export SEED_SCH="selfhash"
export GAMMA=0.5                   # Gamma parameter (not used for splitting in quantile)
if [ "$WATERMARK_TYPE" = "MPAC" ]; then
  export GAMMA=0.25
else
  export GAMMA=0.5 # StealthInk use this configuration.
fi

## logging

export OUTPUT_DIR="./experiments/${WATERMARK_TYPE}-run/"

### Attack ###
# Choose one of the following attacks by setting ATTACK_M:
#   - copy-paste
#   - dipper
#   - scramble
#   - word-deletion
#   - synonym-basic
#   - synonym-context

# For paraphrase attacks
# export ATTACK_M="dipper"
# export CP_ATT_TYPE="single-single" 

# export ATTACK_SUFFIX="cp=.2"
# export ATTACK_SUFFIX="dipper"

# Parameters for attacks
export srcp="80%"                    # copy-paste: insertion length (e.g., '80%' or '20')
export DEL_RATE="10%"                # word-deletion: deletion rate (e.g., '5%', '10%', or 0.1)
export KEEP_STRUCT=T                 # word-deletion: keep sentence structure
export SYN_RATE="20%"                # synonym attacks: substitution rate (e.g., '20%', 0.2)
# To run a specific new attack, set e.g.:
# export ATTACK_M="word-deletion"; export ATTACK_SUFFIX="wd=10%"
export ATTACK_M="synonym-basic";  export ATTACK_SUFFIX="synB=20%"
# export ATTACK_M="synonym-context";export ATTACK_SUFFIX="synC=20%"; export SYN_RATE="20%"

### Evaluation ###
export LOWER_TOL=0; export UPPER_TOL=0

export ORACLE_MODEL=$MODEL_PATH # PPL compute model.
export IGNORE_R_NGRAM=T
export EVAL_METRICS="all_w_ppl"
# export EVAL_METRICS="z-score"

## quantile setting
export MAP_SCHEME="permute" # map same message symbol (e.g., "01") to different buckets according to context.
export epsilon=0 # not used, could be ignored. 
export topk=128
export MESSAGE="" # Fix a message to embed, otherswise random message will be generated for generation.

export RUN_NAME="$(basename $MODEL_PATH)/${WATERMARK_TYPE}-$GAMMA-template1-topk${topk}-L0-${D_NAME}-${MSG_LEN}b-${TOKEN_LEN}T-M${CHUNK_CAPACITY}-${SEED_SCH}-${epsilon}"
mkdir -p ${OUTPUT_DIR}/log/${RUN_NAME}

# Create modified run_pipeline.sh for quantile
# cat > run_quantile_pipeline_temp.sh << 'EOF'
# #!/bin/bash

# Generation
if [ "$RUN_GEN" = "T" ]; then
    echo "========================================="
    echo "Running Generation Pipeline (Quantile Watermark)"
    echo "========================================="

    python generation_pipeline.py \
        --model_name_or_path="$MODEL_PATH" \
        --dataset_name="$D_NAME" \
        --dataset_config_name="$D_CONFIG" \
        --max_new_tokens=$TOKEN_LEN \
        --min_prompt_tokens=50 \
        --limit_indices=$LIMIT_ROWS \
        --min_generations=$MIN_GEN \
        --generation_multiplier=$GENERATION_MULTIPLIER \
        --input_truncation_strategy="prompt_length" \
        --input_filtering_strategy="$INPUT_FILTER" \
        --output_filtering_strategy="length_window" \
        --target_T=$TOKEN_LEN \
        --lower_tolerance_T=$LOWER_TOL \
        --use_sampling=$SAMPLING \
        --sampling_temp=1.0 \
        --top_k=$topk \
        --top_p=1.0 \
        --num_beams=$NUM_BEAMS \
        --generation_batch_size=$BS \
        --watermark_type="$WATERMARK_TYPE" \
        --chunk_capacity=$CHUNK_CAPACITY \
        --seeding_scheme="$SEED_SCH" \
        --gamma=$GAMMA \
        --message_length=$MSG_LEN \
        --load_fp16=$FP16 \
        --use_gpu=True \
        --output_dir="$OUTPUT_DIR/$RUN_NAME" \
        --overwrite=True \
        --mapping_scheme="$MAP_SCHEME" \
        --apply_chat_template=$TEMPLATE \
        --code_length="$MSG_LEN" \
        --epsilon=$epsilon 2>&1 | tee -a ${OUTPUT_DIR}/log/${RUN_NAME}/output.log

    if [ $? -ne 0 ]; then
        echo "Generation failed!"
        exit 1
    fi
fi

# Attack
if [ "$RUN_ATT" = "T" ]; then
    echo "========================================="
    echo "Running Attack Pipeline"
    echo "========================================="

    python attack_pipeline.py \
        --input_dir="$OUTPUT_DIR/$RUN_NAME" \
        --output_dir="$OUTPUT_DIR/$RUN_NAME-$ATTACK_SUFFIX" \
        --attack_method="$ATTACK_M" \
        --cp_attack_type="$CP_ATT_TYPE" \
        --cp_attack_insertion_len="$srcp" \
        --deletion_rate="$DEL_RATE" \
        --keep_structure=$KEEP_STRUCT \
        --syn_sub_rate="$SYN_RATE" \
        --overwrite_output_file=True \
        --overwrite_args=True 2>&1 | tee -a ${OUTPUT_DIR}/log/${RUN_NAME}/output.log

    if [ $? -ne 0 ]; then
        echo "Attack failed!"
        exit 1
    fi
fi

# export MODEL_PATH=$qwen_instruct_14
# Evaluation
if [ "$RUN_EVAL" = "T" ]; then
    echo "========================================="
    echo "Running Evaluation Pipeline (Quantile Detection)"
    echo "========================================="

    if [ "$RUN_ATT" = "T" ]; then
        INPUT_DIR="$OUTPUT_DIR/$RUN_NAME-$ATTACK_SUFFIX"
    else
        INPUT_DIR="$OUTPUT_DIR/$RUN_NAME"
    fi
    # INPUT_DIR="$OUTPUT_DIR/$RUN_NAME-$ATTACK_SUFFIX"
    python evaluation_pipeline.py \
        --input_dir="$INPUT_DIR" \
        --output_dir="$INPUT_DIR" \
        --evaluation_metrics="$EVAL_METRICS" \
        --watermark_type="$WATERMARK_TYPE" \
        --chunk_capacity=$CHUNK_CAPACITY \
        --model_name_or_path="$MODEL_PATH" \
        --load_fp16=$FP16 \
        --seeding_scheme="$SEED_SCH" \
        --gamma=$GAMMA \
        --message_length=$MSG_LEN \
        --ignore_repeated_ngrams=$IGNORE_R_NGRAM \
        --detection_z_threshold=4.0 \
        --wandb=True \
        --overwrite_output_file=True \
        --overwrite_args=True \
        --early_filtering=$EARLY_FILTERING \
        --target_T=$TOKEN_LEN \
        --lower_tolerance_T=$LOWER_TOL \
        --upper_tolerance_T=$UPPER_TOL \
        --oracle_model_name_or_path="$ORACLE_MODEL" \
        --include_prompt_in_detection=False \
        --wandb_entity="" \
        --mapping_scheme="$MAP_SCHEME" \
        --wrap_output_in_chat_template=$TEMPLATE \
        --top_k=$topk \
        --temperature=1.0 \
        --zscore_T_list="" \
        --api_judge_preview False --debug=False \
        --api_judge_store_dimensions True --api_judge_store_reason True \
        --log_raw_series True --log_raw_tabular True --limit_rows -1 2>&1 | tee -a ${OUTPUT_DIR}/log/${RUN_NAME}/output.log

    if [ $? -ne 0 ]; then
        echo "Evaluation failed!"
        exit 1
    fi
fi

echo "========================================="
echo "Pipeline completed successfully!"
echo "========================================="
echo ""
echo "========================================="
echo "Bit Accuracy Results:"
echo "========================================="
# cat ${OUTPUT_DIR}/log/${RUN_NAME}/output.log | grep "bit_acc"
# cat ${OUTPUT_DIR}/log/${RUN_NAME}/output.log | grep "auc"
grep -E "bit_acc|auc|token_count_scored|bit_match_mean|ppl|z_score" "${OUTPUT_DIR}/log/${RUN_NAME}/output.log"

