#!/bin/bash
# Downstream Tasks Experiment Script (MT + Summarization)
# This script mirrors the style of run_quantile_watermark.sh but targets
# generic downstream experiments (machine translation on WMT16 En→Ro and
# summarization on CNN/DailyMail). It calls downstream_pipeline.py.

###############################################################################
# Global environment
###############################################################################
# Wandb / HF / CUDA environment (adapt as needed)
export OPENAI_API_KEY=""
export WANDB_API_KEY=""
export CUDA_VISIBLE_DEVICES=2
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-""}"
export WANDB=T

export WANDB_MODE=offline
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH="$HOME/lib:$LD_LIBRARY_PATH"
RANDOM_PORT=$(shuf -i 10000-65535 -n 1)
export MASTER_PORT=$RANDOM_PORT

###############################################################################
# Common watermark settings (shared by MT & summarization)
###############################################################################

# PRF and key settings (e.g., SHA-256 with a 1024-bit secret key)
export WM_PRF="sha256"
export WM_KEY_BITS=1024
# Texture key length h = 3 (last 3 tokens as context for PRF-like seeding)
export WM_H=3
# Sampling: pure multinomial with temperature 1.0
export WM_TEMPERATURE=1.0
export WM_DO_SAMPLE=True  # multinomial sampling, no top-k/top-p

# Unit capacities to evaluate (m = 1, 2; H = 1 so total bits per sample is m)
UNIT_CAP_LIST=("2")

# Watermark methods to compare.
#   - none: no watermark baseline
#   - quantile_watermark: QuantileMark
#   - mpac: MPAC baseline
#   - stealthink: StealthInk baseline
METHOD_LIST=("none" "quantile_watermark" "mpac" "stealthink")
# METHOD_LIST=("stealthink")
# Output root directory for all downstream experiments
export OUTPUT_ROOT="./experiments/downstream"
mkdir -p "${OUTPUT_ROOT}"

###############################################################################
# Machine Translation (MT) setup: WMT16 En→Ro with MBART
###############################################################################

export MT_TASK_NAME="mt_enro"
# Recommended HF model: facebook/mbart-large-en-ro (implementation choice)
export MT_MODEL_NAME="facebook/mbart-large-en-ro"

# WMT16 En–Ro test split: HF dataset `wmt16`, config `ro-en`
export MT_DATASET_NAME="wmt16"
export MT_DATASET_CONFIG="ro-en"
export MT_DATASET_SPLIT="test"

# Source / target language codes and fields
export MT_SRC_LANG="en"      # translation["en"]
export MT_TGT_LANG="ro"      # translation["ro"]

# Generation settings (implementation choice, consistent with paper description:
# multinomial sampling, T=1.0, sequences <~ 50 tokens)
export MT_MAX_NEW_TOKENS=100
export MT_BATCH_SIZE=256
export MT_RANDOM_SEED=42

###############################################################################
# Summarization setup: CNN/DailyMail with BART-large
###############################################################################

export SUM_TASK_NAME="summ_cnn_dm"
# Recommended HF model: facebook/bart-large-cnn (implementation choice)
export SUM_MODEL_NAME="facebook/bart-large-cnn"
export SUM_DATASET_NAME="cnn_dailymail"
export SUM_DATASET_CONFIG="3.0.0"
export SUM_DATASET_SPLIT="test"

# Article / summary fields
export SUM_SOURCE_FIELD="article"
export SUM_TARGET_FIELD="highlights"

# Generation settings (implementation choice; BART-CNN style lengths)
export SUM_MAX_NEW_TOKENS=60   # effective summary length; driver can override
export SUM_BATCH_SIZE=8
export SUM_RANDOM_SEED=42

###############################################################################
# Helper: run one configuration via Python driver
###############################################################################

run_config () {
  local TASK=$1          # "mt" or "summ"
  local METHOD=$2        # one of METHOD_LIST
  local UNIT_CAP=$3      # 1 or 2

  if [ "$TASK" = "mt" ]; then
    local RUN_TAG="${MT_TASK_NAME}-${METHOD}-m${UNIT_CAP}"
    local OUT_DIR="${OUTPUT_ROOT}/${MT_TASK_NAME}/${RUN_TAG}"
    mkdir -p "${OUT_DIR}"
    echo "========================================="
    echo "Running MT config: METHOD=${METHOD}, m=${UNIT_CAP}"
    echo "Output dir: ${OUT_DIR}"
    echo "========================================="

    python downstream_pipeline.py \
      --task "mt" \
      --run_name "${RUN_TAG}" \
      --output_dir "${OUT_DIR}" \
      --model_name_or_path "${MT_MODEL_NAME}" \
      --dataset_name "${MT_DATASET_NAME}" \
      --dataset_config_name "${MT_DATASET_CONFIG}" \
      --dataset_split "${MT_DATASET_SPLIT}" \
      --src_lang "${MT_SRC_LANG}" \
      --tgt_lang "${MT_TGT_LANG}" \
      --unit_capacity "${UNIT_CAP}" \
      --num_chunks 1 \
      --watermark_method "${METHOD}" \
      --prf_type "${WM_PRF}" \
      --key_bits "${WM_KEY_BITS}" \
      --texture_h "${WM_H}" \
      --temperature "${WM_TEMPERATURE}" \
      --do_sample "${WM_DO_SAMPLE}" \
      --max_new_tokens "${MT_MAX_NEW_TOKENS}" \
      --batch_size "${MT_BATCH_SIZE}" \
      --random_seed "${MT_RANDOM_SEED}" \
      --eval_metrics "bleu,rouge1,bertscore,ppl" \
      --mpac_delta 1.0 \
      2>&1 | tee "${OUT_DIR}/run.log"

    if [ $? -ne 0 ]; then
      echo "MT run failed for METHOD=${METHOD}, m=${UNIT_CAP}"
      exit 1
    fi

  elif [ "$TASK" = "summ" ]; then
    local RUN_TAG="${SUM_TASK_NAME}-${METHOD}-m${UNIT_CAP}"
    local OUT_DIR="${OUTPUT_ROOT}/${SUM_TASK_NAME}/${RUN_TAG}/$(basename ${SUM_MODEL_NAME})"
    mkdir -p "${OUT_DIR}"
    echo "========================================="
    echo "Running Summarization config: METHOD=${METHOD}, m=${UNIT_CAP}"
    echo "Output dir: ${OUT_DIR}"
    echo "========================================="

    python downstream_pipeline.py \
      --task "summ" \
      --run_name "${RUN_TAG}" \
      --output_dir "${OUT_DIR}" \
      --model_name_or_path "${SUM_MODEL_NAME}" \
      --dataset_name "${SUM_DATASET_NAME}" \
      --dataset_config_name "${SUM_DATASET_CONFIG}" \
      --dataset_split "${SUM_DATASET_SPLIT}" \
      --source_field "${SUM_SOURCE_FIELD}" \
      --target_field "${SUM_TARGET_FIELD}" \
      --unit_capacity "${UNIT_CAP}" \
      --num_chunks 12 \
      --watermark_method "${METHOD}" \
      --prf_type "${WM_PRF}" \
      --key_bits "${WM_KEY_BITS}" \
      --texture_h "${WM_H}" \
      --temperature "${WM_TEMPERATURE}" \
      --do_sample "${WM_DO_SAMPLE}" \
      --max_new_tokens "${SUM_MAX_NEW_TOKENS}" \
      --batch_size "${SUM_BATCH_SIZE}" \
      --random_seed "${SUM_RANDOM_SEED}" \
      --eval_metrics "rouge1,bertscore,ppl" \
      --mpac_delta 1.0 \
      --debug_metrics \
      2>&1 | tee "${OUT_DIR}/run.log"

    if [ $? -ne 0 ]; then
      echo "Summarization run failed for METHOD=${METHOD}, m=${UNIT_CAP}"
      exit 1
    fi
  else
    echo "Unknown TASK: ${TASK}"
    exit 1
  fi
}

###############################################################################
# Main loop over methods and unit capacities
###############################################################################

# Flags to enable/disable tasks
RUN_MT=${RUN_MT:-T}
RUN_SUM=${RUN_SUM:-T}

# if [ "$RUN_MT" = "T" ]; then
#   for METHOD in "${METHOD_LIST[@]}"; do
#     for CAP in "${UNIT_CAP_LIST[@]}"; do
#       run_config "mt" "${METHOD}" "${CAP}"
#     done
#   done
# fi

if [ "$RUN_SUM" = "T" ]; then
  for METHOD in "${METHOD_LIST[@]}"; do
    for CAP in "${UNIT_CAP_LIST[@]}"; do
      run_config "summ" "${METHOD}" "${CAP}"
    done
  done
fi

echo "========================================="
echo "Downstream experiments completed."
echo "Outputs at: ${OUTPUT_ROOT}"
echo "========================================="
