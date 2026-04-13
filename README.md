# QuantileMark

This is the official code release for the paper:

"**QuantileMark: A Message-Symmetric Multi-bit Watermark for LLMs**"

## Note

This repository is built upon the following works:
- "A Watermark for Large Language Models"
  ([Code](https://github.com/jwkirchenbauer/lm-watermarking) | [Paper](https://arxiv.org/abs/2301.10226))
- "Advancing Beyond Identification: Multi-bit Watermark for Large Language Models"
  ([Code](https://github.com/bangawayoo/mb-lm-watermarking) | [Paper](https://aclanthology.org/2024.naacl-long.224))

We also adapt the pipelines to make them more convenient for multi-bit watermarking experiments on instruction-tuned models. 

The main QuantileMark implementation is located in:

- `watermark_reliability_release/quantile_watermark_processor.py`

The main experiment script is:

- `watermark_reliability_release/quantile.sh`

## Requirements

```bash
pip install -r requirements.txt
pip install -r watermark_reliability_release/requirements.txt
```



## Quick Start

To run QuantileMark and reproduce results reported in the paper:

```bash
cd watermark_reliability_release
bash quantile.sh
```

This script runs the full pipeline, using `Qwen-2.5-7B-Instruct` on LFQA:

1. generation
2. attack
3. evaluation (watermark detection)


## Before Running

`quantile.sh` contains several environment variables that should usually be edited before execution:

- `CUDA_VISIBLE_DEVICES`: select the GPU
- `MODEL_PATH`: path or Hugging Face name of the base model
- `HF_HOME`, `HF_TOKEN`, `HF_ENDPOINT`: Hugging Face cache and access settings
- `OUTPUT_DIR`: output directory
- `D_NAME`: dataset name
- `RUN_GEN`, `RUN_ATT`, `RUN_EVAL`: enable or disable each stage


`quantile.sh` defaults to sweeping multiple watermark types when `WATERMARK_TYPE` is not set. To run **only QuantileMark**, use:


```bash
WATERMARK_TYPE=quantile bash quantile.sh
```


## Useful Configuration

The most important QuantileMark-related variables in `quantile.sh` are:

- `MSG_LEN`: message length in bits
- `CHUNK_CAPACITY`: bits per symbol
- `SEED_SCH`: seeding scheme
- `MAP_SCHEME`: message-to-interval mapping scheme
- `topk`: generation top-k
- `TOKEN_LEN`: generation length





