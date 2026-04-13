from datasets import Dataset, IterableDataset, load_dataset
from utils.io import read_jsonlines
import os

prompts = {
    0: "",
    1: "Answer the following question in 200-300 words. Explain it like I'm five.\n\n",
}


def load_lfqa(args=None, path="./utils/data/lfqa.jsonl"):
    """
    Original local LFQA loader used in this repo. Reads from a preprocessed
    JSONL file with fields:
      - prefix
      - gold_completion
      - title
      - selftext
      - q_id
    """
    cols_to_load = ["prefix", "gold_completion", "title", "selftext", "q_id"]

    args.dataset_config_name = None
    args.dataset_split = None
    args.columns_to_remove = list(set(args.columns_to_remove + cols_to_load))

    def lfqa_generator():
        for ex in read_jsonlines(path):
            row = {k: ex[k] for k in cols_to_load}
            row["prefix"] = f"{prompts[args.prompt_id]}{row['prefix']}"
            yield row

    dataset = IterableDataset.from_generator(lfqa_generator)
    return dataset


def load_lfqa_hf(args=None, dataset_id: str = "~/data/adamjweintraut/eli5_lfqa_best"):
    """
    HF-based LFQA loader using the full ELI5 LFQA dataset on Hugging Face:
      - dataset: adamjweintraut/eli5_lfqa_best

    This dataset exposes (among others):
      - q_id: Reddit question id
      - question: question text
      - best_answer: a single selected answer
      - context/orig/target: combined question+answer views

    We map it into the same interface as the local LFQA loader by producing:
      - prefix: prompt to feed to the LM
      - gold_completion: reference long-form answer
      - title: question text (for compatibility)
      - selftext: empty string (no separate body here)
      - q_id: copied from dataset
    """
    assert args is not None, "args must be provided to load_lfqa_hf"

    cols_to_load = ["prefix", "gold_completion", "title", "selftext", "q_id"]
    split = args.dataset_split or "train"
    # Treat dataset_id as a local directory containing Parquet files.
    # Expected layout (as produced by `datasets` when saved locally):
    #   dataset_id/
    #     data/train-*.parquet
    #     data/validation-*.parquet
    #     data/test-*.parquet
    data_root = os.path.expanduser(dataset_id)
    data_dir = os.path.join(data_root, "data")

    def _transform_example(ex):
        question = ex.get("question", "") or ""
        best_answer = ex.get("best_answer", "") or ""
        q_id = ex.get("q_id", "")

        prefix = f"{prompts[args.prompt_id]}{question}\nA:"

        return {
            "prefix": prefix,
            "gold_completion": best_answer,
            "title": question,
            "selftext": "",
            "q_id": q_id,
        }

    data_files = {split: os.path.join(data_dir, f"{split}-*.parquet")}

    if args.stream_dataset:
        raw = load_dataset(
            "parquet",
            data_files=data_files,
            split=split,
            streaming=True,
        )

        def lfqa_hf_generator():
            for ex in raw:
                yield _transform_example(ex)

        dataset = IterableDataset.from_generator(lfqa_hf_generator)
    else:
        raw = load_dataset(
            "parquet",
            data_files=data_files,
            split=split,
            streaming=False,
        )
        dataset = raw.map(_transform_example, batched=False)

    args.dataset_config_name = None
    args.dataset_split = split
    args.columns_to_remove = list(set(args.columns_to_remove + cols_to_load))

    return dataset
