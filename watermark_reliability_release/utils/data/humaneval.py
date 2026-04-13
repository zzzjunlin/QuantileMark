from datasets import load_dataset


def load_humaneval(args=None):
    """
    Load the HumanEval dataset from Hugging Face in a form that is
    compatible with the existing generation pipeline.

    The HF dataset `openai/openai_humaneval` has the following fields:
      - task_id
      - prompt
      - canonical_solution
      - test
      - entry_point

    We treat:
      - `prompt` as the model input (prefix)
      - `canonical_solution` as the reference completion

    Downstream, `load_hf_dataset` will set:
      - truncate_input_for_prompt = False
      - input_col_name = "prompt"
      - ref_output_col_name = "canonical_solution"
    so `tokenize_for_generation` can reuse the same logic as for LFQA/essays.
    """
    assert args is not None, "args must be provided to load_humaneval"

    # HumanEval only has a 'test' split; default to it if not specified.
    split = args.dataset_split or "test"

    dataset = load_dataset(
        "openai/openai_humaneval",
        split=split,
        streaming=False,
    )

    # Do not modify the core HF fields; we just record which columns can be
    # dropped after generation to save space. Keep task_id/test/entry_point
    # so that downstream scripts can run unit tests if desired.
    cols_to_remove = ["prompt", "canonical_solution"]

    # Normalise dataset-related args for consistency with other loaders.
    args.dataset_config_name = None
    args.dataset_split = split
    args.columns_to_remove = list(set(getattr(args, "columns_to_remove", []) + cols_to_remove))

    return dataset

