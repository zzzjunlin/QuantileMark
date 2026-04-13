from datasets import load_dataset, IterableDataset


def _transform_example(ex):
    """
    Transform a raw deepmind/code_contests example into the fields
    expected by the generation pipeline.

    We expose:
      - prompt: problem description (+ optional examples)
      - canonical_solution: one reference solution (prefer Python if available)
      - task_name: problem name/id
      - public_tests / private_tests / generated_tests: carried through for
        potential downstream correctness evaluation.
    """
    name = ex.get("name", "")
    desc = ex.get("description", "") or ""

    # Build prompt from description plus public I/O examples.
    public_tests = ex.get("public_tests") or {}
    inputs = public_tests.get("input") or []
    outputs = public_tests.get("output") or []

    example_blocks = []
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        header = f"Example {i + 1}:"
        example_blocks.append(
            header + "\nInput:\n" + inp + "\nOutput:\n" + out
        )
    examples_str = "\n\n".join(example_blocks) if example_blocks else ""

    # High-level instruction to encourage code-only answers.
    instruction = (
        "Write a correct and efficient program in Python that solves the following problem. "
        "Output only the complete code for the solution, without any explanations or prose."
    )

    prompt_parts = [instruction]
    if name:
        prompt_parts.append(str(name))
    if desc:
        prompt_parts.append(desc)
    if examples_str:
        prompt_parts.append(examples_str)
    prompt = "\n\n".join(prompt_parts)

    # Choose a canonical solution: prefer language code 1 (Python) if present.
    sols = ex.get("solutions") or {}
    langs = sols.get("language") or []
    codes = sols.get("solution") or []
    canonical_solution = ""
    if codes:
        idx = 0
        for i, lang in enumerate(langs):
            # In the original dataset, language==1 corresponds to Python.
            if lang == 1:
                idx = i
                break
        canonical_solution = codes[idx]

    return {
        "prompt": prompt,
        "canonical_solution": canonical_solution,
        "task_name": name,
        "source": ex.get("source"),
        "difficulty": ex.get("difficulty"),
        "public_tests": public_tests,
        "private_tests": ex.get("private_tests") or {},
        "generated_tests": ex.get("generated_tests") or {},
    }


def load_code_contests(args=None):
    """
    Load deepmind/code_contests in a form compatible with the generation pipeline.

    The pipeline will later set:
      - truncate_input_for_prompt = False
      - input_col_name = "prompt"
      - ref_output_col_name = "canonical_solution"
    so that tokenize_for_generation can reuse the LFQA/essays path.
    """
    assert args is not None, "args must be provided to load_code_contests"

    split = args.dataset_split or "train"

    if args.stream_dataset:
        raw = load_dataset(
            "deepmind/code_contests",
            split=split,
            streaming=True,
        )

        def generator():
            for ex in raw:
                yield _transform_example(ex)

        dataset = IterableDataset.from_generator(generator)
    else:
        raw = load_dataset(
            "deepmind/code_contests",
            split=split,
            streaming=False,
        )
        dataset = raw.map(_transform_example, batched=False)

    # Normalise dataset-related args and record which columns to drop after generation.
    args.dataset_config_name = None
    args.dataset_split = split
    # We will later remove the large text fields to keep output lean.
    cols_to_remove = ["prompt", "canonical_solution"]
    args.columns_to_remove = list(set(getattr(args, "columns_to_remove", []) + cols_to_remove))

    return dataset
