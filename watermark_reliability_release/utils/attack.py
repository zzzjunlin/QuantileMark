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

import openai
import random
import re
from typing import List, Tuple

from utils.evaluation import OUTPUT_TEXT_COLUMN_NAMES
from utils.copy_paste_attack import single_insertion, triple_insertion_single_len, k_insertion_t_len

SUPPORTED_ATTACK_METHODS = [
    "gpt",
    "dipper",
    "copy-paste",
    "scramble",
    # New attacks (Appendix E)
    "word-deletion",
    "synonym-basic",
    "synonym-context",
]

# Try optional NLTK resources for synonym attacks
try:  # lightweight import, will be absent if nltk not installed or corpora missing
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.wsd import lesk
except Exception:  # pragma: no cover - keep pipeline robust if deps absent
    wn = None
    lesk = None

# A small, built-in list of English stopwords to avoid hard nltk downloads
_FALLBACK_STOPWORDS = {
    "a","an","the","and","or","but","if","while","with","without","on","in","at","to",
    "for","from","by","of","is","am","are","was","were","be","being","been","do","does",
    "did","done","have","has","had","i","you","he","she","it","we","they","this","that",
    "these","those","as","not","no","so","than","too","very","can","could","may","might",
    "will","would","shall","should","must","about","into","over","under","again","once","here",
    "there","when","where","why","how","all","any","both","each","few","more","most","other",
    "some","such","own","same","just","also","only","then","now"
}

# -------------------------
# Helpers for text handling
# -------------------------

_WORD_RE = re.compile(r"\w+")


def _tokenize_words_and_punct(text: str) -> List[str]:
    """Naive, dependency-free tokenizer splitting into words and punctuation.

    Keeps punctuation as separate tokens so we can drop/replace words without
    mangling punctuation. Avoids heavy NLTK punkt downloads.
    """
    # Split on word boundaries while keeping punctuation as tokens
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return tokens


def _reconstruct_text(tokens: List[str]) -> str:
    """Reconstruct text from tokens (words and punctuation).

    Joins with spaces and then fixes spacing before common punctuation.
    """
    if not tokens:
        return ""
    txt = " ".join(tokens)
    # Remove space before punctuation like .,!?;:)
    txt = re.sub(r"\s+([.,!?;:%\)\]\}])", r"\1", txt)
    # Remove space after opening punctuation like ( [ {
    txt = re.sub(r"([\(\[\{])\s+", r"\1", txt)
    # Compact quotes a bit: space before ' or " is usually not desired
    txt = re.sub(r"\s+([\"'])", r"\1", txt)
    return txt


def _select_word_indices(tokens: List[str]) -> List[int]:
    """Return indices of tokens that are pure words (no punctuation)."""
    return [i for i, tok in enumerate(tokens) if _WORD_RE.fullmatch(tok)]


def _chunk_by_sentences(tokens: List[str]) -> List[Tuple[int, int]]:
    """Very light sentence boundary detection based on ., !, ? tokens.

    Returns list of (start_idx, end_idx) token spans for sentences. end is inclusive.
    """
    spans = []
    start = 0
    for i, tok in enumerate(tokens):
        if tok in [".", "!", "?"]:
            spans.append((start, i))
            start = i + 1
    # tail
    if start < len(tokens):
        spans.append((start, len(tokens) - 1))
    return spans


def scramble_attack(example, tokenizer=None, args=None):
    # check if the example is long enough to attack
    for column in ["w_wm_output", "no_wm_output"]:
        if not check_output_column_lengths(example, min_len=args.cp_attack_min_len):
            # # if not, copy the orig w_wm_output to w_wm_output_attacked
            # NOTE changing this to return "" so that those fail/we can filter out these examples
            example[f"{column}_attacked"] = ""
            example[f"{column}_attacked_length"] = 0
        else:
            sentences = example[column].split(".")
            random.shuffle(sentences)
            example[f"{column}_attacked"] = ".".join(sentences)
            example[f"{column}_attacked_length"] = len(
                tokenizer(example[f"{column}_attacked"])["input_ids"]
            )

    # Handle sampled_positions fields for evaluation
    if "sampled_positions" in example:
        example["w_wm_output_sampled_positions"] = example["sampled_positions"]
    example["w_wm_output_attacked_sampled_positions"] = ""

    return example


def gpt_attack(example, attack_prompt=None, args=None):
    assert attack_prompt, "Prompt must be provided for GPT attack"

    gen_row = example

    if args.no_wm_attack:
        original_text = gen_row["no_wm_output"]
    else:
        original_text = gen_row["w_wm_output"]

    attacker_query = attack_prompt + original_text
    query_msg = {"role": "user", "content": attacker_query}

    from tenacity import retry, stop_after_attempt, wait_random_exponential

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(25))
    def completion_with_backoff(model, messages, temperature, max_tokens):
        return openai.ChatCompletion.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )

    outputs = completion_with_backoff(
        model=args.attack_model_name,
        messages=[query_msg],
        temperature=args.attack_temperature,
        max_tokens=args.attack_max_tokens,
    )

    attacked_text = outputs.choices[0].message.content
    assert (
        len(outputs.choices) == 1
    ), "OpenAI API returned more than one response, unexpected for length inference of the output"
    example["w_wm_output_attacked_length"] = outputs.usage.completion_tokens
    example["w_wm_output_attacked"] = attacked_text
    if args.verbose:
        print(f"\nOriginal text (T={example['w_wm_output_length']}):\n{original_text}")
        print(f"\nAttacked text (T={example['w_wm_output_attacked_length']}):\n{attacked_text}")

    # Handle sampled_positions fields for evaluation
    if "sampled_positions" in example:
        example["w_wm_output_sampled_positions"] = example["sampled_positions"]
    example["w_wm_output_attacked_sampled_positions"] = ""

    return example


def dipper_attack(dataset, lex=None, order=None, args=None):
    # Lazy import to avoid heavy deps and downloads at import time
    from utils.dipper_attack_pipeline import generate_dipper_paraphrases
    dataset = generate_dipper_paraphrases(dataset, lex=lex, order=order, args=args)
    return dataset


# -------------------------
# Appendix E: Word Deletion
# -------------------------

def word_deletion_attack(example, tokenizer=None, deletion_rate=0.1, keep_structure=True, args=None):
    """Delete a fraction of words from the text.

    - keep_structure=False: globally sample words to delete uniformly at random.
    - keep_structure=True: delete per-sentence, avoid first/last word in a sentence,
      and avoid deleting adjacent words to preserve readability.
    """
    # normalize rate (accept strings like "10%" or floats)
    if isinstance(deletion_rate, str) and deletion_rate.endswith("%"):
        try:
            deletion_rate = float(deletion_rate[:-1]) / 100.0
        except Exception:
            deletion_rate = 0.1
    try:
        deletion_rate = float(deletion_rate)
    except Exception:
        deletion_rate = 0.1
    deletion_rate = max(0.0, min(1.0, deletion_rate))

    for column in ["w_wm_output", "no_wm_output"]:
        if column not in example or not isinstance(example[column], str):
            # Set attacked fields empty to be filtered downstream
            example[f"{column}_attacked"] = ""
            example[f"{column}_attacked_length"] = 0
            continue

        original_text = example[column]
        tokens = _tokenize_words_and_punct(original_text)
        word_idxs = _select_word_indices(tokens)
        if len(word_idxs) == 0 or deletion_rate == 0:
            attacked_text = original_text
        else:
            total_to_delete = int(round(len(word_idxs) * deletion_rate))
            total_to_delete = max(1, total_to_delete) if deletion_rate > 0 else 0

            delete_set = set()
            if not keep_structure:
                delete_set = set(random.sample(word_idxs, min(total_to_delete, len(word_idxs))))
            else:
                # Per-sentence strategy: distribute deletions across sentences,
                # avoid deleting the first/last word per sentence, avoid adjacency.
                sent_spans = _chunk_by_sentences(tokens)
                remaining = total_to_delete
                for s_start, s_end in sent_spans:
                    # candidate word indices within sentence
                    cand = [i for i in range(s_start, s_end + 1) if i in word_idxs]
                    if len(cand) <= 2:
                        continue  # preserve very short sentences
                    # avoid first/last content word in the sentence
                    inner = cand[1:-1]
                    if not inner:
                        continue
                    # number to delete from this sentence proportional to length
                    k = int(round(len(inner) * deletion_rate))
                    k = min(k, max(0, len(inner) // 2))  # avoid deleting too many
                    # sample without adjacency: simple greedy retry
                    attempts = 0
                    chosen = set()
                    while attempts < 10 * (k + 1) and len(chosen) < k:
                        idx = random.choice(inner)
                        # avoid neighbors
                        if (idx not in chosen) and (idx - 1 not in chosen) and (idx + 1 not in chosen):
                            chosen.add(idx)
                        attempts += 1
                    delete_set.update(chosen)
                    remaining -= len(chosen)
                # if still need deletions (e.g., very short sentences), fill globally
                if remaining > 0:
                    pool = [i for i in word_idxs if (i not in delete_set)]
                    extra = set(random.sample(pool, min(remaining, len(pool))))
                    delete_set.update(extra)

            # build attacked tokens
            attacked_tokens = [tok for i, tok in enumerate(tokens) if i not in delete_set]
            attacked_text = _reconstruct_text(attacked_tokens)

        example[f"{column}_attacked"] = attacked_text
        if tokenizer is not None:
            example[f"{column}_attacked_length"] = len(tokenizer(attacked_text)["input_ids"]) or 0
        else:
            example[f"{column}_attacked_length"] = len(attacked_text.split())

    # Handle sampled_positions fields for evaluation
    if "sampled_positions" in example:
        example["w_wm_output_sampled_positions"] = example.get("sampled_positions", "")
    example["w_wm_output_attacked_sampled_positions"] = ""

    return example


# -----------------------------------------
# Appendix E: Synonym Substitution (basic)
# -----------------------------------------

def _wordnet_synonyms(word: str) -> List[str]:
    """Get candidate synonyms for a word from WordNet (single-token, filtered)."""
    if wn is None:
        return []
    try:
        synsets = wn.synsets(word)
    except LookupError:
        # corpora not available locally
        return []
    cand = set()
    lw = word.lower()
    for s in synsets:
        for l in s.lemmas():
            name = l.name().replace("_", " ")
            if name.lower() != lw and " " not in name:  # single-token only
                cand.add(name)
    return list(cand)


def _preserve_case(src: str, dst: str) -> str:
    if src.isupper():
        return dst.upper()
    if src[0].isupper():
        return dst.capitalize()
    return dst


def basic_synonym_substitution_attack(example, tokenizer=None, sub_rate=0.2, args=None):
    """Replace a fraction of words with WordNet synonyms without context.

    Falls back gracefully if nltk/wordnet are unavailable (no-op for words
    without synonyms).
    """
    if isinstance(sub_rate, str) and sub_rate.endswith("%"):
        try:
            sub_rate = float(sub_rate[:-1]) / 100.0
        except Exception:
            sub_rate = 0.2
    try:
        sub_rate = float(sub_rate)
    except Exception:
        sub_rate = 0.2
    sub_rate = max(0.0, min(1.0, sub_rate))

    for column in ["w_wm_output", "no_wm_output"]:
        if column not in example or not isinstance(example[column], str):
            example[f"{column}_attacked"] = ""
            example[f"{column}_attacked_length"] = 0
            continue

        original_text = example[column]
        tokens = _tokenize_words_and_punct(original_text)
        word_idxs = _select_word_indices(tokens)
        if len(word_idxs) == 0 or sub_rate == 0:
            attacked_text = original_text
        else:
            num_to_sub = int(round(len(word_idxs) * sub_rate))
            num_to_sub = max(1, num_to_sub) if sub_rate > 0 else 0
            candidates = random.sample(word_idxs, min(num_to_sub, len(word_idxs)))
            replaced = 0
            for idx in candidates:
                w = tokens[idx]
                # skip stopwords and very short tokens to preserve structure
                if len(w) <= 2 or w.lower() in _FALLBACK_STOPWORDS:
                    continue
                syns = _wordnet_synonyms(w)
                if not syns:
                    continue
                repl = _preserve_case(w, random.choice(syns))
                tokens[idx] = repl
                replaced += 1
            attacked_text = _reconstruct_text(tokens)

        example[f"{column}_attacked"] = attacked_text
        if tokenizer is not None:
            example[f"{column}_attacked_length"] = len(tokenizer(attacked_text)["input_ids"]) or 0
        else:
            example[f"{column}_attacked_length"] = len(attacked_text.split())

    # Handle sampled_positions fields for evaluation
    if "sampled_positions" in example:
        example["w_wm_output_sampled_positions"] = example.get("sampled_positions", "")
    example["w_wm_output_attacked_sampled_positions"] = ""

    return example


# ---------------------------------------------------
# Appendix E: Synonym Substitution (context-aware)
# ---------------------------------------------------

def context_aware_synonym_substitution_attack(
    example, tokenizer=None, sub_rate=0.2, args=None
):
    """Replace a fraction of words with context-aware synonyms via Lesk WSD.

    Uses nltk.wsd.lesk to disambiguate word sense using the sentence-as-context.
    Falls back to basic synonyms if WSD fails. Requires WordNet corpora.
    """
    if isinstance(sub_rate, str) and sub_rate.endswith("%"):
        try:
            sub_rate = float(sub_rate[:-1]) / 100.0
        except Exception:
            sub_rate = 0.2
    try:
        sub_rate = float(sub_rate)
    except Exception:
        sub_rate = 0.2
    sub_rate = max(0.0, min(1.0, sub_rate))

    for column in ["w_wm_output", "no_wm_output"]:
        if column not in example or not isinstance(example[column], str):
            example[f"{column}_attacked"] = ""
            example[f"{column}_attacked_length"] = 0
            continue

        original_text = example[column]
        tokens = _tokenize_words_and_punct(original_text)
        word_idxs = _select_word_indices(tokens)
        if len(word_idxs) == 0 or sub_rate == 0 or wn is None:
            attacked_text = original_text
        else:
            num_to_sub = int(round(len(word_idxs) * sub_rate))
            num_to_sub = max(1, num_to_sub) if sub_rate > 0 else 0
            candidates = random.sample(word_idxs, min(num_to_sub, len(word_idxs)))

            # Prepare sentence spans for context
            sent_spans = _chunk_by_sentences(tokens)
            for idx in candidates:
                w = tokens[idx]
                if len(w) <= 2 or w.lower() in _FALLBACK_STOPWORDS:
                    continue
                # identify sentence containing idx
                containing = None
                for s_start, s_end in sent_spans:
                    if s_start <= idx <= s_end:
                        containing = (s_start, s_end)
                        break
                s_start, s_end = containing if containing else (0, len(tokens) - 1)
                # Build context as word tokens only within the sentence
                context_tokens = [t for t in tokens[s_start : s_end + 1] if _WORD_RE.fullmatch(t)]
                try:
                    synset = lesk(context_tokens, w) if lesk is not None else None
                except LookupError:
                    synset = None
                cand = []
                if synset is not None:
                    cand = [l.name().replace("_", " ") for l in synset.lemmas()]
                    cand = [c for c in cand if c.lower() != w.lower() and " " not in c]
                if not cand:
                    cand = _wordnet_synonyms(w)
                if not cand:
                    continue
                repl = _preserve_case(w, random.choice(cand))
                tokens[idx] = repl

            attacked_text = _reconstruct_text(tokens)

        example[f"{column}_attacked"] = attacked_text
        if tokenizer is not None:
            example[f"{column}_attacked_length"] = len(tokenizer(attacked_text)["input_ids"]) or 0
        else:
            example[f"{column}_attacked_length"] = len(attacked_text.split())

    # Handle sampled_positions fields for evaluation
    if "sampled_positions" in example:
        example["w_wm_output_sampled_positions"] = example.get("sampled_positions", "")
    example["w_wm_output_attacked_sampled_positions"] = ""

    return example


def check_output_column_lengths(example, min_len=0):
    baseline_completion_len = example["baseline_completion_length"]
    no_wm_output_len = example["no_wm_output_length"]
    w_wm_output_len = example["w_wm_output_length"]
    conds = all(
        [
            baseline_completion_len >= min_len,
            no_wm_output_len >= min_len,
            w_wm_output_len >= min_len,
        ]
    )
    return conds


def tokenize_for_copy_paste(example, tokenizer=None, args=None):
    for text_col in OUTPUT_TEXT_COLUMN_NAMES:
        if text_col in example:
            tokenized = tokenizer(
                example[text_col], return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
            # empty tensors are float type by default
            # this leads to an error when constructing pyarrow table
            if not str(tokenized.dtype) == "torch.int64":
                tokenized = tokenized.long()
            example[f"{text_col}_tokd"] = tokenized
    return example


def copy_paste_attack(example, tokenizer=None, args=None):
    # check if the example is long enough to attack
    if not check_output_column_lengths(example, min_len=args.cp_attack_min_len):
        # # if not, copy the orig w_wm_output to w_wm_output_attacked
        # NOTE changing this to return "" so that those fail/we can filter out these examples
        example["w_wm_output_attacked"] = ""
        example["w_wm_output_attacked_length"] = 0

        # Handle sampled_positions fields for evaluation
        if "sampled_positions" in example:
            example["w_wm_output_sampled_positions"] = example["sampled_positions"]
        example["w_wm_output_attacked_sampled_positions"] = ""

        return example

    # else, attack

    # Understanding the functionality:
    # we always write the result into the "w_wm_output_attacked" column
    # however depending on the detection method we're targeting, the
    # "src" and "dst" columns will be different. However,
    # the internal logic for these functions has old naming conventions of
    # watermarked always being the insertion src and no_watermark always being the dst

    tokenized_dst = example[f"{args.cp_attack_dst_col}_tokd"]
    tokenized_src = example[f"{args.cp_attack_src_col}_tokd"]
    min_token_count = min(len(tokenized_dst), len(tokenized_src))
    # input ids might have been converted to float if empty rows exist
    for key in example.keys():
        if "tokd" in key:
            example[key] = list(map(int, example[key]))

    if args.cp_attack_type == "single-single":  # 1-t
        tokenized_attacked_output = single_insertion(
            args.cp_attack_insertion_len,
            min_token_count,
            tokenized_dst,
            tokenized_src,
        )
    elif args.cp_attack_type == "triple-single":  # 3-t
        tokenized_attacked_output = triple_insertion_single_len(
            args.cp_attack_insertion_len,
            min_token_count,
            tokenized_dst,
            tokenized_src,
        )
    elif args.cp_attack_type == "k-t":
        tokenized_attacked_output = k_insertion_t_len(
            args.cp_attack_num_insertions,  # k
            args.cp_attack_insertion_len,  # t
            min_token_count,
            tokenized_dst,
            tokenized_src,
            verbose=args.verbose,
        )
    elif args.cp_attack_type == "k-random":  # k-t | k>=3, t in [floor(T/2k), T/k)
        raise NotImplementedError(f"Attack type {args.cp_attack_type} not implemented")
    elif args.cp_attack_type == "triple-triple":  # 3-(k_1,k_2,k_3)
        raise NotImplementedError(f"Attack type {args.cp_attack_type} not implemented")
    else:
        raise ValueError(f"Invalid attack type: {args.cp_attack_type}")

    # error occurred during attacking
    if tokenized_attacked_output is None:
        example["w_wm_output_attacked"] = ""
        example["w_wm_output_attacked_length"] = 0

        # Handle sampled_positions fields for evaluation
        if "sampled_positions" in example:
            example["w_wm_output_sampled_positions"] = example["sampled_positions"]
        example["w_wm_output_attacked_sampled_positions"] = ""

        return example

    tokenized_attacked_output = list(map(int, tokenized_attacked_output))

    example["w_wm_output_attacked"] = tokenizer.batch_decode(
        [tokenized_attacked_output], skip_special_tokens=True
    )[0]
    example["w_wm_output_attacked_length"] = len(tokenized_attacked_output)

    # Handle sampled_positions fields for evaluation
    # Copy original sampled_positions to w_wm_output_sampled_positions
    if "sampled_positions" in example:
        example["w_wm_output_sampled_positions"] = example["sampled_positions"]

    # For attacked output, set empty sampled_positions since it's not watermarked
    example["w_wm_output_attacked_sampled_positions"] = ""

    return example
