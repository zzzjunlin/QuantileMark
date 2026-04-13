# coding=utf-8
import json
import hashlib
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


_OVERALL_SCORE_RE = re.compile(r"\"overall_score_1to5\"\\s*:\\s*([0-5])")
_SCORE_RE = re.compile(r"\"score\"\\s*:\\s*([0-5])")
_OVERALL_RE = re.compile(r"\"overall\"\\s*:\\s*([0-5])")


# Keep system prompt empty by default; put all instructions in the user prompt to
# avoid duplication and make prompting easier to audit.
DEFAULT_SYSTEM_PROMPT = ""


def _coerce_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        # Defensive: join list-of-strings cases.
        return " ".join("" if x is None else str(x) for x in v)
    return str(v)


def make_cache_key(model: str, system_prompt: str, label: str, text: str) -> str:
    """
    Stable key for caching a single-text evaluation.
    Include model + system_prompt to avoid cross-config cache poisoning.
    """
    h = hashlib.sha256()
    h.update(str(model).encode("utf-8"))
    h.update(b"\n")
    h.update(str(system_prompt).encode("utf-8"))
    h.update(b"\n")
    h.update(str(label).encode("utf-8"))
    h.update(b"\n")
    h.update(_coerce_text(text).encode("utf-8"))
    return h.hexdigest()


def make_text_cache_key(model: str, system_prompt: str, text: str) -> str:
    """
    Cache key for a single-text evaluation that intentionally does NOT include
    any label like "no_wm_output"/"w_wm_output" to avoid label-dependent caching.
    """
    return make_cache_key(model=model, system_prompt=system_prompt, label="", text=text)


class JSONLKeyValueCache:
    """
    Append-only JSONL cache: each line is {"key": ..., "value": ...}.
    On load, the last occurrence of a key wins.
    """

    def __init__(self, path: str, repair: bool = True):
        self.path = path
        self.repair = bool(repair)
        self._data: Dict[str, Any] = {}
        self.total_lines: int = 0
        self.bad_lines: int = 0
        self.loaded_records: int = 0
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        self.total_lines += 1
                        line = line.strip()
                        if not line:
                            self.bad_lines += 1
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            self.bad_lines += 1
                            continue
                        if isinstance(obj, dict) and "key" in obj and "value" in obj:
                            self._data[str(obj["key"])] = obj["value"]
                            self.loaded_records += 1
                        else:
                            self.bad_lines += 1
            except Exception:
                # Best-effort cache; never fail the evaluation due to cache I/O.
                pass
            # Repair a corrupted cache file (e.g., trailing truncated lines from
            # interrupted runs) by rewriting a clean JSONL file. We preserve a
            # backup for forensics but do not fail the run on any I/O error.
            if self.repair and self.bad_lines > 0:
                self._repair_on_load()

    def get(self, key: str):
        return self._data.get(key)

    def _repair_on_load(self) -> None:
        if not self.path:
            return
        try:
            if not os.path.exists(self.path):
                return
        except Exception:
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = f"{self.path}.corrupt.{ts}"
        tmp = f"{self.path}.tmp.{ts}"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                for k, v in self._data.items():
                    f.write(json.dumps({"key": k, "value": v}, ensure_ascii=False))
                    f.write("\n")
                    # Flush on every line to reduce the chance of truncated tmp
                    # files on unexpected termination.
                    f.flush()
            # Preserve the original file for inspection.
            try:
                os.replace(self.path, backup)
            except Exception:
                # If backup fails, continue and overwrite in place.
                backup = ""
            os.replace(tmp, self.path)
        except Exception:
            # Best-effort only.
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    def put(self, key: str, value: Any):
        self._data[key] = value
        if not self.path:
            return
        try:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
        except Exception:
            pass
        try:
            # Line-buffered append to reduce loss on abrupt termination.
            with open(self.path, "a", encoding="utf-8", buffering=1) as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False))
                f.write("\n")
                # Force the line through Python and OS buffers as best-effort.
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    pass
        except Exception:
            pass


def _try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    text = s.strip()
    # Tolerate accidental trailing text after a JSON object by using raw_decode.
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
        return obj
    except Exception:
        pass
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_score_fallback(s: str) -> Optional[int]:
    text = s or ""
    m = _OVERALL_RE.search(text)
    if not m:
        m = _OVERALL_SCORE_RE.search(text)
    if not m:
        m = _SCORE_RE.search(text)
    if not m:
        return None
    try:
        val = int(m.group(1))
    except Exception:
        return None
    if 0 <= val <= 5:
        return int(val)
    return None


def _parse_score(payload_text: str, key: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Returns (score, reason). Both may be None if parsing fails.
    """
    obj = _try_parse_json(payload_text)
    if isinstance(obj, dict) and key in obj:
        val = obj.get(key)
        if isinstance(val, dict):
            # New short schema: {"coherence": x, "clarity": y, "naturalness": z, "overall": avg}
            if isinstance(val.get("overall", None), (int, float)):
                overall = int(val["overall"])
                if 0 <= overall <= 5:
                    return overall, None

            # Legacy schema: {"overall_score_1to5": 4, ...} or {"score": 4, ...}
            score = val.get("overall_score_1to5", None)
            if score is None:
                score = val.get("score")
            if isinstance(score, (int, float)) and 0 <= float(score) <= 5:
                return int(score), None
        if isinstance(val, (int, float)) and 0 <= float(val) <= 5:
            return int(val), None

    # Fallback: sometimes models return {"score": 4} without our key
    if isinstance(obj, dict):
        if isinstance(obj.get("overall", None), (int, float)):
            overall = int(obj["overall"])
            if 0 <= overall <= 5:
                return overall, None

        if "overall_score_1to5" in obj or "score" in obj:
            score = obj.get("overall_score_1to5", None)
            if score is None:
                score = obj.get("score")
            if isinstance(score, (int, float)) and 0 <= float(score) <= 5:
                return int(score), None

        # Do not compute overall from other fields (no arithmetic fallback).

    # Last resort: regex for a single digit 1-5
    fallback = _extract_score_fallback(payload_text)
    if fallback is not None:
        return fallback, None
    return None, None


def _extract_eval_obj(payload_text: str, key: str) -> Optional[Dict[str, Any]]:
    obj = _try_parse_json(payload_text)
    if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
        return obj[key]
    # Allow single-object responses (no outer label keys) in some debug uses.
    if isinstance(obj, dict) and "dimension_scores" in obj:
        return obj
    # Allow the short schema as a single-object response.
    if isinstance(obj, dict) and all(k in obj for k in ["coherence", "clarity", "naturalness", "overall"]):
        return obj
    return None


def _openai_client(api_key_env: str, base_url: Optional[str] = None):
    from openai import OpenAI

    api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"Missing API key env var {api_key_env!r}. "
            f"Set it before running api-judge-5."
        )
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


@dataclass
class APIJudge5Config:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    temperature: float = 0.0
    max_tokens: int = 128
    timeout_s: float = 60.0
    max_retries: int = 5
    retry_backoff_s: float = 1.0
    store_reason: bool = False
    force_json: bool = True


class APIJudge5:
    """
    Minimal OpenAI-chat-based judge that outputs 1–5 writing quality scores.
    Intended for offline evaluation (slow, potentially expensive).
    """

    def __init__(self, cfg: APIJudge5Config):
        self.cfg = cfg
        self._client = _openai_client(cfg.api_key_env, cfg.base_url)
        self._cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def _chat(self, messages):
        last_err = None
        force_json = bool(self.cfg.force_json)
        for attempt in range(self.cfg.max_retries + 1):
            try:
                # Prefer strict JSON mode when supported by the model.
                kwargs = {}
                if force_json:
                    kwargs["response_format"] = {"type": "json_object"}
                return self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    timeout=self.cfg.timeout_s,
                    **kwargs,
                )
            except Exception as e:
                last_err = e
                # Retry without JSON mode if the server/model rejects it.
                if force_json:
                    force_json = False
                    continue
                if attempt >= self.cfg.max_retries:
                    break
                sleep_s = self.cfg.retry_backoff_s * (2**attempt)
                time.sleep(sleep_s)
        raise RuntimeError(f"api-judge-5 request failed: {last_err!r}")

    def score_pair(
        self,
        text_a_label: str,
        text_a: str,
        text_b_label: str,
        text_b: str,
    ) -> Dict[str, Any]:
        text_a = _coerce_text(text_a)
        text_b = _coerce_text(text_b)

        cache_key = (self.cfg.model, text_a, text_b)
        if cache_key in self._cache:
            return self._cache[cache_key]

        user_prompt = (
            "You are a strict and consistent text-quality evaluator.\n"
            "Use ONLY the given text; do NOT assume the author's intent.\n"
            "The text may start or end abruptly because the generation length is fixed. "
            "Do NOT penalize truncation or incompleteness.\n"
            "Do NOT browse the web or use external knowledge.\n"
            "Do NOT judge factual correctness.\n"
            "\n"
            "Rate each field as an integer from 0 to 5.\n"
            "overall is an independent judgment; do NOT compute overall from the other fields (no arithmetic).\n"
            "\n"
            "Rate the following two texts independently.\n"
            "Return only a JSON object in exactly the following structure:\n"
            "{\n"
            f'  "{text_a_label}": {{\n'
            '    "coherence": int,\n'
            '    "clarity": int,\n'
            '    "naturalness": int,\n'
            '    "overall": int\n'
            "  },\n"
            f'  "{text_b_label}": {{\n'
            '    "coherence": int,\n'
            '    "clarity": int,\n'
            '    "naturalness": int,\n'
            '    "overall": int\n'
            "  }\n"
            "}\n"
            "\n"
            "Texts:\n"
            f"{text_a_label}:\n{text_a}\n\n"
            f"{text_b_label}:\n{text_b}\n"
        )

        messages = [{"role": "user", "content": user_prompt}]
        if self.cfg.system_prompt:
            messages.insert(0, {"role": "system", "content": self.cfg.system_prompt})

        resp = self._chat(messages)
        content = resp.choices[0].message.content or ""

        a_obj = _extract_eval_obj(content, text_a_label)
        b_obj = _extract_eval_obj(content, text_b_label)

        a_score, a_reason = _parse_score(content, text_a_label)
        b_score, b_reason = _parse_score(content, text_b_label)

        result: Dict[str, Any] = {
            "raw": content,
            f"{text_a_label}_eval": a_obj,
            f"{text_b_label}_eval": b_obj,
            f"{text_a_label}_score": a_score,
            f"{text_b_label}_score": b_score,
        }
        if self.cfg.store_reason:
            result[f"{text_a_label}_reason"] = a_reason
            result[f"{text_b_label}_reason"] = b_reason

        self._cache[cache_key] = result
        return result

    def score_text(self, text: str) -> Dict[str, Any]:
        """
        Score a single text WITHOUT passing any label to the model.
        Returns: {"raw": str, "eval": dict|None, "score": int|None}
        """
        text = _coerce_text(text)
        cache_key = (self.cfg.model, self.cfg.system_prompt, text)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return {"raw": cached.get("raw"), "eval": cached.get("eval"), "score": cached.get("score")}

        user_prompt = (
            "You are a strict and consistent text-quality evaluator.\n"
            "Use ONLY the given text; do NOT assume the author's intent.\n"
            "The text may start or end abruptly because the generation length is fixed. "
            "Do NOT penalize truncation or incompleteness.\n"
            "Do NOT judge factual correctness.\n"
            "\n"
            "Rate each field as an integer from 1 to 5.\n"
            "overall is an independent judgment; do NOT compute overall from the other fields (no arithmetic).\n"
            "\n"
            "Rate the following text.\n"
            "Return only a JSON object in exactly the following structure:\n"
            "{\n"
            '  "coherence": int,\n'
            '  "clarity": int,\n'
            '  "naturalness": int,\n'
            '  "overall": int\n'
            "}\n"
            "\n"
            f"Text:\n{text}\n"
        )
        messages = [{"role": "user", "content": user_prompt}]
        if self.cfg.system_prompt:
            messages.insert(0, {"role": "system", "content": self.cfg.system_prompt})
        resp = self._chat(messages)
        content = resp.choices[0].message.content or ""

        eval_obj = _extract_eval_obj(content, "")
        score, _ = _parse_score(content, "")

        result: Dict[str, Any] = {"raw": content, "eval": eval_obj, "score": score}
        self._cache[cache_key] = result

        return {"raw": content, "eval": eval_obj, "score": score}

    # Backwards-compatible wrapper (kept for any external callers).
    def score_single(self, text_label: str, text: str) -> Dict[str, Any]:
        res = self.score_text(text)
        out: Dict[str, Any] = {f"{text_label}_eval": res.get("eval"), f"{text_label}_score": res.get("score"), "raw": res.get("raw")}
        return out
