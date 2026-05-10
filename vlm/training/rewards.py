import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class RewardBreakdown:
    total: float
    json_valid: float
    schema: float
    line_items_populated: float
    content: float
    anti_hallucination: float
    total_match: float
    json_like: float


REQUIRED_ITEM_KEYS = {"name", "count", "price"}


def extract_json(text: str) -> str | None:
    """
    Extract a JSON object from model output.

    Prefer exact JSON. If the model wrapped JSON in extra text, try to extract
    the outermost object, but reward will later penalize extra text.
    """
    text = text.strip()

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]

    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return None


def compute_reward(generated: str, ground_truth: str) -> RewardBreakdown:
    generated = generated.strip()
    json_like_score = _json_like_score(generated)
    extracted = extract_json(generated)

    if extracted is None:
        total = max(0.0, min(0.15, json_like_score + _garbage_penalty(generated)))

        return RewardBreakdown(
            total=total,
            json_valid=0.0,
            schema=0.0,
            line_items_populated=0.0,
            content=0.0,
            anti_hallucination=_garbage_penalty(generated),
            total_match=0.0,
            json_like=json_like_score,
        )

    try:
        parsed = json.loads(extracted)
    except json.JSONDecodeError:
        total = max(0.0, min(0.15, json_like_score + _garbage_penalty(generated)))

        return RewardBreakdown(
            total=total,
            json_valid=0.0,
            schema=0.0,
            line_items_populated=0.0,
            content=0.0,
            anti_hallucination=_garbage_penalty(generated),
            total_match=0.0,
            json_like=json_like_score,
        )

    try:
        gt_parsed = json.loads(ground_truth)
    except Exception:
        gt_parsed = {}

    json_valid_score = 0.20
    schema_score = _schema_score(parsed)
    populated_score = _line_items_populated_score(parsed, gt_parsed)
    content_score = _content_similarity(parsed, gt_parsed)
    anti_hallucination_score = _anti_hallucination_score(parsed, gt_parsed)
    total_match_score = _total_match_score(parsed, gt_parsed)

    penalty = 0.0
    penalty += _extra_text_penalty(generated, extracted)
    penalty += _repetition_penalty(parsed)
    penalty += _empty_line_items_penalty(parsed, gt_parsed)
    penalty += _bad_number_penalty(parsed)
    penalty += _garbage_penalty(generated)

    total = (
        json_valid_score
        + schema_score
        + populated_score
        + content_score
        + anti_hallucination_score
        + total_match_score
        + penalty
    )

    total = max(0.0, min(1.0, total))

    return RewardBreakdown(
        total=total,
        json_valid=json_valid_score,
        schema=schema_score,
        line_items_populated=populated_score,
        content=content_score,
        anti_hallucination=anti_hallucination_score,
        total_match=total_match_score,
        json_like=json_like_score,
    )


def _schema_score(parsed: Any) -> float:
    if not isinstance(parsed, dict):
        return 0.0

    score = 0.0

    if "line_items" in parsed:
        score += 0.06

    if "total" in parsed:
        score += 0.06

    line_items = parsed.get("line_items")

    if isinstance(line_items, list):
        score += 0.04

        valid_item_count = 0

        for item in line_items:
            if not isinstance(item, dict):
                continue

            keys = set(item.keys())

            if "name" in keys:
                score += 0.01
            if "count" in keys:
                score += 0.01
            if "price" in keys:
                score += 0.01

            if REQUIRED_ITEM_KEYS.issubset(keys):
                valid_item_count += 1

        if line_items and valid_item_count == len(line_items):
            score += 0.03

    return min(0.20, score)


def _line_items_populated_score(parsed: dict, ground_truth: dict) -> float:
    pred_items = parsed.get("line_items", [])
    gt_items = ground_truth.get("line_items", [])

    if not isinstance(pred_items, list):
        return 0.0

    if not isinstance(gt_items, list):
        gt_items = []

    if len(gt_items) > 0 and len(pred_items) == 0:
        return 0.0

    if len(pred_items) == 0:
        return 0.0

    score = 0.08

    if len(gt_items) > 0:
        ratio = min(len(pred_items), len(gt_items)) / max(len(pred_items), len(gt_items))
        score += 0.07 * ratio
    else:
        score += 0.03

    return min(0.15, score)


def _content_similarity(predicted: dict, ground_truth: dict) -> float:
    pred_tokens = _content_tokens(predicted)
    gt_tokens = _content_tokens(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)

    true_positives = len(pred_set & gt_set)

    precision = true_positives / max(1, len(pred_set))
    recall = true_positives / max(1, len(gt_set))

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    score = 0.25 * f1

    if precision < 0.15:
        score -= 0.05

    return max(-0.05, min(0.25, score))


def _anti_hallucination_score(predicted: dict, ground_truth: dict) -> float:
    pred_tokens = _content_tokens(predicted)
    gt_tokens = _content_tokens(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)

    extra_ratio = len(pred_set - gt_set) / max(1, len(pred_set))

    if len(pred_tokens) > 3 * max(1, len(gt_tokens)):
        return -0.10

    if extra_ratio > 0.80:
        return -0.08

    if extra_ratio > 0.60:
        return -0.04

    return 0.10


def _total_match_score(predicted: dict, ground_truth: dict) -> float:
    pred_total = _normalize_numberish(predicted.get("total", ""))
    gt_total = _normalize_numberish(ground_truth.get("total", ""))

    if not pred_total or not gt_total:
        return 0.0

    if pred_total == gt_total:
        return 0.10

    if pred_total in gt_total or gt_total in pred_total:
        return 0.04

    return 0.0


def _json_like_score(text: str) -> float:
    text_l = text.lower().strip()
    score = 0.0

    if "{" in text_l:
        score += 0.03
    if "}" in text_l:
        score += 0.03
    if "line_items" in text_l:
        score += 0.04
    if "total" in text_l:
        score += 0.04
    if "price" in text_l:
        score += 0.02
    if "name" in text_l:
        score += 0.02
    if text_l.startswith("{"):
        score += 0.02

    return min(score, 0.15)


def _extra_text_penalty(original: str, extracted: str) -> float:
    original = original.strip()
    extracted = extracted.strip()

    if original == extracted:
        return 0.0

    extra_len = len(original) - len(extracted)

    if extra_len <= 0:
        return 0.0

    if extra_len > 100:
        return -0.08

    return -0.04


def _empty_line_items_penalty(predicted: dict, ground_truth: dict) -> float:
    pred_items = predicted.get("line_items", [])
    gt_items = ground_truth.get("line_items", [])

    if not isinstance(pred_items, list):
        return 0.0

    if isinstance(gt_items, list) and len(gt_items) > 0 and len(pred_items) == 0:
        return -0.15

    return 0.0


def _repetition_penalty(predicted: dict) -> float:
    items = predicted.get("line_items", [])

    if not isinstance(items, list) or len(items) <= 1:
        return 0.0

    signatures = []

    for item in items:
        if not isinstance(item, dict):
            continue

        name = _norm_text(item.get("name", ""))
        price = _normalize_numberish(item.get("price", ""))
        count = _norm_text(item.get("count", ""))

        signatures.append((name, count, price))

    if len(signatures) <= 1:
        return 0.0

    unique = set(signatures)
    duplicate_ratio = 1.0 - (len(unique) / max(1, len(signatures)))

    if duplicate_ratio > 0.75:
        return -0.12

    if duplicate_ratio > 0.50:
        return -0.08

    if duplicate_ratio > 0.25:
        return -0.04

    return 0.0


def _bad_number_penalty(predicted: dict) -> float:
    penalty = 0.0

    total = str(predicted.get("total", ""))

    if len(re.sub(r"\D", "", total)) > 10:
        penalty -= 0.05

    if re.search(r"\d+[.,]\d{3}[.,]\d{3}", total):
        penalty -= 0.03

    items = predicted.get("line_items", [])

    if isinstance(items, list):
        for item in items[:10]:
            if not isinstance(item, dict):
                continue

            price = str(item.get("price", ""))

            if len(re.sub(r"\D", "", price)) > 10:
                penalty -= 0.02

    return max(-0.10, penalty)


def _garbage_penalty(text: str) -> float:
    text = text.strip()

    if len(text) > 800:
        return -0.08

    if "<|user|>" in text or "<|assistant|>" in text:
        return -0.10

    if _repeated_character_ratio(text) > 0.50:
        return -0.08

    return 0.0


def _repeated_character_ratio(text: str) -> float:
    if not text:
        return 0.0

    repeated = 0

    for a, b in zip(text, text[1:]):
        if a == b:
            repeated += 1

    return repeated / max(1, len(text) - 1)


def _normalize_numberish(value) -> str:
    text = str(value).lower().strip()
    return re.sub(r"[^0-9]", "", text)


def _norm_text(value) -> str:
    return re.sub(r"\s+", " ", str(value).lower().strip())


def _content_tokens(d: dict) -> list[str]:
    """
    Tokenize only values, not JSON field names.

    This avoids giving content reward just because the output contains keys like
    line_items, name, count, price, total.
    """
    values = []

    def collect(x):
        if isinstance(x, dict):
            for v in x.values():
                collect(v)
        elif isinstance(x, list):
            for v in x:
                collect(v)
        elif x is not None:
            values.append(str(x))

    collect(d)

    text = " ".join(values).lower()
    return re.findall(r"\b\w+\b", text)