import json
import re
from dataclasses import dataclass


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


def extract_json(text: str) -> str | None:
    text = text.strip()

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return None


def compute_reward(generated: str, ground_truth: str) -> RewardBreakdown:
    json_like_score = _json_like_score(generated)
    extracted = extract_json(generated)

    if extracted is None:
        total = max(0.0, min(0.25, json_like_score))
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

    parsed = json.loads(extracted)

    json_valid_score = 0.30

    has_line_items = "line_items" in parsed
    has_total = "total" in parsed

    if has_line_items and has_total:
        schema_score = 0.20
    elif has_line_items or has_total:
        schema_score = 0.10
    else:
        schema_score = 0.0

    populated_score = 0.0
    if has_line_items and isinstance(parsed["line_items"], list) and len(parsed["line_items"]) > 0:
        populated_score = 0.15

    content_score = 0.0
    anti_hallucination_score = 0.0
    total_match_score = 0.0

    try:
        gt_parsed = json.loads(ground_truth)
        content_score = _content_similarity(parsed, gt_parsed)
        anti_hallucination_score = _anti_hallucination_score(parsed, gt_parsed)
        total_match_score = _total_match_score(parsed, gt_parsed)
    except Exception:
        pass

    total = (
        json_valid_score
        + schema_score
        + populated_score
        + content_score
        + anti_hallucination_score
        + total_match_score
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


def _json_like_score(text: str) -> float:
    text_l = text.lower().strip()
    score = 0.0

    if "{" in text_l:
        score += 0.04
    if "}" in text_l:
        score += 0.04
    if "line_items" in text_l:
        score += 0.06
    if "total" in text_l:
        score += 0.06
    if "price" in text_l:
        score += 0.03
    if "name" in text_l:
        score += 0.03
    if text_l.startswith("{"):
        score += 0.04

    return min(score, 0.25)


def _garbage_penalty(text: str) -> float:
    if len(text) > 800:
        return -0.05
    return 0.0


def _content_similarity(predicted: dict, ground_truth: dict) -> float:
    pred_tokens = _tokenize_dict(predicted)
    gt_tokens = _tokenize_dict(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)

    true_positives = len(pred_set & gt_set)
    precision = true_positives / max(1, len(pred_set))
    recall = true_positives / max(1, len(gt_set))

    if precision + recall == 0:
        return 0.0

    if precision < 0.25:
        return -0.05

    f1 = 2 * precision * recall / (precision + recall)
    return 0.20 * f1


def _anti_hallucination_score(predicted: dict, ground_truth: dict) -> float:
    pred_tokens = _tokenize_dict(predicted)
    gt_tokens = _tokenize_dict(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    if len(pred_tokens) > 3 * len(gt_tokens):
        return -0.10

    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)

    extra_ratio = len(pred_set - gt_set) / max(1, len(pred_set))

    if extra_ratio > 0.75:
        return -0.05

    return 0.05


def _total_match_score(predicted: dict, ground_truth: dict) -> float:
    pred_total = _normalize_numberish(predicted.get("total", ""))
    gt_total = _normalize_numberish(ground_truth.get("total", ""))

    if not pred_total or not gt_total:
        return 0.0

    if pred_total == gt_total:
        return 0.05

    return 0.0


def _normalize_numberish(value) -> str:
    text = str(value).lower().strip()
    return re.sub(r"[^0-9.]", "", text)


def _tokenize_dict(d: dict) -> list[str]:
    text = json.dumps(d, ensure_ascii=False).lower()
    return re.findall(r"\b\w+\b", text)