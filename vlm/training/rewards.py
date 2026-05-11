"""
Reward function for RL alignment of the receipt VLM.

Aligned to the assignment spec:
    "Reward formatting compliance and schema adherence;
     penalize hallucinations, unformatted text blocks, or malformed JSON."

Components:
    format        up to 0.30   valid JSON, preferably no surrounding text
    schema        up to 0.30   required keys present and correctly typed
    content       up to 0.30   line item count + total accuracy + name overlap
    hallucination negative     duplicate items, garbage text, excessive output

The final reward is clipped to [0, 1].
"""

import re
from dataclasses import dataclass
from itertools import pairwise

from vlm.utils.json_extractor import parse_json_object


@dataclass
class RewardBreakdown:
    total: float
    format: float
    schema: float
    content: float
    hallucination: float


REQUIRED_TOP_KEYS = {"line_items", "total"}

# `count` is intentionally optional: the dataset only emits it when the
# receipt actually shows a quantity, so requiring it would punish the
# model for faithfully matching the image.
REQUIRED_ITEM_KEYS = {"name", "price"}


def compute_reward(generated: str, ground_truth: str) -> RewardBreakdown:
    """Score a single generated completion against its ground truth."""
    generated = generated.strip()

    parsed = parse_json_object(generated)
    gt = parse_json_object(ground_truth)
    if not isinstance(gt, dict):
        gt = {}

    format_score = _format_score(generated, parsed)

    # parsed is None (unparseable) or not a dict (e.g. top-level JSON list).
    # Neither can satisfy our receipt schema, so format credit only.
    if not isinstance(parsed, dict):
        return RewardBreakdown(
            total=max(0.0, min(1.0, format_score)),
            format=format_score,
            schema=0.0,
            content=0.0,
            hallucination=0.0,
        )

    schema_score = _schema_score(parsed)
    content_score = _content_score(parsed, gt)
    hallucination_score = _hallucination_penalty(generated, parsed, gt)

    total = format_score + schema_score + content_score + hallucination_score

    return RewardBreakdown(
        total=max(0.0, min(1.0, total)),
        format=format_score,
        schema=schema_score,
        content=content_score,
        hallucination=hallucination_score,
    )


def _format_score(text: str, parsed) -> float:
    """Reward strict JSON.

    The assignment defines format adherence as "successfully parse as valid
    JSON", which means json.loads on the entire output must succeed. We give
    full credit only to strict JSON and a small consolation signal to
    parseable-but-wrapped JSON so RL still sees a gradient toward "remove
    the prose wrapper". Everything else gets zero.
    """
    if parsed is None:
        return 0.0

    # Strict JSON: the full generated string is the JSON object.
    if text.startswith("{") and text.endswith("}") and parse_json_object(text) is not None:
        return 0.30

    # JSON exists but is wrapped in prose. Small partial credit only.
    return 0.05


def _schema_score(parsed) -> float:
    """Reward required receipt schema."""
    if not isinstance(parsed, dict):
        return 0.0

    score = 0.0

    if "line_items" in parsed:
        score += 0.05

    if "total" in parsed:
        score += 0.05

    items = parsed.get("line_items")
    if not isinstance(items, list):
        return score

    score += 0.05

    if not items:
        return score

    well_formed = sum(
        1
        for item in items
        if isinstance(item, dict) and REQUIRED_ITEM_KEYS.issubset(item.keys())
    )

    score += 0.15 * (well_formed / len(items))

    return min(0.30, score)


def _content_score(parsed: dict, gt: dict) -> float:
    """Reward item count similarity, total match, and item-name overlap."""
    score = 0.0

    pred_items = parsed.get("line_items") or []
    gt_items = gt.get("line_items") or []

    if isinstance(pred_items, list) and isinstance(gt_items, list) and gt_items:
        ratio = min(len(pred_items), len(gt_items)) / max(len(pred_items), len(gt_items))
        score += 0.10 * ratio

    pred_total = _digits_only(parsed.get("total", ""))
    gt_total = _digits_only(gt.get("total", ""))

    if pred_total and gt_total:
        if pred_total == gt_total:
            score += 0.10
        elif pred_total in gt_total or gt_total in pred_total:
            score += 0.04

    pred_names = _name_tokens(pred_items)
    gt_names = _name_tokens(gt_items)

    if pred_names and gt_names:
        overlap = len(pred_names & gt_names) / len(gt_names)
        score += 0.10 * min(1.0, overlap)

    return min(0.30, score)


def _hallucination_penalty(text: str, parsed: dict, gt: dict) -> float:
    """Negative score for duplicate, excessive, or garbage-like outputs."""
    penalty = 0.0

    items = parsed.get("line_items") if isinstance(parsed, dict) else None

    if isinstance(items, list) and len(items) > 1:
        signatures = [
            (_norm(item.get("name", "")), _digits_only(item.get("price", "")))
            for item in items
            if isinstance(item, dict)
        ]

        if signatures:
            duplicate_ratio = 1.0 - (len(set(signatures)) / len(signatures))

            if duplicate_ratio > 0.50:
                penalty -= 0.10
            elif duplicate_ratio > 0.25:
                penalty -= 0.05

    gt_items = gt.get("line_items")
    if isinstance(items, list) and isinstance(gt_items, list) and gt_items and len(items) > 3 * len(gt_items):
            penalty -= 0.05

    if "<|user|>" in text or "<|assistant|>" in text:
        penalty -= 0.10

    if _repeated_char_ratio(text) > 0.50:
        penalty -= 0.05

    return penalty


def _digits_only(value) -> str:
    return re.sub(r"\D", "", str(value))


def _norm(value) -> str:
    return re.sub(r"\s+", " ", str(value).lower().strip())


def _name_tokens(items) -> set[str]:
    """Set of word tokens from all predicted or ground-truth item names."""
    tokens: set[str] = set()

    if not isinstance(items, list):
        return tokens

    for item in items:
        if not isinstance(item, dict):
            continue

        name = _norm(item.get("name", ""))
        tokens.update(re.findall(r"\b\w+\b", name))

    return tokens



def _repeated_char_ratio(text: str) -> float:
    if len(text) < 2:
        return 0.0

    repeated = sum(1 for a, b in pairwise(text) if a == b)
    return repeated / (len(text) - 1)