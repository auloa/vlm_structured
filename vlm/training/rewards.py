"""
Reward function for RL alignment of the receipt VLM.

Aligned to the assignment spec:
    "Reward formatting compliance and schema adherence;
     penalize hallucinations, unformatted text blocks, or malformed JSON."

Components (all in [0, 1] after clipping):

    format        0.30   valid JSON, no surrounding text
    schema        0.30   required keys present and correctly typed
    content       0.30   line item count + total accuracy + name overlap
    hallucination negative penalty for extra/garbage tokens

Assumes the dataset has been filtered to only contain samples with
non-empty line_items (see CORDDataset filtering).
"""

import json
import re
from dataclasses import dataclass


@dataclass
class RewardBreakdown:
    total: float
    format: float
    schema: float
    content: float
    hallucination: float


REQUIRED_TOP_KEYS = {"line_items", "total"}
REQUIRED_ITEM_KEYS = {"name", "count", "price"}


# ── public API ────────────────────────────────────────────────────────────────

def compute_reward(generated: str, ground_truth: str) -> RewardBreakdown:
    """Score a single generated completion against its ground truth."""
    generated = generated.strip()

    # Try to parse the prediction.
    parsed = _try_parse(generated)
    gt = _try_parse(ground_truth) or {}

    # ── Format: did we produce clean, parseable JSON? ────────────────────────
    format_score = _format_score(generated, parsed)

    if parsed is None:
        total = format_score  # only format partial credit available
        return RewardBreakdown(
            total=max(0.0, min(1.0, total)),
            format=format_score,
            schema=0.0,
            content=0.0,
            hallucination=0.0,
        )

    # ── Schema: required keys present with right types ───────────────────────
    schema_score = _schema_score(parsed)

    # ── Content: count/total/name accuracy vs GT ─────────────────────────────
    content_score = _content_score(parsed, gt)

    # ── Hallucination penalty (negative) ─────────────────────────────────────
    hallucination_score = _hallucination_penalty(generated, parsed, gt)

    total = format_score + schema_score + content_score + hallucination_score

    return RewardBreakdown(
        total=max(0.0, min(1.0, total)),
        format=format_score,
        schema=schema_score,
        content=content_score,
        hallucination=hallucination_score,
    )


# ── format ────────────────────────────────────────────────────────────────────

def _format_score(text: str, parsed) -> float:
    """0.30 if clean JSON, less if wrapped, 0 if unparseable."""
    if parsed is None:
        # Partial credit for JSON-shaped but invalid output.
        looks_jsonish = "{" in text and "}" in text
        return 0.10 if looks_jsonish else 0.0

    # Clean JSON: text is exactly the JSON object.
    try:
        if json.dumps(json.loads(text)) and text.startswith("{") and text.endswith("}"):
            return 0.30
    except json.JSONDecodeError:
        pass

    # Parseable but wrapped in extra text.
    return 0.20


# ── schema ────────────────────────────────────────────────────────────────────

def _schema_score(parsed) -> float:
    """0.30 if full schema, scaled down for missing/malformed pieces."""
    if not isinstance(parsed, dict):
        return 0.0

    score = 0.0

    # Top-level keys (0.10 total)
    if "line_items" in parsed:
        score += 0.05
    if "total" in parsed:
        score += 0.05

    # line_items is a list (0.05)
    items = parsed.get("line_items")
    if not isinstance(items, list):
        return score

    score += 0.05

    # All items are well-formed dicts with required keys (0.15)
    if not items:
        return score

    well_formed = sum(
        1 for item in items
        if isinstance(item, dict) and REQUIRED_ITEM_KEYS.issubset(item.keys())
    )
    score += 0.15 * (well_formed / len(items))

    return score


# ── content ───────────────────────────────────────────────────────────────────

def _content_score(parsed: dict, gt: dict) -> float:
    """0.30 split across item count match, total match, and name overlap."""
    score = 0.0

    pred_items = parsed.get("line_items") or []
    gt_items = gt.get("line_items") or []

    # Item count similarity (0.10)
    if isinstance(pred_items, list) and gt_items:
        ratio = min(len(pred_items), len(gt_items)) / max(len(pred_items), len(gt_items))
        score += 0.10 * ratio

    # Total match (0.10)
    pred_total = _digits_only(parsed.get("total", ""))
    gt_total = _digits_only(gt.get("total", ""))
    if pred_total and gt_total:
        if pred_total == gt_total:
            score += 0.10
        elif pred_total in gt_total or gt_total in pred_total:
            score += 0.04

    # Item name overlap (0.10)
    pred_names = _name_tokens(pred_items)
    gt_names = _name_tokens(gt_items)
    if pred_names and gt_names:
        overlap = len(pred_names & gt_names) / len(gt_names)
        score += 0.10 * min(1.0, overlap)

    return score


# ── hallucination penalty ─────────────────────────────────────────────────────

def _hallucination_penalty(text: str, parsed: dict, gt: dict) -> float:
    """Negative score for clear hallucination signals."""
    penalty = 0.0

    # Duplicate line items
    items = parsed.get("line_items") if isinstance(parsed, dict) else None
    if isinstance(items, list) and len(items) > 1:
        signatures = [
            (_norm(i.get("name", "")), _digits_only(i.get("price", "")))
            for i in items if isinstance(i, dict)
        ]
        if signatures:
            dup_ratio = 1 - (len(set(signatures)) / len(signatures))
            if dup_ratio > 0.5:
                penalty -= 0.10
            elif dup_ratio > 0.25:
                penalty -= 0.05

    # Way too many items vs GT
    if isinstance(items, list) and gt.get("line_items"):
        if len(items) > 3 * len(gt["line_items"]):
            penalty -= 0.05

    # Junk in the raw text
    if "<|user|>" in text or "<|assistant|>" in text:
        penalty -= 0.10
    if _repeated_char_ratio(text) > 0.5:
        penalty -= 0.05

    return penalty


# ── helpers ───────────────────────────────────────────────────────────────────

def _try_parse(text: str):
    """Try to extract and parse JSON from text. Returns dict or None."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try outermost {...}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _digits_only(value) -> str:
    return re.sub(r"\D", "", str(value))


def _norm(value) -> str:
    return re.sub(r"\s+", " ", str(value).lower().strip())


def _name_tokens(items) -> set:
    """Set of word tokens from all item names."""
    tokens = set()
    if not isinstance(items, list):
        return tokens
    for item in items:
        if isinstance(item, dict):
            name = _norm(item.get("name", ""))
            tokens.update(re.findall(r"\b\w+\b", name))
    return tokens


def _repeated_char_ratio(text: str) -> float:
    if len(text) < 2:
        return 0.0
    repeated = sum(1 for a, b in zip(text, text[1:]) if a == b)
    return repeated / (len(text) - 1)