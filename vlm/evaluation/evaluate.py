import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vlm.data.dataset import CORDDataset
from vlm.models.receipt_vlm import ReceiptVLM
from vlm.training.generate import generate_k_outputs
from vlm.training.rewards import compute_reward, extract_json
from vlm.utils.device import get_device
from vlm.utils.training import assert_only_projector_trainable, set_seed


@dataclass
class EvalMetrics:
    num_samples: int
    valid_json_rate: float
    schema_rate: float
    format_adherence_rate: float
    non_empty_line_items_rate: float
    empty_line_items_rate: float
    numeric_name_rate: float
    duplicate_item_rate: float
    extra_text_rate: float
    average_reward: float
    average_generation_length: float


def evaluate_checkpoint(
    dataset_name: str,
    split: str,
    num_samples: int,
    vision_model_name: str,
    image_height: int,
    image_width: int,
    lm_name: str,
    instruction: str,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    max_completion_tokens: int = 192,
    temperature: float = 0.1,
    do_sample: bool = False,
    top_p: float = 0.95,
    generation_repetition_penalty: float = 1.0,
    seed: int = 42,
) -> EvalMetrics:
    """
    Evaluate a projector checkpoint on held-out receipt images.

    This can be used for:
        - SFT checkpoint evaluation
        - RL checkpoint evaluation

    It measures the assignment-critical metric:
        format_adherence_rate =
            valid JSON AND contains required top-level keys:
            line_items and total.
    """

    set_seed(seed)

    device = get_device()
    print(f"device: {device}")

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("loading model...")
    model = ReceiptVLM(
        device=device,
        vision_model_name=vision_model_name,
        image_height=image_height,
        image_width=image_width,
        lm_name=lm_name,
    )

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Supports both your full checkpoint format and raw projector state_dict.
    if isinstance(ckpt, dict) and "projector_state_dict" in ckpt:
        model.projector.load_state_dict(ckpt["projector_state_dict"])
    else:
        model.projector.load_state_dict(ckpt)

    print(f"loaded checkpoint: {checkpoint_path}")

    # Freeze/eval safety.
    for p in model.vision_encoder.parameters():
        p.requires_grad_(False)
    for p in model.lm.model.parameters():
        p.requires_grad_(False)
    for p in model.projector.parameters():
        p.requires_grad_(False)

    model.vision_encoder.eval()
    model.lm.model.eval()
    model.projector.eval()

    assert_only_projector_trainable(model)

    tokenizer = model.lm.tokenizer
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading dataset...")
    dataset = CORDDataset(
        split=split,
        max_samples=num_samples,
        dataset_name=dataset_name,
    )

    print(f"eval dataset: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=_eval_collate,
    )

    records: list[dict[str, Any]] = []

    counts = {
        "valid_json": 0,
        "schema": 0,
        "format_adherence": 0,
        "non_empty_line_items": 0,
        "empty_line_items": 0,
        "numeric_name": 0,
        "duplicate_items": 0,
        "extra_text": 0,
    }

    total_reward = 0.0
    total_gen_length = 0

    print("running evaluation...")

    for idx, sample in enumerate(tqdm(loader, total=len(loader))):
        image = sample["image"][0]
        ground_truth = sample["label"][0]

        with torch.no_grad():
            gen = generate_k_outputs(
                model=model,
                image=image,
                tokenizer=tokenizer,
                instruction=instruction,
                k=1,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                top_p=top_p,
                generation_repetition_penalty=generation_repetition_penalty,
                do_sample=do_sample,
            )

        generated = gen.texts[0]
        reward = compute_reward(generated, ground_truth)

        analysis = analyze_generation(generated)

        counts["valid_json"] += int(analysis["valid_json"])
        counts["schema"] += int(analysis["schema"])
        counts["format_adherence"] += int(analysis["format_adherence"])
        counts["non_empty_line_items"] += int(analysis["non_empty_line_items"])
        counts["empty_line_items"] += int(analysis["empty_line_items"])
        counts["numeric_name"] += int(analysis["numeric_name"])
        counts["duplicate_items"] += int(analysis["duplicate_items"])
        counts["extra_text"] += int(analysis["extra_text"])

        total_reward += reward.total
        total_gen_length += len(generated)

        records.append(
            {
                "index": idx,
                "generated": generated,
                "ground_truth": ground_truth,
                "reward": asdict(reward),
                "analysis": analysis,
            }
        )

    n = max(1, len(records))

    metrics = EvalMetrics(
        num_samples=len(records),
        valid_json_rate=counts["valid_json"] / n,
        schema_rate=counts["schema"] / n,
        format_adherence_rate=counts["format_adherence"] / n,
        non_empty_line_items_rate=counts["non_empty_line_items"] / n,
        empty_line_items_rate=counts["empty_line_items"] / n,
        numeric_name_rate=counts["numeric_name"] / n,
        duplicate_item_rate=counts["duplicate_items"] / n,
        extra_text_rate=counts["extra_text"] / n,
        average_reward=total_reward / n,
        average_generation_length=total_gen_length / n,
    )

    metrics_path = output_dir / "metrics.json"
    samples_path = output_dir / "samples.jsonl"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)

    with samples_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print_metrics(metrics)
    print(f"\nmetrics saved to: {metrics_path}")
    print(f"samples saved to: {samples_path}")

    return metrics


def analyze_generation(text: str) -> dict[str, Any]:
    """
    Analyze a generated output for assignment-relevant format metrics.
    """

    stripped = text.strip()
    extracted = extract_json(stripped)

    result = {
        "valid_json": False,
        "schema": False,
        "format_adherence": False,
        "non_empty_line_items": False,
        "empty_line_items": False,
        "numeric_name": False,
        "duplicate_items": False,
        "extra_text": False,
        "num_line_items": 0,
    }

    if extracted is None:
        return result

    try:
        parsed = json.loads(extracted)
    except json.JSONDecodeError:
        return result

    result["valid_json"] = True
    result["extra_text"] = stripped != extracted.strip()

    if not isinstance(parsed, dict):
        return result

    has_line_items = "line_items" in parsed
    has_total = "total" in parsed

    line_items = parsed.get("line_items")

    result["schema"] = (
        has_line_items
        and has_total
        and isinstance(line_items, list)
    )

    result["format_adherence"] = result["valid_json"] and result["schema"]

    if isinstance(line_items, list):
        result["num_line_items"] = len(line_items)
        result["non_empty_line_items"] = len(line_items) > 0
        result["empty_line_items"] = len(line_items) == 0
        result["numeric_name"] = _has_numeric_only_name(line_items)
        result["duplicate_items"] = _has_duplicate_items(line_items)

    return result


def _has_numeric_only_name(line_items: list[Any]) -> bool:
    for item in line_items:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name", "")).strip()

        if not name:
            continue

        if re.fullmatch(r"[\d\s.,]+", name):
            return True

    return False


def _has_duplicate_items(line_items: list[Any]) -> bool:
    signatures = []

    for item in line_items:
        if not isinstance(item, dict):
            continue

        name = _norm_text(item.get("name", ""))
        count = _norm_text(item.get("count", ""))
        price = _norm_number(item.get("price", ""))

        signatures.append((name, count, price))

    if len(signatures) <= 1:
        return False

    return len(set(signatures)) < len(signatures)


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).lower().strip())


def _norm_number(value: Any) -> str:
    return re.sub(r"\D", "", str(value))


def print_metrics(metrics: EvalMetrics):
    print("\nEvaluation metrics")
    print("------------------")
    print(f"num_samples:                 {metrics.num_samples}")
    print(f"valid_json_rate:             {metrics.valid_json_rate:.2%}")
    print(f"schema_rate:                 {metrics.schema_rate:.2%}")
    print(f"format_adherence_rate:       {metrics.format_adherence_rate:.2%}")
    print(f"non_empty_line_items_rate:   {metrics.non_empty_line_items_rate:.2%}")
    print(f"empty_line_items_rate:       {metrics.empty_line_items_rate:.2%}")
    print(f"numeric_name_rate:           {metrics.numeric_name_rate:.2%}")
    print(f"duplicate_item_rate:         {metrics.duplicate_item_rate:.2%}")
    print(f"extra_text_rate:             {metrics.extra_text_rate:.2%}")
    print(f"average_reward:              {metrics.average_reward:.4f}")
    print(f"average_generation_length:   {metrics.average_generation_length:.1f}")


def _eval_collate(batch):
    return {
        "image": [sample["image"] for sample in batch],
        "label": [sample["label"] for sample in batch],
    }