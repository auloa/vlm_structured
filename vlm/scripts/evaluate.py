import json
from pathlib import Path
from typing import Any, TypedDict

import torch
from vlm.config import DEFAULT_CONFIG
from vlm.data.dataset import CORDDataset
from vlm.models.receipt_vlm import ReceiptVLM
from vlm.training.generate import generate_k_outputs
from vlm.training.rewards import compute_reward, extract_json
from vlm.utils.device import get_device


class EvalResult(TypedDict):
    output: str
    ground_truth: str
    reward: float
    json_like: bool
    json_valid: bool
    schema_ok: bool
    format_ok: bool
    line_items_populated: bool
    content_score: float
    anti_hallucination_score: float
    total_match_score: float


def evaluate(
    dataset_name: str,
    split: str,
    num_samples: int,
    vision_model_name: str,
    image_height: int,
    image_width: int,
    lm_name: str,
    instruction: str,
    checkpoint_path: str | Path | None,
    label: str,
    max_completion_tokens: int,
    temperature: float,
):
    device = get_device()
    print(f"device: {device}")

    model = ReceiptVLM(
        device=device,
        vision_model_name=vision_model_name,
        image_height=image_height,
        image_width=image_width,
        lm_name=lm_name,
    )

    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.projector.load_state_dict(ckpt["projector_state_dict"])
        print(f"loaded {label} checkpoint: {checkpoint_path}")

    tokenizer = model.lm.tokenizer

    dataset = CORDDataset(
        split=split,
        max_samples=num_samples,
        dataset_name=dataset_name,
    )

    print(
        f"dataset ready: {len(dataset)} samples "

    )
    print(f"evaluating {label} on {len(dataset)} samples...")

    results: list[EvalResult] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        image = sample["image"]
        ground_truth = sample["label"]

        gen = generate_k_outputs(
            model=model,
            image=image,
            tokenizer=tokenizer,
            instruction=instruction,
            k=1,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            do_sample=False,
        )

        output = gen.texts[0]
        reward = compute_reward(output, ground_truth)
        extracted = extract_json(output)

        json_valid = False
        schema_ok = False
        format_ok = False

        if extracted is not None:
            try:
                parsed = json.loads(extracted)
                json_valid = True
                schema_ok = (
                    "line_items" in parsed
                    and "total" in parsed
                    and isinstance(parsed["line_items"], list)
                )
                format_ok = json_valid and schema_ok
            except Exception:
                pass

        results.append(
            {
                "output": output,
                "ground_truth": ground_truth,
                "reward": reward.total,
                "json_like": reward.json_like > 0,
                "json_valid": json_valid,
                "schema_ok": schema_ok,
                "format_ok": format_ok,
                "line_items_populated": reward.line_items_populated > 0,
                "content_score": reward.content,
                "anti_hallucination_score": reward.anti_hallucination,
                "total_match_score": reward.total_match,
            }
        )

        if (idx + 1) % 10 == 0:
            format_rate = sum(r["format_ok"] for r in results) / len(results)
            json_like_rate = sum(r["json_like"] for r in results) / len(results)

            print(
                f"[{idx + 1}/{len(dataset)}] "
                f"format adherence: {format_rate:.1%} | "
                f"json-like: {json_like_rate:.1%}"
            )

    return _summarize_results(results, label)


def _summarize_results(results: list[EvalResult], label: str) -> dict[str, Any]:
    n = max(1, len(results))

    json_like_rate = sum(r["json_like"] for r in results) / n
    format_adherence = sum(r["format_ok"] for r in results) / n
    json_valid_rate = sum(r["json_valid"] for r in results) / n
    schema_rate = sum(r["schema_ok"] for r in results) / n
    populated_rate = sum(r["line_items_populated"] for r in results) / n
    mean_reward = sum(r["reward"] for r in results) / n
    mean_content_score = sum(r["content_score"] for r in results) / n
    mean_anti_hallucination_score = (
        sum(r["anti_hallucination_score"] for r in results) / n
    )
    mean_total_match_score = sum(r["total_match_score"] for r in results) / n

    print(f"\nEvaluation — {label}")
    print("-" * 60)
    print(f"samples evaluated:              {len(results)}")
    print(f"json-like rate:                 {json_like_rate:.1%}")
    print(f"format adherence rate:          {format_adherence:.1%}")
    print(f"valid JSON rate:                {json_valid_rate:.1%}")
    print(f"schema adherence rate:          {schema_rate:.1%}")
    print(f"line_items populated rate:      {populated_rate:.1%}")
    print(f"mean reward:                    {mean_reward:.3f}")
    print(f"mean content score:             {mean_content_score:.3f}")
    print(f"mean anti-hallucination score:  {mean_anti_hallucination_score:.3f}")
    print(f"mean total-match score:         {mean_total_match_score:.3f}")

    print("\nsample outputs:")
    for i, result in enumerate(results[:3]):
        gt = json.loads(result["ground_truth"])

        print("\n" + "=" * 60)
        print(f"sample {i}")
        print("-" * 60)
        print("output:")
        print(result["output"][:500])
        print("\nground truth summary:")
        print(f"total={gt.get('total')}, items={len(gt.get('line_items', []))}")
        print(
            f"json-like={result['json_like']} | "
            f"format ok={result['format_ok']} | "
            f"reward={result['reward']:.3f}"
        )

    return {
        "json_like_rate": json_like_rate,
        "format_adherence": format_adherence,
        "json_valid_rate": json_valid_rate,
        "schema_rate": schema_rate,
        "populated_rate": populated_rate,
        "mean_reward": mean_reward,
        "mean_content_score": mean_content_score,
        "mean_anti_hallucination_score": mean_anti_hallucination_score,
        "mean_total_match_score": mean_total_match_score,
        "results": results,
    }


def compare_sft_vs_rl():
    cfg = DEFAULT_CONFIG

    print("\n" + "=" * 80)
    print("SFT checkpoint")
    print("=" * 80)

    sft = evaluate(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.test_split,
        num_samples=cfg.data.test_samples,
        vision_model_name=cfg.vision.model_name,
        image_height=cfg.vision.image_height,
        image_width=cfg.vision.image_width,
        lm_name=cfg.model.lm_name,
        instruction=cfg.model.instruction,
        checkpoint_path=cfg.paths.sft_best_checkpoint,
        label="SFT",
        max_completion_tokens=cfg.eval.max_completion_tokens,
        temperature=cfg.eval.temperature,
    )

    print("\n" + "=" * 80)
    print("RL checkpoint")
    print("=" * 80)

    rl = evaluate(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.test_split,
        num_samples=cfg.data.test_samples,
        vision_model_name=cfg.vision.model_name,
        image_height=cfg.vision.image_height,
        image_width=cfg.vision.image_width,
        lm_name=cfg.model.lm_name,
        instruction=cfg.model.instruction,
        checkpoint_path=cfg.paths.rl_best_checkpoint,
        label="RL",
        max_completion_tokens=cfg.eval.max_completion_tokens,
        temperature=cfg.eval.temperature,
    )

    _print_comparison(sft, rl)
    _save_eval_summary(sft, rl)


def _print_comparison(sft: dict, rl: dict):
    def pct_delta(a, b):
        return f"{b - a:+.1%}"

    def float_delta(a, b):
        return f"{b - a:+.3f}"

    print("\nSFT vs RL")
    print("-" * 80)
    print(f"{'metric':<25} {'SFT':>12} {'RL':>12} {'delta':>12}")
    print("-" * 80)

    rows = [
        (
            "json-like",
            f"{sft['json_like_rate']:.1%}",
            f"{rl['json_like_rate']:.1%}",
            pct_delta(sft["json_like_rate"], rl["json_like_rate"]),
        ),
        (
            "format adherence",
            f"{sft['format_adherence']:.1%}",
            f"{rl['format_adherence']:.1%}",
            pct_delta(sft["format_adherence"], rl["format_adherence"]),
        ),
        (
            "valid JSON",
            f"{sft['json_valid_rate']:.1%}",
            f"{rl['json_valid_rate']:.1%}",
            pct_delta(sft["json_valid_rate"], rl["json_valid_rate"]),
        ),
        (
            "schema adherence",
            f"{sft['schema_rate']:.1%}",
            f"{rl['schema_rate']:.1%}",
            pct_delta(sft["schema_rate"], rl["schema_rate"]),
        ),
        (
            "mean reward",
            f"{sft['mean_reward']:.3f}",
            f"{rl['mean_reward']:.3f}",
            float_delta(sft["mean_reward"], rl["mean_reward"]),
        ),
    ]

    for metric, sft_value, rl_value, delta in rows:
        print(f"{metric:<25} {sft_value:>12} {rl_value:>12} {delta:>12}")


def _save_eval_summary(sft: dict, rl: dict):
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "sft": _strip_large_results(sft),
        "rl": _strip_large_results(rl),
        "delta": {
            "json_like_rate": rl["json_like_rate"] - sft["json_like_rate"],
            "format_adherence": rl["format_adherence"] - sft["format_adherence"],
            "json_valid_rate": rl["json_valid_rate"] - sft["json_valid_rate"],
            "schema_rate": rl["schema_rate"] - sft["schema_rate"],
            "mean_reward": rl["mean_reward"] - sft["mean_reward"],
        },
        "sample_outputs": {
            "sft": sft["results"][:3],
            "rl": rl["results"][:3],
        },
    }

    output_path = output_dir / "eval_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nsaved evaluation summary: {output_path}")


def _strip_large_results(metrics: dict):
    return {k: v for k, v in metrics.items() if k != "results"}


if __name__ == "__main__":
    compare_sft_vs_rl()