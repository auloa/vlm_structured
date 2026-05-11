"""
Donut inference script with custom processor image size (640x960).
Runs on a sample of CORD-v2 and visualizes / saves results.

Usage:
    python donut_inference.py --samples 8 --width 640 --height 960 --output ./results
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from vlm.data.dataset import CORDDataset

# ── inference helpers ────────────────────────────────────────────────────────

MODEL_ID = "naver-clova-ix/donut-base-finetuned-cord-v2"
TASK_PROMPT = "<s_cord-v2>"


def load_model_and_processor(width: int, height: int):
    """Load Donut and patch the processor to use a custom image size."""
    processor = DonutProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

    # ── custom image size ────────────────────────────────────────────────────
    # DonutImageProcessor stores the target size as (height, width)
    processor.image_processor.size = {"height": height, "width": width}
    processor.image_processor.do_align_long_axis = False
    # Keep the encoder config in sync so positional embeddings are consistent
    model.config.encoder.image_size = [height, width]
    # ────────────────────────────────────────────────────────────────────────

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"Model on {device} | image size: {height}h X {width}w")
    return processor, model, device


def run_inference(image: Image.Image, processor, model, device: str) -> str:
    """Run Donut on a single PIL image and return decoded JSON string."""
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    decoder_input_ids = processor.tokenizer(
        TASK_PROMPT,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    seq = processor.batch_decode(outputs.sequences)[0]
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, ""
    )
    seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # strip task token
    return seq


# ── visualisation ─────────────────────────────────────────────────────────-─

def pretty(text: str) -> str:
    """Attempt to pretty-print JSON, fall back to raw."""
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except Exception:
        return text


def visualize_sample(
    idx: int,
    image: Image.Image,
    gt: str,
    pred: str,
    output_dir: Path,
    show: bool = False,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    fig.suptitle(f"Sample {idx}", fontsize=14, fontweight="bold")

    # image
    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    # ground truth
    axes[1].text(
        0.02, 0.98, pretty(gt),
        transform=axes[1].transAxes,
        fontsize=7, verticalalignment="top", family="monospace",
        wrap=True,
    )
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    # prediction
    axes[2].text(
        0.02, 0.98, pretty(pred),
        transform=axes[2].transAxes,
        fontsize=7, verticalalignment="top", family="monospace",
        wrap=True,
    )
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = output_dir / f"sample_{idx:03d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  saved → {out_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=8, help="Number of samples to run")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, default="./donut_results")
    parser.add_argument("--show", action="store_true", help="Also display plots interactively")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load processor + model
    processor, model, device = load_model_and_processor(args.width, args.height)

    # load dataset (no tokenizer filter needed for inference)
    print(f"Loading CORD-v2 [{args.split}] …")
    dataset = CORDDataset(split=args.split, max_samples=args.samples, prefer_high_resolution=True)
    print(
        f"  loaded {dataset.num_loaded} | parse failures: {dataset.num_parse_failed} "
        f"| kept: {dataset.num_after_filtering}"
    )

    # run inference
    results = []
    for i in range(min(args.samples, len(dataset))):
        sample = dataset[i]
        image: Image.Image = sample["image"]
        gt: str = sample["label"]

        print(f"[{i+1}/{args.samples}] inferring …", end=" ", flush=True)
        pred = run_inference(image, processor, model, device)
        print("done")

        results.append({"index": i, "ground_truth": gt, "prediction": pred})
        visualize_sample(i, image, gt, pred, output_dir, show=args.show)

    # save all results as JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()