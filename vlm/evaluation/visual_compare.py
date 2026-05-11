import base64
import html
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from vlm.data.dataset import CORDDataset
from vlm.models.receipt_vlm import ReceiptVLM
from vlm.training.generate import generate_k_outputs
from vlm.training.rewards import compute_reward, extract_json
from vlm.utils.device import get_device
from vlm.utils.training import set_seed


def build_visual_comparison_report(
    dataset_name: str,
    split: str,
    num_samples: int,
    vision_model_name: str,
    image_height: int,
    image_width: int,
    lm_name: str,
    instruction: str,
    checkpoint_path: str | Path,
    output_html_path: str | Path,
    max_completion_tokens: int = 192,
    temperature: float = 0.1,
    do_sample: bool = False,
    seed: int = 42,
):
    """
    Creates an HTML report comparing:
        receipt image | ground-truth JSON | model-generated JSON | reward/diagnostics

    Works for both SFT and RL projector checkpoints.
    """

    set_seed(seed)

    device = get_device()
    print(f"device: {device}")

    checkpoint_path = Path(checkpoint_path)
    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    print("loading model...")
    model = ReceiptVLM(
        device=device,
        vision_model_name=vision_model_name,
        image_height=image_height,
        image_width=image_width,
        lm_name=lm_name,
    )

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "projector_state_dict" in ckpt:
        model.projector.load_state_dict(ckpt["projector_state_dict"])
    else:
        model.projector.load_state_dict(ckpt)

    print(f"loaded checkpoint: {checkpoint_path}")

    for p in model.parameters():
        p.requires_grad_(False)

    model.vision_encoder.eval()
    model.lm.model.eval()
    model.projector.eval()

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

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=_visual_collate,
    )

    rows: list[dict[str, Any]] = []

    print("generating outputs...")

    for idx, sample in enumerate(tqdm(loader, total=len(loader))):
        image: Image.Image = sample["image"][0]
        ground_truth: str = sample["label"][0]

        with torch.no_grad():
            gen = generate_k_outputs(
                model=model,
                image=image,
                tokenizer=tokenizer,
                instruction=instruction,
                k=1,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )

        generated = gen.texts[0]
        reward = compute_reward(generated, ground_truth)

        rows.append(
            {
                "idx": idx,
                "image": image,
                "ground_truth": ground_truth,
                "generated": generated,
                "reward": reward,
                "diagnostics": analyze_pair(generated, ground_truth),
            }
        )

    html_text = render_html_report(
        rows=rows,
        title=f"Visual comparison: {checkpoint_path}",
    )

    output_html_path.write_text(html_text, encoding="utf-8")

    print(f"\nreport saved to: {output_html_path}")


def analyze_pair(generated: str, ground_truth: str) -> dict[str, Any]:
    extracted = extract_json(generated)

    diagnostics = {
        "valid_json": False,
        "schema": False,
        "line_items_pred": 0,
        "line_items_gt": 0,
        "empty_line_items": False,
        "numeric_name": False,
        "duplicate_items": False,
        "total_pred": "",
        "total_gt": "",
        "total_exact_match": False,
    }

    try:
        gt = json.loads(ground_truth)
    except Exception:
        gt = {}

    gt_items = gt.get("line_items", [])
    if isinstance(gt_items, list):
        diagnostics["line_items_gt"] = len(gt_items)

    diagnostics["total_gt"] = str(gt.get("total", ""))

    if extracted is None:
        return diagnostics

    try:
        pred = json.loads(extracted)
    except Exception:
        return diagnostics

    diagnostics["valid_json"] = True

    if not isinstance(pred, dict):
        return diagnostics

    line_items = pred.get("line_items", [])
    total = pred.get("total", "")

    diagnostics["schema"] = (
        "line_items" in pred
        and "total" in pred
        and isinstance(line_items, list)
    )

    diagnostics["total_pred"] = str(total)
    diagnostics["total_exact_match"] = _norm_number(total) == _norm_number(gt.get("total", ""))

    if isinstance(line_items, list):
        diagnostics["line_items_pred"] = len(line_items)
        diagnostics["empty_line_items"] = len(line_items) == 0
        diagnostics["numeric_name"] = _has_numeric_only_name(line_items)
        diagnostics["duplicate_items"] = _has_duplicate_items(line_items)

    return diagnostics


def render_html_report(rows: list[dict[str, Any]], title: str) -> str:
    cards = []

    for row in rows:
        image_src = image_to_base64(row["image"])

        gt_pretty = pretty_json(row["ground_truth"])
        pred_pretty = pretty_json_or_text(row["generated"])

        reward = row["reward"]
        diag = row["diagnostics"]

        reward_table = f"""
        <table class="metrics">
          <tr><td>Total reward</td><td>{reward.total:.4f}</td></tr>
          <tr><td>JSON valid reward</td><td>{reward.json_valid:.4f}</td></tr>
          <tr><td>Schema reward</td><td>{reward.schema:.4f}</td></tr>
          <tr><td>Line items populated reward</td><td>{reward.line_items_populated:.4f}</td></tr>
          <tr><td>Content reward</td><td>{reward.content:.4f}</td></tr>
          <tr><td>Anti hallucination reward</td><td>{reward.anti_hallucination:.4f}</td></tr>
          <tr><td>Total match reward</td><td>{reward.total_match:.4f}</td></tr>
        </table>
        """

        diag_table = f"""
        <table class="metrics">
          <tr><td>Valid JSON</td><td>{badge(diag["valid_json"])}</td></tr>
          <tr><td>Schema</td><td>{badge(diag["schema"])}</td></tr>
          <tr><td>GT line items</td><td>{diag["line_items_gt"]}</td></tr>
          <tr><td>Pred line items</td><td>{diag["line_items_pred"]}</td></tr>
          <tr><td>Empty line_items</td><td>{badge(diag["empty_line_items"], invert=True)}</td></tr>
          <tr><td>Numeric-only name</td><td>{badge(diag["numeric_name"], invert=True)}</td></tr>
          <tr><td>Duplicate items</td><td>{badge(diag["duplicate_items"], invert=True)}</td></tr>
          <tr><td>Total exact match</td><td>{badge(diag["total_exact_match"])}</td></tr>
          <tr><td>GT total</td><td>{html.escape(str(diag["total_gt"]))}</td></tr>
          <tr><td>Pred total</td><td>{html.escape(str(diag["total_pred"]))}</td></tr>
        </table>
        """

        card = f"""
        <section class="card">
          <h2>Sample {row["idx"]}</h2>

          <div class="grid">
            <div class="image-pane">
              <h3>Receipt image</h3>
              <img src="{image_src}" />
            </div>

            <div class="json-pane">
              <h3>Ground truth</h3>
              <pre>{html.escape(gt_pretty)}</pre>
            </div>

            <div class="json-pane">
              <h3>Model output</h3>
              <pre>{html.escape(pred_pretty)}</pre>
            </div>
          </div>

          <div class="grid-small">
            <div>
              <h3>Diagnostics</h3>
              {diag_table}
            </div>
            <div>
              <h3>Reward breakdown</h3>
              {reward_table}
            </div>
          </div>
        </section>
        """

        cards.append(card)

    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>{html.escape(title)}</title>
      <style>
        body {{
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 24px;
          background: #f7f7f8;
          color: #111;
        }}

        h1 {{
          margin-bottom: 8px;
        }}

        .subtitle {{
          color: #555;
          margin-bottom: 24px;
        }}

        .card {{
          background: white;
          border: 1px solid #ddd;
          border-radius: 12px;
          padding: 20px;
          margin-bottom: 28px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }}

        .grid {{
          display: grid;
          grid-template-columns: 1fr 1fr 1fr;
          gap: 16px;
          align-items: start;
        }}

        .grid-small {{
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 16px;
          margin-top: 16px;
        }}

        .image-pane img {{
          max-width: 100%;
          max-height: 650px;
          border: 1px solid #ddd;
          border-radius: 8px;
          background: #fff;
        }}

        pre {{
          white-space: pre-wrap;
          word-break: break-word;
          background: #f2f2f4;
          border: 1px solid #ddd;
          border-radius: 8px;
          padding: 12px;
          max-height: 650px;
          overflow: auto;
          font-size: 13px;
          line-height: 1.35;
        }}

        table.metrics {{
          border-collapse: collapse;
          width: 100%;
          background: #fafafa;
          border: 1px solid #ddd;
          border-radius: 8px;
          overflow: hidden;
        }}

        table.metrics td {{
          border-bottom: 1px solid #e5e5e5;
          padding: 8px 10px;
          font-size: 14px;
        }}

        table.metrics td:first-child {{
          font-weight: 600;
          width: 55%;
        }}

        .yes {{
          color: #087f23;
          font-weight: 700;
        }}

        .no {{
          color: #b00020;
          font-weight: 700;
        }}

        @media (max-width: 1200px) {{
          .grid {{
            grid-template-columns: 1fr;
          }}

          .grid-small {{
            grid-template-columns: 1fr;
          }}
        }}
      </style>
    </head>
    <body>
      <h1>{html.escape(title)}</h1>
      <div class="subtitle">
        Side-by-side receipt image, ground truth JSON, model output, reward, and diagnostics.
      </div>
      {''.join(cards)}
    </body>
    </html>
    """


def image_to_base64(image: Image.Image, max_width: int = 900) -> str:
    image = image.convert("RGB")

    if image.width > max_width:
        scale = max_width / image.width
        new_size = (max_width, int(image.height * scale))
        image = image.resize(new_size)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def pretty_json(text: str) -> str:
    try:
        obj = json.loads(text)
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return text


def pretty_json_or_text(text: str) -> str:
    extracted = extract_json(text)

    if extracted is not None:
        try:
            obj = json.loads(extracted)
            pretty = json.dumps(obj, indent=2, ensure_ascii=False)

            if text.strip() != extracted.strip():
                return (
                    "EXTRACTED JSON:\n"
                    + pretty
                    + "\n\nRAW MODEL OUTPUT:\n"
                    + text
                )

            return pretty
        except Exception:
            pass

    return text


def badge(value: bool, invert: bool = False) -> str:
    """
    For normal metrics:
        True is good, False is bad.

    For bad-condition metrics like numeric_name:
        invert=True makes False good and True bad.
    """
    good = value if not invert else not value

    if good:
        return '<span class="yes">YES</span>'

    return '<span class="no">NO</span>'


def _has_numeric_only_name(line_items: list[Any]) -> bool:
    for item in line_items:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name", "")).strip()

        if name and re.fullmatch(r"[\d\s.,]+", name):
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


def _visual_collate(batch):
    return {
        "image": [sample["image"] for sample in batch],
        "label": [sample["label"] for sample in batch],
    }