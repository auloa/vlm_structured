import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image, ImageTk
from vlm.config import DEFAULT_CONFIG
from vlm.data.dataset import parse_ground_truth
from vlm.models.vision_encoder import DonutVisionEncoder
from vlm.utils.device import get_device

FLAGS_PATH = Path("data/flagged_cord_raw_images.jsonl")
PROCESSED_IMAGE_DIR = Path("data/cord_processed_review_images")

# Set to None to review/export all raw rows in the split.
MAX_REVIEW_SAMPLES = None

# Keep True if you only want rows whose ground_truth parses into your simplified schema.
SKIP_UNPARSEABLE_LABELS = True


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def summarize_label(label: str) -> str:
    try:
        parsed = json.loads(label)
        items = parsed.get("line_items", [])
        total = parsed.get("total")

        lines = [
            f"total: {total}",
            f"line_items: {len(items)}",
            "",
        ]

        for item in items[:8]:
            lines.append(
                f"- {item.get('name', '')} | "
                f"count={item.get('count', '')} | "
                f"price={item.get('price', '')}"
            )

        if len(items) > 8:
            lines.append(f"... +{len(items) - 8} more")

        return "\n".join(lines)

    except Exception:
        return label[:1000]


def build_review_rows(
    dataset_name: str,
    split: str,
    max_samples: int | None,
    skip_unparseable_labels: bool,
) -> list[dict]:
    raw = load_dataset(dataset_name, split=split)

    rows: list[dict] = []

    for raw_index in range(len(raw)):
        item = raw[raw_index]

        image = item["image"].convert("RGB")
        ground_truth = item["ground_truth"]

        try:
            parsed = parse_ground_truth(ground_truth)
            label = compact_json(parsed)
            parse_ok = True
            parse_error = ""
        except Exception as exc:
            if skip_unparseable_labels:
                continue

            label = ground_truth
            parse_ok = False
            parse_error = str(exc)

        rows.append(
            {
                "raw_index": raw_index,
                "image": image,
                "label": label,
                "parse_ok": parse_ok,
                "parse_error": parse_error,
                "raw_size": image.size,
            }
        )

        if max_samples is not None and len(rows) >= max_samples:
            break

    return rows


class RawCORDImageFlagger:
    def __init__(
        self,
        root: tk.Tk,
        rows: list[dict],
        output_path: Path,
        vision_encoder: DonutVisionEncoder,
        processed_dir: Path,
    ):
        self.root = root
        self.rows = rows
        self.output_path = output_path
        self.vision_encoder = vision_encoder
        self.processed_dir = processed_dir

        self.index = 0
        self.show_processed = False
        self.flags = self.load_existing_flags()

        self.root.title("Raw CORD Image Flagger")

        self.image_label = ttk.Label(root)
        self.image_label.pack(padx=10, pady=10)

        self.info_text = tk.Text(root, height=17, width=125, wrap="word")
        self.info_text.pack(padx=10, pady=10)

        reason_frame = ttk.Frame(root)
        reason_frame.pack(pady=5)

        ttk.Label(reason_frame, text="Bad reason:").grid(row=0, column=0, padx=5)

        self.reason_var = tk.StringVar(value="bad_quality")
        self.reason_box = ttk.Combobox(
            reason_frame,
            textvariable=self.reason_var,
            values=[
                "bad_quality",
                "blurred",
                "too_small",
                "cutoff",
                "rotated",
                "not_receipt",
                "unreadable",
                "wrong_label",
                "bad_after_processing",
                "deleted_from_processed_folder",
                "other",
            ],
            state="readonly",
            width=30,
        )
        self.reason_box.grid(row=0, column=1, padx=5)

        nav_frame = ttk.Frame(root)
        nav_frame.pack(pady=8)

        ttk.Button(nav_frame, text="Previous", command=self.prev_sample).grid(
            row=0, column=0, padx=5
        )
        ttk.Button(nav_frame, text="Good [g]", command=lambda: self.flag("good")).grid(
            row=0, column=1, padx=5
        )
        ttk.Button(nav_frame, text="Bad [b]", command=lambda: self.flag("bad")).grid(
            row=0, column=2, padx=5
        )
        ttk.Button(nav_frame, text="Unsure [u]", command=lambda: self.flag("unsure")).grid(
            row=0, column=3, padx=5
        )
        ttk.Button(nav_frame, text="Next", command=self.next_sample).grid(
            row=0, column=4, padx=5
        )

        extra_frame = ttk.Frame(root)
        extra_frame.pack(pady=5)

        ttk.Button(
            extra_frame,
            text="Toggle raw/processed [v]",
            command=self.toggle_view,
        ).grid(row=0, column=0, padx=5)

        ttk.Button(
            extra_frame,
            text="Save current processed [s]",
            command=self.save_current_processed_image,
        ).grid(row=0, column=1, padx=5)

        ttk.Button(
            extra_frame,
            text="Export all processed [S]",
            command=self.export_all_processed_images,
        ).grid(row=0, column=2, padx=5)

        ttk.Button(
            extra_frame,
            text="Build flags from deleted images [j]",
            command=self.build_flags_from_missing_processed_images,
        ).grid(row=0, column=3, padx=5)

        self.root.bind("<Left>", lambda event: self.prev_sample())
        self.root.bind("<Right>", lambda event: self.next_sample())
        self.root.bind("<space>", lambda event: self.next_sample())

        self.root.bind("g", lambda event: self.flag("good"))
        self.root.bind("b", lambda event: self.flag("bad"))
        self.root.bind("u", lambda event: self.flag("unsure"))

        self.root.bind("v", lambda event: self.toggle_view())
        self.root.bind("s", lambda event: self.save_current_processed_image())
        self.root.bind("S", lambda event: self.export_all_processed_images())
        self.root.bind("j", lambda event: self.build_flags_from_missing_processed_images())

        self.root.bind("q", lambda event: self.root.destroy())
        self.root.bind("<Escape>", lambda event: self.root.destroy())

        self.current_photo = None
        self.show_sample()

    def current_row(self) -> dict:
        return self.rows[self.index]

    def load_existing_flags(self) -> dict[int, dict]:
        flags: dict[int, dict] = {}

        if not self.output_path.exists():
            return flags

        with open(self.output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    flags[int(item["raw_index"])] = item

        return flags

    def save_flags(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            for raw_index in sorted(self.flags):
                f.write(json.dumps(self.flags[raw_index], ensure_ascii=False) + "\n")

    def processed_filename_for_raw_index(self, raw_index: int) -> Path:
        return self.processed_dir / f"cord_{raw_index:06d}.png"

    def resize_for_display(self, image: Image.Image, max_width=900, max_height=680):
        image = image.copy()
        image.thumbnail((max_width, max_height))
        return image

    def get_processed_image(self, image: Image.Image) -> tuple[Image.Image, torch.Tensor]:
        processor = self.vision_encoder.processor

        inputs = processor(
            images=[image.convert("RGB")],
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"][0].detach().cpu().float()

        display_tensor = pixel_values.clone()

        image_mean = getattr(processor, "image_mean", None)
        image_std = getattr(processor, "image_std", None)

        if image_mean is not None and image_std is not None:
            mean = torch.tensor(image_mean).view(-1, 1, 1)
            std = torch.tensor(image_std).view(-1, 1, 1)
            display_tensor = display_tensor * std + mean

        display_tensor = display_tensor.clamp(0, 1)

        if display_tensor.shape[0] == 1:
            display_tensor = display_tensor.repeat(3, 1, 1)

        display_tensor = display_tensor.permute(1, 2, 0)
        array = (display_tensor.numpy() * 255).astype("uint8")

        processed_image = Image.fromarray(array)

        return processed_image, pixel_values

    def get_current_display_image(self) -> tuple[Image.Image, str]:
        row = self.current_row()
        raw_image = row["image"]

        if self.show_processed:
            processed_image, pixel_values = self.get_processed_image(raw_image)
            meta = (
                f"View mode: PROCESSED BY VISION PROCESSOR\n"
                f"Processed display image size: {processed_image.size}\n"
                f"Pixel tensor shape: {tuple(pixel_values.shape)}\n"
                f"Pixel tensor min/max: {pixel_values.min().item():.4f} / {pixel_values.max().item():.4f}\n"
                f"Pixel tensor mean/std: {pixel_values.mean().item():.4f} / {pixel_values.std().item():.4f}"
            )
            return processed_image, meta

        meta = (
            f"View mode: RAW CORD IMAGE\n"
            f"Raw image size: {raw_image.size}"
        )
        return raw_image, meta

    def show_sample(self):
        row = self.current_row()

        raw_index = row["raw_index"]
        image = row["image"]
        label = row["label"]

        display_source, mode_meta = self.get_current_display_image()

        display_image = self.resize_for_display(display_source)
        self.current_photo = ImageTk.PhotoImage(display_image)
        self.image_label.configure(image=self.current_photo)

        current_flag = self.flags.get(raw_index)
        if current_flag:
            flag_text = f"{current_flag.get('flag')} | reason={current_flag.get('reason', '')}"
        else:
            flag_text = "not flagged"

        processed_path = self.processed_filename_for_raw_index(raw_index)
        processed_exists = processed_path.exists()

        info = (
            f"Review sample: {self.index + 1}/{len(self.rows)}\n"
            f"Raw CORD index: {raw_index}\n"
            f"Original image size: {image.size}\n"
            f"Label parse ok: {row['parse_ok']}\n"
            f"Parse error: {row['parse_error']}\n"
            f"Current flag: {flag_text}\n"
            f"Flags saved to: {self.output_path}\n"
            f"Processed folder: {self.processed_dir}\n"
            f"Current processed file exists: {processed_exists}\n"
            f"{'-' * 100}\n"
            f"{mode_meta}\n"
            f"{'-' * 100}\n"
            f"Keys:\n"
            f"  g=good | b=bad | u=unsure\n"
            f"  v=toggle raw/processed\n"
            f"  s=save current processed image\n"
            f"  S=export all processed images\n"
            f"  j=build bad flags from deleted/missing processed images\n"
            f"  right/space=next | left=previous | q=quit\n"
            f"{'-' * 100}\n"
            f"{summarize_label(label)}"
        )

        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, info)

    def flag(self, flag: str):
        row = self.current_row()
        raw_index = row["raw_index"]
        reason = self.reason_var.get() if flag == "bad" else ""

        self.flags[raw_index] = {
            "raw_index": raw_index,
            "review_index": self.index,
            "flag": flag,
            "reason": reason,
            "raw_size": list(row["raw_size"]),
            "parse_ok": row["parse_ok"],
        }

        self.save_flags()

        if self.index < len(self.rows) - 1:
            self.index += 1

        self.show_sample()

    def toggle_view(self):
        self.show_processed = not self.show_processed
        self.show_sample()

    def save_current_processed_image(self):
        row = self.current_row()
        raw_index = row["raw_index"]

        processed_image, _ = self.get_processed_image(row["image"])

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.processed_filename_for_raw_index(raw_index)
        processed_image.save(output_path)

        print(f"saved processed image: {output_path}")

        self.show_sample()

    def export_all_processed_images(self):
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        print(f"exporting processed images to: {self.processed_dir}")

        manifest = []

        for idx, row in enumerate(self.rows):
            raw_index = row["raw_index"]
            processed_image, pixel_values = self.get_processed_image(row["image"])

            output_path = self.processed_filename_for_raw_index(raw_index)
            processed_image.save(output_path)

            manifest.append(
                {
                    "review_index": idx,
                    "raw_index": raw_index,
                    "filename": output_path.name,
                    "raw_size": list(row["raw_size"]),
                    "processed_size": list(processed_image.size),
                    "pixel_shape": list(pixel_values.shape),
                    "parse_ok": row["parse_ok"],
                    "label": row["label"],
                }
            )

            if (idx + 1) % 50 == 0:
                print(f"exported {idx + 1}/{len(self.rows)}")

        manifest_path = self.processed_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"done exporting {len(self.rows)} images")
        print(f"manifest saved to: {manifest_path}")

        self.show_sample()

    def build_flags_from_missing_processed_images(self):
        if not self.processed_dir.exists():
            print(f"processed folder does not exist: {self.processed_dir}")
            return

        missing_count = 0
        kept_count = 0

        for idx, row in enumerate(self.rows):
            raw_index = row["raw_index"]
            image_path = self.processed_filename_for_raw_index(raw_index)

            if image_path.exists():
                kept_count += 1
                continue

            self.flags[raw_index] = {
                "raw_index": raw_index,
                "review_index": idx,
                "flag": "bad",
                "reason": "deleted_from_processed_folder",
                "raw_size": list(row["raw_size"]),
                "parse_ok": row["parse_ok"],
            }
            missing_count += 1

        self.save_flags()

        print(
            f"built flags from folder deletion: "
            f"{missing_count} missing/deleted -> bad, "
            f"{kept_count} existing -> unchanged"
        )
        print(f"flags saved to: {self.output_path}")

        self.show_sample()

    def next_sample(self):
        if self.index < len(self.rows) - 1:
            self.index += 1
            self.show_sample()

    def prev_sample(self):
        if self.index > 0:
            self.index -= 1
            self.show_sample()


def main():
    cfg = DEFAULT_CONFIG
    device = get_device()

    rows = build_review_rows(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.train_split,
        max_samples=MAX_REVIEW_SAMPLES,
        skip_unparseable_labels=SKIP_UNPARSEABLE_LABELS,
    )

    vision_encoder = DonutVisionEncoder(
        model_name=cfg.vision.model_name,
        device=device,
        default_processor=cfg.vision.default_processor,
        img_shape=(cfg.vision.image_height, cfg.vision.image_width),
        freeze=True,
    )

    print(f"review rows: {len(rows)}")
    print(f"flags path: {FLAGS_PATH}")
    print(f"processed image dir: {PROCESSED_IMAGE_DIR}")
    print(
        "controls: g=good, b=bad, u=unsure, "
        "v=toggle, s=save, S=export all, j=build flags, q=quit"
    )

    root = tk.Tk()
    RawCORDImageFlagger(
        root=root,
        rows=rows,
        output_path=FLAGS_PATH,
        vision_encoder=vision_encoder,
        processed_dir=PROCESSED_IMAGE_DIR,
    )
    root.mainloop()


if __name__ == "__main__":
    main()