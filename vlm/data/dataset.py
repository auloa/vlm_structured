import json
from typing import Any, TypedDict, cast

from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def _to_str(value) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value).strip()
    if value is None:
        return ""
    return str(value).strip()


def parse_ground_truth(gt_string: str) -> dict:
    try:
        gt = json.loads(gt_string)
        gt_parse = gt["gt_parse"]

        line_items = [
            {
                "name": _to_str(item.get("nm")),
                "count": _to_str(item.get("cnt")),
                "price": _to_str(item.get("price")),
            }
            for item in gt_parse.get("menu", [])
            if isinstance(item, dict) and (item.get("nm") or item.get("price"))
        ]

        total = _to_str(gt_parse.get("total", {}).get("total_price"))

        return {
            "line_items": line_items,
            "total": total,
        }

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Could not parse ground truth string: {gt_string}") from exc


class CORDRow(TypedDict):
    image: Image.Image
    ground_truth: str


class CORDDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        max_samples: int | None = 400,
        dataset_name: str = "naver-clova-ix/cord-v2",
        prefer_high_resolution: bool = True,
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_target_length: int | None = None,
    ):

        raw = load_dataset(dataset_name, split=split)

        candidates: list[dict[str, Any]] = []

        self.num_loaded = len(raw)
        self.num_parse_failed = 0
        self.num_too_long = 0

        for idx in range(len(raw)):
            item = cast(CORDRow, raw[idx])
            image = item["image"].convert("RGB")
            try:
                parsed = parse_ground_truth(item["ground_truth"])
            except RuntimeError:
                self.num_parse_failed += 1
                continue
            label = json.dumps(parsed, ensure_ascii=False)
            if tokenizer is not None and max_target_length is not None:
                eos = tokenizer.eos_token or ""
                tokenized = tokenizer(
                    label + eos,
                    add_special_tokens=False,
                )
                target_len = len(tokenized["input_ids"])
                if target_len > max_target_length:
                    self.num_too_long += 1
                    continue
            else:
                target_len = None

            width, height = image.size
            area = width * height
            candidates.append(
                {
                    "image": image,
                    "label": label,
                    "area": area,
                    "width": width,
                    "height": height,
                    "target_len": target_len,
                }
            )

        if prefer_high_resolution:
            candidates.sort(key=lambda x: x["area"], reverse=True)

        if max_samples is not None:
            candidates = candidates[:max_samples]

        self.samples = [
            {
                "image": item["image"],
                "label": item["label"],
            }
            for item in candidates
        ]

        self.num_after_filtering = len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]