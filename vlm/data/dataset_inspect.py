"""Print all (or N) samples from CORDDataset.

Run from the repo root:
    python print_samples.py
    python print_samples.py --split validation
    python print_samples.py --split train --n 20 --pretty
    python print_samples.py --max-samples 0   # 0 = no cap
"""

import argparse
import json

from vlm.data.dataset import CORDDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train",
                    choices=["train", "validation", "test"])
    ap.add_argument("--dataset", default="naver-clova-ix/cord-v2")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="cap passed to CORDDataset (0 = no cap)")
    ap.add_argument("--n", type=int, default=1000,
                    help="how many to print (0 = all kept samples)")
    ap.add_argument("--pretty", action="store_true",
                    help="pretty-print the JSON label")
    args = ap.parse_args()

    ds = CORDDataset(
        split=args.split,
        max_samples=None if args.max_samples == 0 else args.max_samples,
        dataset_name=args.dataset,
        prefer_high_resolution=False,
    )

    print(
        f"\n[{args.split}] loaded: {ds.num_loaded} | kept: {ds.num_after_filtering} | "
        f"parse_failed: {ds.num_parse_failed} | "
        f"empty_items: {ds.num_empty_items} | "
        f"missing_total: {ds.num_missing_total} | "
        f"too_long: {ds.num_too_long}\n"
    )

    total_to_print = len(ds) if args.n == 0 else min(args.n, len(ds))

    for i in range(total_to_print):
        sample = ds[i]
        image = sample["image"]
        label = sample["label"]

        print(f"===== sample {i} =====")
        print(f"image: {image.size[0]}x{image.size[1]} {image.mode}")

        if args.pretty:
            print(json.dumps(json.loads(label), indent=2, ensure_ascii=False))
        else:
            print(label)
        print()


if __name__ == "__main__":
    main()