# #!/usr/bin/env python3
# """
# Print only ground_truth["gt_parse"] from a Hugging Face dataset.
#
# Usage:
#   python print_gt_parse.py --dataset your_org/your_dataset --split train --n 5
#   python print_gt_parse.py --dataset your_org/your_dataset --config config_name --split train --n 5
# """
#
# import argparse
# import json
# from datasets import load_dataset
#
#
# def maybe_json_load(value):
#     """Handle ground_truth stored as either a dict or a JSON string."""
#     if isinstance(value, str):
#         return json.loads(value)
#     return value
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", default='naver-clova-ix/cord-v2', help="HF dataset name or path")
#     parser.add_argument("--config", default=None, help="Optional dataset config")
#     parser.add_argument("--split", default="train", help="Dataset split")
#     parser.add_argument("--n", type=int, default=1000, help="Number of examples to print")
#     parser.add_argument("--column", default="ground_truth", help="Column containing ground truth JSON")
#     parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
#     args = parser.parse_args()
#
#     if args.config:
#         ds = load_dataset(args.dataset, args.config, split=args.split)
#     else:
#         ds = load_dataset(args.dataset, split=args.split)
#
#     for i, row in enumerate(ds.select(range(min(args.n, len(ds))))):
#         ground_truth = maybe_json_load(row[args.column])
#         gt_parse = ground_truth["gt_parse"]
#
#         print(f"\n===== Example {i} gt_parse =====")
#         if args.pretty:
#             print(json.dumps(gt_parse, indent=2, ensure_ascii=False))
#         else:
#             print(json.dumps(gt_parse, ensure_ascii=False))
#
#
# if __name__ == "__main__":
#     main()


import torch
from vlm.models.receipt_vlm import ReceiptVLM
from vlm.training.common import prepare_tokenizer, build_instruction
from vlm.data.dataset import CORDDataset

model = ReceiptVLM(device=torch.device("cpu"))   # or your usual device
tok = prepare_tokenizer(model.lm.tokenizer)
instruction = build_instruction(tok, "Extract the tabular data from this document and output it in JSON format.")
ds = CORDDataset(split="train", max_samples=1)

prompt = tok(instruction, return_tensors="pt", add_special_tokens=True)
inputs_embeds, attn = model.prepare_inputs_embeds(
    images=[ds[0]["image"]],
    input_ids=prompt["input_ids"].to(model.device),
    attention_mask=prompt["attention_mask"].to(model.device),
)
out = model.lm.model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=attn,
    max_length=inputs_embeds.shape[1] + 20,
    do_sample=False,
    pad_token_id=tok.pad_token_id,
    eos_token_id=tok.eos_token_id,
    return_dict_in_generate=True,
    use_cache=True,
)

print("sequences shape :", out.sequences.shape)
print("max_new_tokens  :", 20)
print("decoded         :", tok.decode(out.sequences[0], skip_special_tokens=False))