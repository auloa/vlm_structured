"""
Quick sanity-check script for DonutVisionEncoder.
Paste your DonutVisionEncoder class here or import it.

Usage:
    python check_encoder.py
    python check_encoder.py --device cuda
    python check_encoder.py --img-shape 1280 960
"""

import argparse
import time

import torch
from PIL import Image, ImageDraw

# ── import or paste your encoder ─────────────────────────────────────────────
from vlm.models.vision_encoder import DonutVisionEncoder


# ── helpers ───────────────────────────────────────────────────────────────────

def make_dummy_image(w: int, h: int, text: str = "") -> Image.Image:
    """RGB image with some structure so it's not a blank canvas."""
    img = Image.new("RGB", (w, h), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)
    # fake receipt-like lines
    for i, y in enumerate(range(40, h - 40, 30)):
        draw.rectangle([30, y, w - 30, y + 18], fill=(220, 220, 215))
    if text:
        draw.text((10, 10), text, fill=(0, 0, 0))
    return img


def section(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


# ── checks ────────────────────────────────────────────────────────────────────

def check_output_shape(encoder, img_shape):
    section("1. Output shape")
    w, h = img_shape[1], img_shape[0]
    img = make_dummy_image(w, h, "shape check")
    with torch.no_grad():
        out = encoder([img])
    B, N, D = out.shape
    print(f"  input image : {h}h × {w}w")
    print(f"  output      : [B={B}, N={N} tokens, D={D} dim]")
    print(f"  hidden_size : {encoder.hidden_size}")
    assert D == encoder.hidden_size, "hidden_size mismatch!"
    return N, D


def check_frozen(encoder):
    section("2. Frozen parameters")
    total = sum(p.numel() for p in encoder.model.parameters())
    trainable = sum(p.numel() for p in encoder.model.parameters() if p.requires_grad)
    print(f"  total params     : {total:,}")
    print(f"  trainable params : {trainable:,}")
    print(f"  frozen           : {'✓' if trainable == 0 else '✗ some params are not frozen!'}")


def check_dtype(encoder):
    section("3. Dtype")
    param = next(encoder.model.parameters())
    print(f"  expected : {encoder.model_dtype}")
    print(f"  actual   : {param.dtype}")
    print(f"  match    : {'✓' if param.dtype == encoder.model_dtype else '✗'}")


def check_batch(encoder, img_shape):
    section("4. Batch consistency")
    w, h = img_shape[1], img_shape[0]
    imgs = [make_dummy_image(w, h, f"img {i}") for i in range(3)]
    with torch.no_grad():
        out_batch = encoder(imgs)
        out_single = encoder([imgs[0]])
    print(f"  batch  shape : {list(out_batch.shape)}")
    print(f"  single shape : {list(out_single.shape)}")
    # first item in batch should match single forward
    diff = (out_batch[0] - out_single[0]).abs().max().item()
    print(f"  max diff (item 0 vs single) : {diff:.2e}  {'✓' if diff < 1e-3 else '✗ mismatch!'}")


def check_train_mode_lock(encoder):
    section("5. Eval mode survives .train()")
    encoder.train()
    is_eval = not encoder.model.training
    print(f"  encoder.model.training after .train() : {encoder.model.training}")
    print(f"  stays in eval : {'✓' if is_eval else '✗ override train() not implemented!'}")
    encoder.eval()


def check_speed(encoder, img_shape, n: int = 5):
    section(f"6. Speed ({n} forward passes, single image)")
    w, h = img_shape[1], img_shape[0]
    img = make_dummy_image(w, h)
    # warmup
    with torch.no_grad():
        encoder([img])
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        with torch.no_grad():
            encoder([img])
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    print(f"  avg : {avg*1000:.1f} ms  |  min : {min(times)*1000:.1f} ms  |  max : {max(times)*1000:.1f} ms")


def check_default_vs_custom_processor(device, img_shape):
    section("7. Default vs custom processor output shape")
    w, h = img_shape[1], img_shape[0]
    img = make_dummy_image(w, h)

    enc_default = DonutVisionEncoder(device=device, default_processor=True)
    enc_custom  = DonutVisionEncoder(device=device, default_processor=False, img_shape=img_shape)

    with torch.no_grad():
        out_d = enc_default([img])
        out_c = enc_custom([img])

    print(f"  default processor output : {list(out_d.shape)}")
    print(f"  custom  processor output : {list(out_c.shape)}")
    same = out_d.shape == out_c.shape
    print(f"  same shape : {'✓' if same else '✗ different — custom size is working'}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--img-shape", nargs=2, type=int, default=[640, 960],
                        metavar=("HEIGHT", "WIDTH"))
    args = parser.parse_args()

    device = torch.device(args.device)
    img_shape = tuple(args.img_shape)

    print(f"\nDevice: {device} | Image shape: {img_shape[0]}h × {img_shape[1]}w")

    encoder = DonutVisionEncoder(device=device, img_shape=img_shape, freeze=True)

    check_output_shape(encoder, img_shape)
    check_frozen(encoder)
    check_dtype(encoder)
    check_batch(encoder, img_shape)
    check_train_mode_lock(encoder)
    check_speed(encoder, img_shape)
    check_default_vs_custom_processor(device, img_shape)

    print("\n✓ All checks done\n")


if __name__ == "__main__":
    main()