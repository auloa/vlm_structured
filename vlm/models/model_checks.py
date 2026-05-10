from vlm.models.receipt_vlm import ReceiptVLM
from vlm.utils.device import get_device


def check_model():
    model = ReceiptVLM(device=get_device())

    total = 0
    trainable = 0

    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            print(f"[TRAINABLE] {name} | {p.numel()}")

    print("\n--- SUMMARY ---")
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")

    assert trainable > 0
    assert all("projector" in n for n, p in model.named_parameters() if p.requires_grad)

    print("✅ Only projector is trainable")

if __name__ == "__main__":
    check_model()