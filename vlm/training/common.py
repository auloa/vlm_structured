from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from vlm.utils.training import assert_only_projector_trainable


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_tokenizer(tokenizer):
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def set_projector_only_trainable(model) -> None:
    """Freeze the base models and train only the projector."""
    for param in model.vision_encoder.parameters():
        param.requires_grad_(False)

    for param in model.lm.model.parameters():
        param.requires_grad_(False)

    for param in model.projector.parameters():
        param.requires_grad_(True)

    model.vision_encoder.eval()
    model.lm.model.eval()

    assert_only_projector_trainable(model)


def log_text(
    writer: SummaryWriter,
    tag: str,
    text: str,
    step: int | None = None,
) -> None:
    """Print a message and optionally mirror it to TensorBoard."""
    print(text)

    if step is not None:
        writer.add_text(tag, f"```text\n{text}\n```", step)