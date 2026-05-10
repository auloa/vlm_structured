import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from vlm.data.collator import CORDCollator
from vlm.data.dataset import CORDDataset
from vlm.models.receipt_vlm import ReceiptVLM
from vlm.utils.device import get_device
from vlm.utils.training import assert_only_projector_trainable, get_autocast


def train_sft(
    dataset_name: str,
    train_split: str,
    val_split: str,
    train_samples: int,
    val_samples: int,
    vision_model_name: str,
    image_height: int,
    image_width: int,
    lm_name: str,
    instruction: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_accum_steps: int,
    max_target_length: int,
    log_every: int,
    sample_every: int,
    run_dir: str | Path,
    checkpoint_dir: str | Path,
    best_checkpoint_path: str | Path,
):
    device = get_device()
    print(f"device: {device}")

    run_dir = Path(run_dir)
    checkpoint_dir = Path(checkpoint_dir)
    best_checkpoint_path = Path(best_checkpoint_path)

    print("loading model...")
    model = ReceiptVLM(
        device=device,
        vision_model_name=vision_model_name,
        image_height=image_height,
        image_width=image_width,
        lm_name=lm_name,
    )

    assert_only_projector_trainable(model)

    tokenizer = model.lm.tokenizer

    print("loading datasets...")
    train_dataset = CORDDataset(
        split=train_split,
        max_samples=train_samples,
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_target_length=max_target_length,
    )

    val_dataset = CORDDataset(
        split=val_split,
        max_samples=val_samples,
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_target_length=max_target_length,
    )

    print(
        f"train dataset: {len(train_dataset)} samples | "
        f"parse failed: {train_dataset.num_parse_failed} | "
        f"too long: {train_dataset.num_too_long}"
    )

    print(
        f"val dataset: {len(val_dataset)} samples | "
        f"parse failed: {val_dataset.num_parse_failed} | "
        f"too long: {val_dataset.num_too_long}"
    )

    collator = CORDCollator(
        tokenizer=tokenizer,
        instruction=instruction,
        max_target_length=max_target_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = AdamW(
        model.projector.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_steps = max(1, (len(train_loader) // grad_accum_steps) * epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    writer = SummaryWriter(log_dir=str(run_dir))
    os.makedirs(checkpoint_dir, exist_ok=True)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.projector.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        print(f"\nepoch {epoch + 1}/{epochs}")

        for step, batch in enumerate(train_loader):
            with get_autocast(device):
                output = model(
                    images=batch.images,
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    labels=batch.labels,
                )

            output_loss: torch.Tensor = output.loss
            loss = output_loss / grad_accum_steps
            loss.backward()

            epoch_loss += output_loss.item()

            should_step = (step + 1) % grad_accum_steps == 0
            is_last_step = step + 1 == len(train_loader)

            if should_step or is_last_step:
                torch.nn.utils.clip_grad_norm_(
                    model.projector.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % log_every == 0:
                    avg_loss = epoch_loss / (step + 1)
                    lr = scheduler.get_last_lr()[0]

                    writer.add_scalar("train/loss", output_loss.item(), global_step)
                    writer.add_scalar("train/avg_loss", avg_loss, global_step)
                    writer.add_scalar("train/lr", lr, global_step)

                    print(
                        f"step {global_step:4d} | "
                        f"batch {step + 1}/{len(train_loader)} | "
                        f"loss {output_loss.item():.4f} | "
                        f"avg {avg_loss:.4f} | "
                        f"lr {lr:.2e}"
                    )

                if global_step % sample_every == 0:
                    _log_sample(
                        model=model,
                        batch=batch,
                        tokenizer=tokenizer,
                        instruction=instruction,
                        device=device,
                        global_step=global_step,
                        writer=writer,
                        max_completion_tokens=128,
                    )

        val_loss = _validate(model, val_loader)
        writer.add_scalar("val/loss", val_loss, global_step)

        train_loss = epoch_loss / max(1, len(train_loader))
        print(
            f"epoch {epoch + 1} complete | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                val_loss=val_loss,
                checkpoint_path=best_checkpoint_path,
            )

            print(
                f"checkpoint saved: {best_checkpoint_path} "
                f"(val loss {val_loss:.4f})"
            )

    writer.close()
    print("SFT training complete")


def _validate(model, val_loader) -> float:
    model.projector.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            with get_autocast(model.device):
                output = model(
                    images=batch.images,
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    labels=batch.labels,
                )

            total_loss += output.loss.item()

    return total_loss / max(1, len(val_loader))


def _log_sample(
    model,
    batch,
    tokenizer,
    instruction: str,
    device,
    global_step,
    writer,
    max_completion_tokens: int,
):
    model.projector.eval()

    prompt_tokens = tokenizer(
        instruction,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = prompt_tokens["input_ids"].to(device)
    attention_mask = prompt_tokens["attention_mask"].to(device)

    with torch.no_grad():
        inputs_embeds, full_attention_mask = model.prepare_inputs_embeds(
            images=[batch.images[0]],
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        generated = model.lm.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_completion_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)

    writer.add_text("samples/output", decoded, global_step)
    print(f"\nsample @ step {global_step}:\n{decoded[:300]}")

    model.projector.train()


def _save_checkpoint(
    model,
    optimizer,
    epoch: int,
    step: int,
    val_loss: float,
    checkpoint_path: str | Path,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "projector_state_dict": model.projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )