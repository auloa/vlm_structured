from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from vlm.configs.training_schema import TrainingConfig
from vlm.data.dataset import CORDDataset
from vlm.models.receipt_vlm import ReceiptVLM
from vlm.training.common import (
    ensure_dir,
    log_text,
    prepare_tokenizer,
    set_projector_only_trainable,
)
from vlm.training.generate import generate_k_outputs
from vlm.training.rewards import compute_reward
from vlm.training.rl_utils import (
    clone_reference_projector,
    compute_completion_token_log_probs,
    compute_pg_kl_loss,
)
from vlm.utils.device import get_device
from vlm.utils.training import get_autocast, set_seed


def train_rl(cfg: TrainingConfig) -> None:
    data = cfg.data
    vision = cfg.vision
    model_cfg = cfg.model
    rl = cfg.rl

    device = get_device()
    set_seed(42)

    assert rl.completions_per_image >= 2, (
        "completions_per_image must be >= 2 because this RL loop uses "
        "group-relative advantages."
    )

    ensure_dir(cfg.rl_checkpoint_dir)

    print(f"device: {device}")
    print("loading model...")

    model = ReceiptVLM(
        device=device,
        vision_model_name=vision.model_name,
        default_vision_processor=vision.default_processor,
        image_height=vision.image_height,
        image_width=vision.image_width,
        lm_name=model_cfg.lm_name,
    )

    ckpt = torch.load(cfg.sft_best_checkpoint, map_location=device)
    model.projector.load_state_dict(ckpt["projector_state_dict"])
    print(f"loaded SFT checkpoint: {cfg.sft_best_checkpoint}")

    set_projector_only_trainable(model)

    # Frozen SFT projector used as KL/reference policy.
    ref_projector = clone_reference_projector(model.projector)

    print("loading dataset...")

    dataset = CORDDataset(
        split=data.train_split,
        max_samples=data.train_samples,
        dataset_name=data.dataset_name,
    )

    print(f"dataset ready: {len(dataset)} samples")

    tokenizer = prepare_tokenizer(model.lm.tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=_rl_collate,
    )

    optimizer = AdamW(
        model.projector.parameters(),
        lr=rl.learning_rate,
        weight_decay=rl.weight_decay,
    )

    global_step = 0
    best_mean_reward = -float("inf")

    with SummaryWriter(log_dir=str(cfg.rl_run_dir)) as writer:
        log_text(writer, "rl/log", f"device: {device}", step=0)

        for epoch in range(rl.epochs):
            model.projector.train()
            epoch_rewards: list[float] = []

            print(f"\nepoch {epoch + 1}/{rl.epochs}")

            for step, sample in enumerate(loader):
                image = sample["image"][0]
                ground_truth = sample["label"][0]

                # 1. Generate K completions from the current policy.
                model.projector.eval()

                with torch.no_grad():
                    gen = generate_k_outputs(
                        model=model,
                        image=image,
                        tokenizer=tokenizer,
                        instruction=model_cfg.instruction,
                        k=rl.completions_per_image,
                        max_completion_tokens=rl.max_completion_tokens,
                        temperature=rl.temperature,
                        do_sample=True,
                    )

                model.projector.train()

                # 2. Compute rewards.
                breakdowns = [
                    compute_reward(text, ground_truth)
                    for text in gen.texts
                ]

                rewards = torch.tensor(
                    [b.total for b in breakdowns],
                    dtype=torch.float32,
                    device=device,
                )

                epoch_rewards.extend(rewards.tolist())

                # 3. Group-relative advantages.
                reward_mean = rewards.mean()
                reward_std = rewards.std(unbiased=False)
                advantages = (rewards - reward_mean) / (reward_std + 1e-8)

                # If all completions receive the same reward, there is no
                # useful preference signal for this image.
                if torch.allclose(
                    advantages,
                    torch.zeros_like(advantages),
                    atol=1e-6,
                ):
                    global_step += 1

                    if global_step % rl.log_every == 0:
                        _log_metrics(
                            writer=writer,
                            global_step=global_step,
                            rewards=rewards,
                            breakdowns=breakdowns,
                            policy_loss=None,
                            kl_loss=None,
                            loss=None,
                            step=step,
                            total_steps=len(loader),
                            skipped=True,
                        )

                    if global_step % rl.sample_every == 0:
                        _log_sample(
                            gen=gen,
                            rewards=rewards,
                            ground_truth=ground_truth,
                            global_step=global_step,
                            writer=writer,
                        )

                    continue

                # 4. Score completions under frozen SFT reference projector.
                with torch.no_grad():
                    ref_token_log_probs, token_mask = compute_completion_token_log_probs(
                        model=model,
                        image=image,
                        completions=gen.texts,
                        tokenizer=tokenizer,
                        instruction=model_cfg.instruction,
                        projector=ref_projector,
                        max_completion_length=rl.max_completion_tokens,
                        require_grad=False,
                    )

                # 5. Score completions under current projector with gradients.
                with get_autocast(device):
                    policy_token_log_probs, token_mask = compute_completion_token_log_probs(
                        model=model,
                        image=image,
                        completions=gen.texts,
                        tokenizer=tokenizer,
                        instruction=model_cfg.instruction,
                        projector=model.projector,
                        max_completion_length=rl.max_completion_tokens,
                        require_grad=True,
                    )

                    loss, policy_loss, kl_loss = compute_pg_kl_loss(
                        policy_token_log_probs=policy_token_log_probs,
                        ref_token_log_probs=ref_token_log_probs,
                        token_mask=token_mask,
                        advantages=advantages,
                        beta=rl.kl_coef,
                    )

                # 6. Projector-only optimization step.
                if not torch.isfinite(loss) or loss.abs() > 100:
                    print(
                        f"warning: invalid RL loss at step {global_step}; "
                        f"loss={loss.item() if torch.isfinite(loss) else loss}"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.projector.parameters(),
                    max_norm=rl.grad_clip_norm,
                )

                optimizer.step()
                global_step += 1

                # 7. Logging.
                if global_step % rl.log_every == 0:
                    _log_metrics(
                        writer=writer,
                        global_step=global_step,
                        rewards=rewards,
                        breakdowns=breakdowns,
                        policy_loss=policy_loss,
                        kl_loss=kl_loss,
                        loss=loss,
                        step=step,
                        total_steps=len(loader),
                        skipped=False,
                    )

                    writer.add_scalar(
                        "rl/optim/grad_norm",
                        float(grad_norm),
                        global_step,
                    )

                if global_step % rl.sample_every == 0:
                    _log_sample(
                        gen=gen,
                        rewards=rewards,
                        ground_truth=ground_truth,
                        global_step=global_step,
                        writer=writer,
                    )

            mean_epoch_reward = sum(epoch_rewards) / max(1, len(epoch_rewards))

            writer.add_scalar(
                "rl/epoch_mean_reward",
                mean_epoch_reward,
                epoch + 1,
            )

            epoch_msg = (
                f"epoch {epoch + 1} complete | "
                f"mean reward {mean_epoch_reward:.3f}"
            )
            log_text(writer, "rl/log", epoch_msg, step=global_step)

            if mean_epoch_reward > best_mean_reward:
                best_mean_reward = mean_epoch_reward

                _save_rl_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                    mean_reward=mean_epoch_reward,
                    checkpoint_path=cfg.rl_best_checkpoint,
                    completions_per_image=rl.completions_per_image,
                    kl_coef=rl.kl_coef,
                )

                log_text(
                    writer,
                    "rl/log",
                    (
                        f"checkpoint saved: {cfg.rl_best_checkpoint} "
                        f"(mean reward {mean_epoch_reward:.3f})"
                    ),
                    step=global_step,
                )

    print("RL training complete")


def _log_metrics(
    writer,
    global_step,
    rewards,
    breakdowns,
    policy_loss,
    kl_loss,
    loss,
    step,
    total_steps,
    skipped: bool = False,
) -> None:
    mean_reward = rewards.mean().item()

    mean_format = sum(b.format for b in breakdowns) / max(1, len(breakdowns))
    mean_schema = sum(b.schema for b in breakdowns) / max(1, len(breakdowns))
    mean_content = sum(b.content for b in breakdowns) / max(1, len(breakdowns))
    mean_hallucination = sum(b.hallucination for b in breakdowns) / max(
        1, len(breakdowns)
    )

    clean_json_rate = sum(
        b.format >= 0.30 for b in breakdowns
    ) / max(1, len(breakdowns))

    parseable_or_wrapped_json_rate = sum(
        b.format >= 0.20 for b in breakdowns
    ) / max(1, len(breakdowns))

    schema_rate = sum(
        b.schema >= 0.30 for b in breakdowns
    ) / max(1, len(breakdowns))

    writer.add_scalar("rl/reward/mean", mean_reward, global_step)
    writer.add_scalar("rl/reward/clean_json_rate", clean_json_rate, global_step)
    writer.add_scalar(
        "rl/reward/parseable_or_wrapped_json_rate",
        parseable_or_wrapped_json_rate,
        global_step,
    )
    writer.add_scalar("rl/reward/schema_rate", schema_rate, global_step)

    writer.add_scalar("rl/reward_components/format", mean_format, global_step)
    writer.add_scalar("rl/reward_components/schema", mean_schema, global_step)
    writer.add_scalar("rl/reward_components/content", mean_content, global_step)
    writer.add_scalar(
        "rl/reward_components/hallucination",
        mean_hallucination,
        global_step,
    )

    if skipped:
        writer.add_scalar("rl/update/skipped", 1.0, global_step)

        print(
            f"step {global_step:4d} | "
            f"batch {step + 1}/{total_steps} | "
            f"reward {mean_reward:.3f} | "
            f"json {clean_json_rate:.0%} | "
            f"schema {schema_rate:.0%} | "
            f"skipped update: identical rewards"
        )
        return

    writer.add_scalar("rl/update/skipped", 0.0, global_step)
    writer.add_scalar("rl/loss/policy", policy_loss.item(), global_step)
    writer.add_scalar("rl/loss/kl", kl_loss.item(), global_step)
    writer.add_scalar("rl/loss/total", loss.item(), global_step)

    print(
        f"step {global_step:4d} | "
        f"batch {step + 1}/{total_steps} | "
        f"reward {mean_reward:.3f} | "
        f"json {clean_json_rate:.0%} | "
        f"schema {schema_rate:.0%} | "
        f"format {mean_format:.3f} | "
        f"content {mean_content:.3f} | "
        f"hallucination {mean_hallucination:.3f} | "
        f"policy {policy_loss.item():.4f} | "
        f"kl {kl_loss.item():.4f} | "
        f"loss {loss.item():.4f}"
    )

def _log_sample(gen, rewards, ground_truth, global_step, writer) -> None:
    best_idx = rewards.argmax().item()
    worst_idx = rewards.argmin().item()

    log_text = (
        f"**ground truth:** {ground_truth[:300]}\n\n"
        f"**best reward={rewards[best_idx]:.3f}:**\n"
        f"{gen.texts[best_idx][:500]}\n\n"
        f"**worst reward={rewards[worst_idx]:.3f}:**\n"
        f"{gen.texts[worst_idx][:500]}"
    )

    writer.add_text("rl/samples", log_text, global_step)

    print(
        f"\nbest @ step {global_step} "
        f"(reward={rewards[best_idx]:.3f}):\n"
        f"{gen.texts[best_idx][:300]}"
    )


def _save_rl_checkpoint(
    model,
    optimizer,
    epoch: int,
    step: int,
    mean_reward: float,
    checkpoint_path: str | Path,
    completions_per_image: int,
    kl_coef: float,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "mean_reward": mean_reward,
            "projector_state_dict": model.projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "algorithm": "group_relative_policy_gradient_projector_only",
            "completions_per_image": completions_per_image,
            "kl_coef": kl_coef,
        },
        checkpoint_path,
    )


def _rl_collate(batch):
    return {
        "image": [sample["image"] for sample in batch],
        "label": [sample["label"] for sample in batch],
    }