import copy
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vlm.data.dataset import CORDDataset
from vlm.models.receipt_vlm import ReceiptVLM
from vlm.training.generate import generate_k_outputs
from vlm.training.rewards import compute_reward
from vlm.utils.device import get_device
from vlm.utils.training import assert_only_projector_trainable, get_autocast, set_seed


def train_rl(
    dataset_name: str,
    train_split: str,
    train_samples: int,
    vision_model_name: str,
    image_height: int,
    image_width: int,
    lm_name: str,
    instruction: str,
    epochs: int,
    completions_per_image: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    max_completion_tokens: int,
    kl_coef: float,
    clip_eps: float,
    log_every: int,
    sample_every: int,
    sft_checkpoint_path: str | Path,
    run_dir: str | Path,
    checkpoint_dir: str | Path,
    best_checkpoint_path: str | Path,
):
    device = get_device()
    print(f"device: {device}")

    set_seed(42)

    run_dir = Path(run_dir)
    checkpoint_dir = Path(checkpoint_dir)
    sft_checkpoint_path = Path(sft_checkpoint_path)
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

    ckpt = torch.load(sft_checkpoint_path, map_location=device)
    model.projector.load_state_dict(ckpt["projector_state_dict"])
    print(f"loaded SFT checkpoint: {sft_checkpoint_path}")

    # Frozen SFT projector used as the KL reference policy.
    ref_projector = clone_reference_projector(model.projector)

    print("loading dataset...")
    dataset = CORDDataset(
        split=train_split,
        max_samples=train_samples,
        dataset_name=dataset_name,

    )

    print(
        f"dataset ready: {len(dataset)} samples "
    )

    tokenizer = model.lm.tokenizer

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=_rl_collate,
    )

    optimizer = AdamW(
        model.projector.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    writer = SummaryWriter(log_dir=str(run_dir))
    os.makedirs(checkpoint_dir, exist_ok=True)

    global_step = 0
    best_mean_reward = -float("inf")

    for epoch in range(epochs):
        model.projector.train()
        epoch_rewards: list[float] = []

        print(f"\nepoch {epoch + 1}/{epochs}")

        for step, sample in enumerate(loader):
            image = sample["image"][0]
            ground_truth = sample["label"][0]

            # Generate multiple completions from the current policy.
            model.projector.eval()
            gen = generate_k_outputs(
                model=model,
                image=image,
                tokenizer=tokenizer,
                instruction=instruction,
                k=completions_per_image,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                do_sample=True,
            )
            model.projector.train()

            breakdowns = [compute_reward(text, ground_truth) for text in gen.texts]

            rewards = torch.tensor(
                [b.total for b in breakdowns],
                dtype=torch.float32,
                device=device,
            )

            epoch_rewards.extend(rewards.tolist())

            # Group-normalized advantage, GRPO-style.
            reward_std = rewards.std(unbiased=False)
            advantages = (rewards - rewards.mean()) / (reward_std + 1e-8)


            with torch.no_grad():
                old_token_log_probs, token_mask = compute_completion_token_log_probs(
                    model=model,
                    image=image,
                    completions=gen.texts,
                    tokenizer=tokenizer,
                    instruction=instruction,
                    projector=model.projector,
                    max_completion_length=max_completion_tokens,
                    require_grad=False,
                )

                ref_token_log_probs, _ = compute_completion_token_log_probs(
                    model=model,
                    image=image,
                    completions=gen.texts,
                    tokenizer=tokenizer,
                    instruction=instruction,
                    projector=ref_projector,
                    max_completion_length=max_completion_tokens,
                    require_grad=False,
                )

            # current_token_log_probs:
            #   current projector with gradient enabled.
            with get_autocast(device):
                current_token_log_probs, token_mask = compute_completion_token_log_probs(
                    model=model,
                    image=image,
                    completions=gen.texts,
                    tokenizer=tokenizer,
                    instruction=instruction,
                    projector=model.projector,
                    max_completion_length=max_completion_tokens,
                    require_grad=True,
                )

                loss, policy_loss, kl_loss, clip_fraction = compute_grpo_loss(
                    current_token_log_probs=current_token_log_probs,
                    old_token_log_probs=old_token_log_probs,
                    ref_token_log_probs=ref_token_log_probs,
                    token_mask=token_mask,
                    advantages=advantages,
                    clip_eps=clip_eps,
                    beta=kl_coef,
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.projector.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            global_step += 1

            if global_step % log_every == 0:
                _log_metrics(
                    writer=writer,
                    global_step=global_step,
                    rewards=rewards,
                    breakdowns=breakdowns,
                    policy_loss=policy_loss,
                    kl_loss=kl_loss,
                    clip_fraction=clip_fraction,
                    loss=loss,
                    step=step,
                    total_steps=len(loader),
                )

            if global_step % sample_every == 0:
                _log_sample(
                    gen=gen,
                    rewards=rewards,
                    ground_truth=ground_truth,
                    global_step=global_step,
                    writer=writer,
                )

        mean_epoch_reward = sum(epoch_rewards) / max(1, len(epoch_rewards))
        writer.add_scalar("rl/epoch_mean_reward", mean_epoch_reward, epoch)

        print(
            f"epoch {epoch + 1} complete | "
            f"mean reward {mean_epoch_reward:.3f}"
        )

        if mean_epoch_reward > best_mean_reward:
            best_mean_reward = mean_epoch_reward

            _save_rl_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                mean_reward=mean_epoch_reward,
                checkpoint_path=best_checkpoint_path,
                completions_per_image=completions_per_image,
                clip_eps=clip_eps,
                kl_coef=kl_coef,
            )

            print(
                f"checkpoint saved: {best_checkpoint_path} "
                f"(mean reward {mean_epoch_reward:.3f})"
            )

    writer.close()
    print("RL training complete")


def clone_reference_projector(projector):
    ref_projector = copy.deepcopy(projector)
    ref_projector.to(next(projector.parameters()).device)
    ref_projector.eval()
    ref_projector.requires_grad_(False)
    return ref_projector


def get_visual_embeddings_with_projector(
    model: ReceiptVLM,
    image,
    projector,
    require_grad: bool,
) -> torch.Tensor:
    """Encode image with frozen vision encoder, then apply selected projector."""

    with torch.no_grad():
        visual_features = model.vision_encoder([image])

    if require_grad:
        visual_embeddings = projector(visual_features)
    else:
        with torch.no_grad():
            visual_embeddings = projector(visual_features)

    return visual_embeddings.float()


def compute_completion_token_log_probs(
    model: ReceiptVLM,
    image,
    completions: list[str],
    tokenizer,
    instruction: str,
    projector=None,
    max_completion_length: int = 128,
    require_grad: bool = True,
):
    """Compute token log-probs for sampled completions.

    This scores generated completions under:
        visual prefix + instruction prefix + completion prefix

    The projector can be either:
        - current trainable projector
        - frozen reference projector
    """

    device = model.device
    projector = projector or model.projector
    k = len(completions)

    tokens = tokenizer(
        completions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_completion_length,
        add_special_tokens=False,
    )

    completion_ids = tokens["input_ids"].to(device)
    completion_mask = tokens["attention_mask"].to(device)

    if completion_ids.shape[1] < 2:
        return (
            torch.zeros(k, 1, device=device),
            torch.zeros(k, 1, device=device),
        )

    # Teacher forcing:
    # input predicts the next token.
    input_ids = completion_ids[:, :-1]
    target_ids = completion_ids[:, 1:]
    target_mask = completion_mask[:, 1:].float()

    visual_embeds = get_visual_embeddings_with_projector(
        model=model,
        image=image,
        projector=projector,
        require_grad=require_grad,
    )

    visual_embeds = visual_embeds.expand(k, -1, -1)

    prompt_tokens = tokenizer(
        instruction,
        return_tensors="pt",
        add_special_tokens=True,
    )

    prompt_ids = prompt_tokens["input_ids"].to(device)
    prompt_mask = prompt_tokens["attention_mask"].to(device)

    prompt_ids = prompt_ids.expand(k, -1)
    prompt_mask = prompt_mask.expand(k, -1)

    with torch.no_grad():
        prompt_embeds = model.lm.model.get_input_embeddings()(prompt_ids).float()
        completion_embeds = model.lm.model.get_input_embeddings()(input_ids).float()

    inputs_embeds = torch.cat(
        [visual_embeds, prompt_embeds, completion_embeds],
        dim=1,
    )

    inputs_embeds = inputs_embeds.to(dtype=model.lm.model_dtype)

    visual_mask = torch.ones(
        k,
        visual_embeds.shape[1],
        device=device,
        dtype=torch.long,
    )

    attention_mask = torch.cat(
        [visual_mask, prompt_mask, completion_mask[:, :-1]],
        dim=1,
    )

    if require_grad:
        outputs = model.lm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
    else:
        with torch.no_grad():
            outputs = model.lm.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

    logits = outputs.logits

    prompt_len = visual_embeds.shape[1] + prompt_embeds.shape[1]

    completion_logits = logits[
        :,
        prompt_len : prompt_len + target_ids.shape[1],
        :,
    ]

    log_probs = F.log_softmax(completion_logits, dim=-1)

    token_log_probs = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    token_log_probs = token_log_probs * target_mask

    return token_log_probs, target_mask


def compute_grpo_loss(
    current_token_log_probs,
    old_token_log_probs,
    ref_token_log_probs,
    token_mask,
    advantages,
    clip_eps: float,
    beta: float,
):
    """GRPO/PPO-style clipped policy loss with KL to SFT reference projector."""

    advantages = advantages.detach().unsqueeze(1)

    log_ratio = current_token_log_probs - old_token_log_probs
    ratio = torch.exp(log_ratio)

    unclipped = ratio * advantages
    clipped = torch.clamp(
        ratio,
        1.0 - clip_eps,
        1.0 + clip_eps,
    ) * advantages

    policy_objective = torch.minimum(unclipped, clipped)

    policy_loss = -(
        policy_objective * token_mask
    ).sum() / token_mask.sum().clamp_min(1.0)

    # Simple token-level KL proxy against the frozen SFT reference projector.
    kl = (current_token_log_probs - ref_token_log_probs) * token_mask
    kl_loss = kl.sum() / token_mask.sum().clamp_min(1.0)

    loss = policy_loss + beta * kl_loss

    clip_fraction = (
        ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float()
        * token_mask
    ).sum() / token_mask.sum().clamp_min(1.0)

    return loss, policy_loss, kl_loss, clip_fraction


def _log_metrics(
    writer,
    global_step,
    rewards,
    breakdowns,
    policy_loss,
    kl_loss,
    clip_fraction,
    loss,
    step,
    total_steps,
):
    mean_reward = rewards.mean().item()

    json_valid_rate = sum(
        b.json_valid > 0 for b in breakdowns
    ) / max(1, len(breakdowns))

    schema_rate = sum(
        b.schema > 0.1 for b in breakdowns
    ) / max(1, len(breakdowns))

    writer.add_scalar("rl/mean_reward", mean_reward, global_step)
    writer.add_scalar("rl/json_valid_rate", json_valid_rate, global_step)
    writer.add_scalar("rl/schema_rate", schema_rate, global_step)
    writer.add_scalar("rl/policy_loss", policy_loss.item(), global_step)
    writer.add_scalar("rl/kl_loss", kl_loss.item(), global_step)
    writer.add_scalar("rl/clip_fraction", clip_fraction.item(), global_step)
    writer.add_scalar("rl/loss", loss.item(), global_step)

    print(
        f"step {global_step:4d} | "
        f"batch {step + 1}/{total_steps} | "
        f"reward {mean_reward:.3f} | "
        f"json {json_valid_rate:.0%} | "
        f"schema {schema_rate:.0%} | "
        f"policy {policy_loss.item():.4f} | "
        f"kl {kl_loss.item():.4f} | "
        f"clip {clip_fraction.item():.2f}"
    )


def _log_sample(gen, rewards, ground_truth, global_step, writer):
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
    clip_eps: float,
    kl_coef: float,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "mean_reward": mean_reward,
            "projector_state_dict": model.projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "algorithm": "grpo_projector_only",
            "completions_per_image": completions_per_image,
            "clip_eps": clip_eps,
            "kl_coef": kl_coef,
        },
        checkpoint_path,
    )


def _rl_collate(batch):
    return {
        "image": [sample["image"] for sample in batch],
        "label": [sample["label"] for sample in batch],
    }