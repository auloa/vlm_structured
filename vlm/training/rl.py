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
    grad_clip_norm: float,
    kl_coef: float,
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

    assert completions_per_image >= 2, (
        "completions_per_image must be >= 2 because this RL loop uses "
        "group-relative advantages."
    )

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

    ckpt = torch.load(sft_checkpoint_path, map_location=device)
    model.projector.load_state_dict(ckpt["projector_state_dict"])
    print(f"loaded SFT checkpoint: {sft_checkpoint_path}")

    # Freeze base model explicitly.
    for p in model.vision_encoder.parameters():
        p.requires_grad_(False)

    for p in model.lm.model.parameters():
        p.requires_grad_(False)

    for p in model.projector.parameters():
        p.requires_grad_(True)

    # Frozen modules should stay in eval mode.
    model.vision_encoder.eval()
    model.lm.model.eval()

    assert_only_projector_trainable(model)

    # Frozen SFT projector used as KL/reference policy.
    ref_projector = clone_reference_projector(model.projector)

    print("loading dataset...")
    dataset = CORDDataset(
        split=train_split,
        max_samples=train_samples,
        dataset_name=dataset_name,
    )

    print(f"dataset ready: {len(dataset)} samples")

    tokenizer = model.lm.tokenizer

    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

            # ------------------------------------------------------------
            # 1. Generate K completions from the current policy.
            # ------------------------------------------------------------
            model.projector.eval()

            with torch.no_grad():
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

            # ------------------------------------------------------------
            # 2. Compute rewards for sampled completions.
            # ------------------------------------------------------------
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

            # ------------------------------------------------------------
            # 3. Group-relative advantages.
            # ------------------------------------------------------------
            reward_mean = rewards.mean()
            reward_std = rewards.std(unbiased=False)

            advantages = (rewards - reward_mean) / (reward_std + 1e-8)

            # If all completions receive the same reward, there is no
            # useful preference signal for this image. Skip the update.
            if torch.allclose(
                advantages,
                torch.zeros_like(advantages),
                atol=1e-6,
            ):
                global_step += 1

                if global_step % log_every == 0:
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

                if global_step % sample_every == 0:
                    _log_sample(
                        gen=gen,
                        rewards=rewards,
                        ground_truth=ground_truth,
                        global_step=global_step,
                        writer=writer,
                    )

                continue

            # ------------------------------------------------------------
            # 4. Score completions under frozen SFT reference projector.
            # ------------------------------------------------------------
            with torch.no_grad():
                ref_token_log_probs, token_mask = compute_completion_token_log_probs(
                    model=model,
                    image=image,
                    completions=gen.texts,
                    tokenizer=tokenizer,
                    instruction=instruction,
                    projector=ref_projector,
                    max_completion_length=max_completion_tokens,
                    require_grad=False,
                )

            # ------------------------------------------------------------
            # 5. Score completions under current projector with gradients.
            # ------------------------------------------------------------
            with get_autocast(device):
                policy_token_log_probs, token_mask = compute_completion_token_log_probs(
                    model=model,
                    image=image,
                    completions=gen.texts,
                    tokenizer=tokenizer,
                    instruction=instruction,
                    projector=model.projector,
                    max_completion_length=max_completion_tokens,
                    require_grad=True,
                )

                loss, policy_loss, kl_loss = compute_pg_kl_loss(
                    policy_token_log_probs=policy_token_log_probs,
                    ref_token_log_probs=ref_token_log_probs,
                    token_mask=token_mask,
                    advantages=advantages,
                    beta=kl_coef,
                )

            # ------------------------------------------------------------
            # 6. Projector-only optimization step.
            # ------------------------------------------------------------
            if torch.isnan(loss) or torch.isinf(loss) or loss.abs() > 100:
                print(
                    f"warning: invalid RL loss at step {global_step}; "
                    f"loss={loss.item() if torch.isfinite(loss) else loss}"
                )
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.projector.parameters(),
                max_norm=grad_clip_norm,
            )

            optimizer.step()
            global_step += 1

            # ------------------------------------------------------------
            # 7. Logging.
            # ------------------------------------------------------------
            if global_step % log_every == 0:
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
    """
    Encode image with frozen vision encoder, then apply selected projector.

    Gradients flow only through the projector when require_grad=True.
    """

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
    """
    Compute token log-probs for sampled completions.

    Scores all completion tokens, including the first generated token.

    Layout:
        inputs: visual + prompt + completion[:-1]
        targets: completion

    The final prompt position predicts completion[0].
    Completion positions predict the following completion tokens.
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
    completion_mask = tokens["attention_mask"].to(device).float()

    if completion_ids.shape[1] < 1:
        return (
            torch.zeros(k, 1, device=device),
            torch.zeros(k, 1, device=device),
        )

    # Feed all but the final completion token.
    # The final prompt token predicts completion_ids[:, 0].
    if completion_ids.shape[1] > 1:
        completion_input_ids = completion_ids[:, :-1]
        completion_input_mask = completion_mask[:, :-1].long()
    else:
        completion_input_ids = completion_ids[:, :0]
        completion_input_mask = completion_mask[:, :0].long()

    target_ids = completion_ids
    target_mask = completion_mask

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

        if completion_input_ids.shape[1] > 0:
            completion_embeds = model.lm.model.get_input_embeddings()(
                completion_input_ids
            ).float()
        else:
            completion_embeds = torch.empty(
                k,
                0,
                prompt_embeds.shape[-1],
                device=device,
                dtype=prompt_embeds.dtype,
            )

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
        [visual_mask, prompt_mask, completion_input_mask],
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

    visual_len = visual_embeds.shape[1]
    prompt_len = prompt_embeds.shape[1]

    # The token at index visual_len + prompt_len - 1 predicts the first
    # completion token.
    start = visual_len + prompt_len - 1
    end = start + target_ids.shape[1]

    completion_logits = logits[:, start:end, :]

    log_probs = F.log_softmax(completion_logits, dim=-1)

    token_log_probs = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    token_log_probs = token_log_probs * target_mask

    return token_log_probs, target_mask


def compute_pg_kl_loss(
    policy_token_log_probs,
    ref_token_log_probs,
    token_mask,
    advantages,
    beta: float,
):
    """
    Group-relative policy-gradient loss with KL regularization.

    Args:
        policy_token_log_probs: [K, T]
        ref_token_log_probs:    [K, T]
        token_mask:             [K, T]
        advantages:             [K]
        beta:                   KL coefficient

    The policy-gradient term increases the likelihood of above-average
    completions and decreases the likelihood of below-average completions.

    The KL term discourages the current projector from drifting too far from
    the frozen SFT projector.
    """

    advantages = advantages.detach()

    token_counts = token_mask.sum(dim=1).clamp_min(1.0)

    # Length-normalized sequence log-probs.
    policy_seq_log_probs = (
        policy_token_log_probs * token_mask
    ).sum(dim=1) / token_counts

    ref_seq_log_probs = (
        ref_token_log_probs * token_mask
    ).sum(dim=1) / token_counts

    # REINFORCE-style policy-gradient term.
    #
    # advantage > 0 -> increase log-prob
    # advantage < 0 -> decrease log-prob
    policy_loss = -(advantages * policy_seq_log_probs).mean()

    # Non-negative KL-style penalty.
    #
    # This is more stable than directly using:
    #   policy_seq_log_probs - ref_seq_log_probs
    #
    # because this expression is >= 0.
    log_ratio_ref = ref_seq_log_probs - policy_seq_log_probs
    kl_per_sample = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
    kl_loss = kl_per_sample.mean()

    loss = policy_loss + beta * kl_loss

    return loss, policy_loss, kl_loss


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

    writer.add_scalar(
        "rl/reward_json_like",
        sum(b.json_like for b in breakdowns) / max(1, len(breakdowns)),
        global_step,
    )
    writer.add_scalar(
        "rl/reward_line_items_populated",
        sum(b.line_items_populated for b in breakdowns) / max(1, len(breakdowns)),
        global_step,
    )
    writer.add_scalar(
        "rl/reward_content",
        sum(b.content for b in breakdowns) / max(1, len(breakdowns)),
        global_step,
    )
    writer.add_scalar(
        "rl/reward_anti_hallucination",
        sum(b.anti_hallucination for b in breakdowns) / max(1, len(breakdowns)),
        global_step,
    )
    writer.add_scalar(
        "rl/reward_total_match",
        sum(b.total_match for b in breakdowns) / max(1, len(breakdowns)),
        global_step,
    )

    if not skipped:
        writer.add_scalar("rl/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("rl/kl_loss", kl_loss.item(), global_step)
        writer.add_scalar("rl/loss", loss.item(), global_step)

        print(
            f"step {global_step:4d} | "
            f"batch {step + 1}/{total_steps} | "
            f"reward {mean_reward:.3f} | "
            f"json {json_valid_rate:.0%} | "
            f"schema {schema_rate:.0%} | "
            f"policy {policy_loss.item():.4f} | "
            f"kl {kl_loss.item():.4f} | "
            f"loss {loss.item():.4f}"
        )
    else:
        writer.add_scalar("rl/skipped_update", 1.0, global_step)

        print(
            f"step {global_step:4d} | "
            f"batch {step + 1}/{total_steps} | "
            f"reward {mean_reward:.3f} | "
            f"json {json_valid_rate:.0%} | "
            f"schema {schema_rate:.0%} | "
            f"skipped update: identical rewards"
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