import copy

import torch
import torch.nn.functional as F
from vlm.models.receipt_vlm import ReceiptVLM


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
    """Encode image with frozen vision encoder, then apply selected projector.

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
    """Compute token log-probs for sampled completions.

    Layout:
        inputs:  visual + prompt + completion[:-1]
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
    ).to(dtype=model.lm.model_dtype)

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
    """Group-relative policy-gradient loss with KL regularization."""
    advantages = advantages.detach()

    token_counts = token_mask.sum(dim=1).clamp_min(1.0)

    policy_seq_log_probs = (
        policy_token_log_probs * token_mask
    ).sum(dim=1) / token_counts

    ref_seq_log_probs = (
        ref_token_log_probs * token_mask
    ).sum(dim=1) / token_counts

    policy_loss = -(advantages * policy_seq_log_probs).mean()

    log_ratio_ref = ref_seq_log_probs - policy_seq_log_probs
    kl_per_sample = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
    kl_loss = kl_per_sample.mean()

    loss = policy_loss + beta * kl_loss

    return loss, policy_loss, kl_loss