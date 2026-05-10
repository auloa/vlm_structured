from dataclasses import dataclass

import torch
from PIL import Image

from vlm.models.receipt_vlm import ReceiptVLM


@dataclass
class GenerationOutput:
    texts: list[str]


def generate_k_outputs(
    model: ReceiptVLM,
    image: Image.Image,
    tokenizer,
    instruction: str,
    k: int = 4,
    max_completion_tokens: int = 128,
    temperature: float = 0.8,
    do_sample: bool = True,
) -> GenerationOutput:
    device = model.device
    model.eval()

    prompt_tokens = tokenizer(
        instruction,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = prompt_tokens["input_ids"].to(device)
    attention_mask = prompt_tokens["attention_mask"].to(device)

    if k > 1:
        input_ids = input_ids.expand(k, -1)
        attention_mask = attention_mask.expand(k, -1)

    images = [image] * k

    generate_kwargs = {
        "inputs_embeds": None,
        "attention_mask": None,
        "max_new_tokens": max_completion_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "return_dict_in_generate": True,
    }

    if do_sample:
        generate_kwargs["temperature"] = temperature

    with torch.no_grad():
        inputs_embeds, full_attention_mask = model.prepare_inputs_embeds(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        generate_kwargs["inputs_embeds"] = inputs_embeds
        generate_kwargs["attention_mask"] = full_attention_mask

        output = model.lm.model.generate(**generate_kwargs)

    texts = tokenizer.batch_decode(
        output.sequences,
        skip_special_tokens=True,
    )

    return GenerationOutput(texts=texts)