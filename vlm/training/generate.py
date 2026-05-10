from copy import deepcopy
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

    generation_config = deepcopy(model.lm.model.generation_config)
    generation_config.max_length = None
    generation_config.max_new_tokens = max_completion_tokens
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.use_cache = True
    generation_config.return_dict_in_generate = True
    generation_config.do_sample = do_sample

    if do_sample:
        generation_config.temperature = temperature

    with torch.no_grad():
        inputs_embeds, full_attention_mask = model.prepare_inputs_embeds(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        output = model.lm.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            generation_config=generation_config,
        )

    texts = tokenizer.batch_decode(
        output.sequences,
        skip_special_tokens=True,
    )

    return GenerationOutput(texts=texts)