# vlm — Receipt Document VLM with RL Alignment

A vision-language model that extracts tabular data from scanned receipts and emits it as strict JSON. A frozen Donut vision encoder is connected to a frozen TinyLlama language model through a trainable projection module. The projector is trained in two stages: supervised fine-tuning (SFT) for basic extraction, then reinforcement learning (RL) to enforce strict JSON formatting and schema compliance.

Built for the Document AI Alignment assignment. Target hardware is a single consumer GPU (RTX 3080/3090/4080 class, 8–16 GB).


## Quick Start

```bash
# Setup
uv sync
python -m vlm.utils.hf_login   # one-time HF auth for the model + dataset

# Train
python -m vlm.scripts.train_sft --config receipt-base
python -m vlm.scripts.train_rl  --config receipt-base

# Evaluate (runs SFT and RL checkpoints, writes a comparison)
python -m vlm.scripts.evaluate  --config receipt-base

# Monitor
tensorboard --logdir runs/
```

Use `--config debug` for a 20-sample smoke test (a few minutes end-to-end).

**Requirements:** Python 3.10+, single CUDA-capable GPU recommended (CPU works for the debug config).


## Architecture

The model bypasses the LM's input embedding lookup for visual tokens and feeds the LM an `inputs_embeds` tensor that prepends projected visual tokens to text token embeddings.

```
  image ──► Donut encoder ──► visual features  (B, Nv, 1024)
                                    │
                                    ▼
                          Projector (MLP + LayerNorm)
                                    │
                                    ▼  visual tokens (B, Nv, 2048)
                                                                  ┐
                                                                  │
                                                                  ├──► concat ──► TinyLlama ──► JSON
                                                                  │
  prompt ──► chat template ──► tokenize ──► LM embed ──► text tokens (B, Nt, 2048)
                                                                  ┘

  FROZEN:    Donut encoder (~110M), TinyLlama (~1.1B)
  TRAINABLE: Projector (~6M params, ≈0.5% of total)
```

### Vision encoder: Donut

`naver-clova-ix/donut-base-finetuned-cord-v2`. Donut is a Swin-Transformer encoder pretrained on document images with a text-decoding objective, which gives it dense, text-aware visual features without needing OCR as a separate step. We use the encoder only; the decoder is discarded.

Compared to general-purpose vision encoders (CLIP, DINO, SigLIP), Donut better captures the small dense text typical of receipts.

Input is resized to 640×960 with `do_align_long_axis=False`, producing roughly 1200 visual tokens per image at hidden size 1024.

### Projector

```python
nn.Sequential(
    nn.Linear(1024, 4096),   # 2× LM embedding dim
    nn.GELU(),
    nn.Linear(4096, 2048),   # TinyLlama embedding dim
    nn.LayerNorm(2048),
)
```

A two-layer MLP with a wider middle layer and a final LayerNorm. The LayerNorm at the output is important: it constrains the magnitude of injected visual tokens to a scale compatible with the frozen LM's learned embedding distribution. Without it the LM tends to ignore the visual prefix or produce degenerate outputs.

### Language model: TinyLlama-1.1B-Chat

`TinyLlama/TinyLlama-1.1B-Chat-v1.0`. Small enough to fit on consumer GPUs in bf16, instruction-tuned so it follows the prompt format, and standard enough that the chat template is well-defined.

The instruction is wrapped at runtime via the tokenizer's `apply_chat_template`:

```python
instruction = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Extract the tabular data from this document and output it in JSON format."}],
    tokenize=False,
    add_generation_prompt=True,
)
```

This produces the `<|user|>...</s>\n<|assistant|>` format the model was actually trained on. Earlier experiments using a plain `"...Assistant:"` suffix caused the model to emit chat-template tokens in its outputs (which the reward function then penalized).

### Visual token routing

This is the central piece of plumbing (`receipt_vlm.py::prepare_inputs_embeds`):

1. Encode image → projected visual tokens, shape `(B, Nv, 2048)`.
2. Embed text input ids via the LM's `get_input_embeddings()`, shape `(B, Nt, 2048)`.
3. Concatenate along the sequence dimension: `inputs_embeds = [visual ++ text]`.
4. Build a matching attention mask: ones over visual positions, then the text attention mask.
5. For training: mask labels with `-100` over the visual prefix (no token ids to predict there) and over the instruction prefix (only the target JSON contributes to loss).
6. Pass `inputs_embeds=` instead of `input_ids=` to the LM's forward / generate.

The LM never sees visual content as token ids; it sees them as continuous embeddings that the projector has learned to make look like its own vocabulary.


## Data Pipeline

**Dataset:** [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) — scanned receipts (mostly Indonesian) with parsed JSON ground truth.

**Training schema:**

```json
{
  "line_items": [
    {"name": "PEPPER AUS (WELL DONE)", "count": "1", "price": "165,000"},
    {"name": "ICED LEMON TEA", "price": "22,000"}
  ],
  "total": "580,965"
}
```

Design decisions:

- All fields are strings. Mixing types makes generation harder for a 1.1B model.
- `count` is **optional** — only emitted when the receipt visually shows a quantity. Defaulting to `"1"` would teach the model to hallucinate quantities not present in the image.
- Sub-items (modifiers like `"WELL DONE"`, `"MEDIUM WELL"`) are **flattened into the parent name**. They're not real line items (no own price), but the information is preserved.
- Prices kept as-is (`"43.636"`, `"33,000"`); normalization happens in the reward and eval, not in labels.

**Parser normalization** (`data/dataset.py::parse_ground_truth`):
- CORD stores `menu` as a list for multi-item receipts and as a **single dict** for single-item ones. Both are normalized to a list. (Missing this in v0 silently dropped ~340 of 800 training samples.)
- `gt_parse.total` may be missing or non-dict; defensive extraction.
- Items missing both name and price are dropped.

**Dataset filtering** (`data/dataset.py::CORDDataset`):
The dataset exposes per-reason counters so failures are visible:
`num_loaded`, `num_parse_failed`, `num_empty_items`, `num_missing_total`, `num_too_long`, `num_after_filtering`.

Defaults: 800 train / 100 val / 100 test samples, sorted by image area (highest resolution first, since small text is the main difficulty).


## Training

### Stage 1 — Supervised Fine-Tuning

Teach the projector to translate Donut features into TinyLlama's embedding space well enough that the LM produces receipt-shaped JSON.

| Setting | Value |
|---|---|
| Loss | Cross-entropy on target JSON tokens (visual + instruction masked) |
| Epochs | 15 |
| Batch size | 4 (× 4 grad accum) |
| Optimizer | AdamW, lr 5e-5, weight decay 0.01 |
| Scheduler | Cosine with warmup |
| Gradient clipping | 0.5 |
| Mixed precision | bf16 on CUDA, fp32 on CPU |
| Trainable | Projector only (`set_projector_only_trainable`) |
| Checkpoint | Best val loss saved as `sft_best.pt` |

### Stage 2 — RL Alignment

Reinforce strict-JSON formatting and schema compliance via reward optimization.

In practice, SFT alone gets the model to emit strict JSON in most cases by ~step 100 — format adherence is largely an SFT win on this dataset. RL's marginal value is the residual failure modes that pure-loss minimization doesn't directly penalize: duplicate line items, runaway repetition, and content drift from the image. The reward function's hallucination component is what drives those gains.

**Algorithm: group-relative REINFORCE with KL regularization.**

For each image:
1. Sample K=4 completions from the current policy (the projector under training).
2. Compute the reward for each completion.
3. Compute group-relative advantages: `A_i = (R_i - μ) / (σ + 1e-8)`.
4. If all rewards are equal, skip the update (no preference signal).
5. Score completions under both the **current policy** and the **frozen SFT projector** (the reference) to get token log-probs.
6. Optimize:

```
   loss = -E[ A · mean_token_log_pi(y|x) ] + β · KL(π_ref || π_policy)
```

| Setting | Value |
|---|---|
| K (completions/image) | 4 |
| Temperature | 0.7 (top-p 0.95) |
| Max completion tokens | 192 |
| Optimizer | AdamW, lr 5e-6, weight decay 0.01 |
| KL coefficient β | 0.02 |
| KL estimator | Per-token Schulman k3: `exp(r) - r - 1` where `r = log π_ref − log π_policy` |
| Gradient clipping | 0.5 |
| Epochs | 2 over the training data |

**Why not full GRPO with PPO clipping?** Because the standard rationale for clipping doesn't apply here:
- We do a single optimization step per generated batch (no multi-epoch reuse over old samples), so the on-policy / off-policy gap is zero at the moment of update.
- Only the projector trains (~6M params), with a very small LR. The policy can't drift far in one step.
- The KL term to the SFT reference already constrains drift.

The honest naming for what's implemented: **group-relative REINFORCE + KL**. The advantage structure is GRPO-shaped; the surrogate loss is plain log-prob, not the clipped ratio.


## Reward Design

The reward function lives in `training/rewards.py` and operates on a single (generated_text, ground_truth_text) pair. It aligns to the assignment language: *"Reward formatting compliance and schema adherence; penalize hallucinations, unformatted text blocks, or malformed JSON."*

Four components, total clipped to [0, 1]:

| Component | Range | What it rewards |
|---|---|---|
| **format** | 0.0 – 0.30 | 0.30 for strict JSON, 0.05 for parseable-but-wrapped, 0 otherwise |
| **schema** | 0.0 – 0.30 | Required top-level keys + required item keys, line_items is a list |
| **content** | 0.0 – 0.30 | Line-item count similarity, total digit match, item-name token overlap |
| **hallucination** | ≤ 0 | Duplicates, leaked chat tokens, repeated-character garbage, >3× too many items |

### Aligning the format reward to the eval metric

The assignment defines format adherence as *"successfully parses as valid JSON"* — i.e. `json.loads(prediction)` succeeds on the entire output. The eval metric (`format_adherence_rate`) implements exactly this.

The reward gives full credit (0.30) only when the output is strict JSON (the whole string is the JSON object). Wrapped JSON ("Sure, here's the data: {...} let me know if...") gets a small 0.05 — enough gradient for RL to learn that removing the prose wrapper is preferred, without falling off a cliff to zero. The 0.25 differential between wrapped and strict is the training signal that pushes toward the assignment's headline metric.

### Required item keys: `{name, price}`

`count` is **not** required. The dataset only emits `count` when the receipt visually shows a quantity, so requiring it in the reward would punish the model for faithfully matching the image. The model still learns to emit `count` when present (via SFT labels), it just isn't penalized for omitting it.


## Evaluation

`evaluation/evaluate.py` runs both SFT and RL checkpoints over the held-out test split and writes a comparison summary.

**Headline metric: `format_adherence_rate`** — the percentage of test outputs where:
1. `json.loads(prediction)` succeeds on the full string (strict JSON, no extraction).
2. The result is a dict containing both `line_items` and `total`.
3. `line_items` is a list.
4. `total` is a non-empty value.

This is exactly the assignment's required metric.

**Also tracked:**
- `strict_json_rate` — strict parse only, no key check.
- `extractable_json_rate` — JSON parse after regex extraction (the lenient version).
- `required_keys_rate`, `line_items_list_rate`, `total_present_rate` — schema components.
- `total_match_rate` — digit-only equality with ground-truth total.
- `mean_reward` — same reward function used during RL.

Outputs per stage:
- `results/eval_{stage}/summary.json`
- `results/eval_{stage}/samples.jsonl`
- `results/eval_{stage}/samples.md` — human-readable side-by-side
- `results/eval_comparison.json` — SFT vs RL deltas


## Results

> **Status:** run pending — populate after current training completes.

Headline numbers from a `receipt-base` run:

| Metric | SFT | RL | Δ |
|---|---|---|---|
| format_adherence_rate | _TBD_ | _TBD_ | _TBD_ |
| strict_json_rate | _TBD_ | _TBD_ | _TBD_ |
| required_keys_rate | _TBD_ | _TBD_ | _TBD_ |
| mean_reward | _TBD_ | _TBD_ | _TBD_ |

Loss curves and per-step metrics: see TensorBoard at `runs/`.

### Behavior shift

SFT learns strict JSON quickly. The interesting RL win is on failure modes that loss alone doesn't penalize:

**SFT failure mode 1 — duplicate line items:**
```
<TBD: SFT sample with duplicated items, e.g. PESA PESA × 3>
```

**RL on the same image:**
```
<TBD: RL sample with deduplicated items>
```

**SFT failure mode 2 — runaway repetition:**
```
<TBD: SFT sample with truncation runaway, e.g. PAKET 1, PAKET 2, ..., PAKET 17 ...>
```

**RL on the same image:**
```
<TBD: RL sample, well-bounded output>
```


## Project Structure

```
vlm/
├── configs/
│   ├── paths.py
│   ├── training_configs.py    # named run configs: debug, receipt-base
│   └── training_schema.py     # dataclass schemas
├── data/
│   ├── collator.py            # SFT batch builder, label masking
│   └── dataset.py             # CORD loader + ground-truth normalization
├── models/
│   ├── language_model.py      # TinyLlama wrapper
│   ├── projector.py           # MLP bridge
│   ├── receipt_vlm.py         # composite model + visual token routing
│   ├── vision_encoder.py      # Donut encoder wrapper
│   └── model_checks.py
├── training/
│   ├── common.py              # tokenizer prep, chat template helper, freezing
│   ├── generate.py            # K-sample generation for RL
│   ├── rewards.py             # reward function
│   ├── rl.py                  # RL training loop
│   ├── rl_utils.py            # token log-prob computation, GRPO-style loss
│   └── sft.py                 # SFT training loop
├── evaluation/
│   ├── evaluate.py            # test-set evaluation
│   └── visual_compare.py      # side-by-side SFT vs RL
├── utils/
│   ├── device.py
│   ├── hf_login.py
│   ├── json_extractor.py      # robust JSON parsing/extraction
│   └── training.py            # autocast, seeds, freeze assertions
└── scripts/                   # CLI entry points
    ├── train_sft.py
    ├── train_rl.py
    └── evaluate.py
```


## Design Decisions

A few non-obvious calls worth surfacing:

**Donut over CLIP/SigLIP.** Donut is pretrained on documents; CLIP is pretrained on web images. For receipts the difference is real — Donut handles small dense text far better at the same parameter count.

**LayerNorm at the projector output.** Without it, the visual tokens have magnitudes that don't match the LM's learned token embedding distribution, and the LM tends to either ignore them or produce degenerate output.

**`count` only when present.** Encoding "no quantity shown on the receipt" as `count: "1"` would train the model to fabricate quantities. Dropping the key when the receipt doesn't show one keeps the model honest to the visual input.

**Strict JSON as the format target, with a small consolation reward for wrapped JSON.** Reward and eval use the same definition of "valid JSON" (full-string `json.loads`). The 0.05 reward for wrapped JSON exists only as a gradient — without it, a model emitting wrapped JSON has zero signal toward dropping the wrapper.

**Per-token KL (Schulman k3), not sequence-mean.** k3 is convex, so `mean(k3(x))` ≠ `k3(mean(x))`. Computing KL on sequence-averaged log-probs systematically underestimates token-level divergence.

**No PPO clipping.** With one optimization step per generated batch and projector-only training, the clipped surrogate reduces to log-prob in expectation. Adding it would be theater. See the RL section above.


## Scaling to Production (Multi-page Bills of Lading)

A production document-AI pipeline differs from this assignment on several axes:

**1. Throughput**
- Current K=4, batch=1 RL is sequential. In production, batch over images and over K simultaneously.
- Cache Donut features per image; reuse across SFT/RL/eval passes.
- Serve the LM via vLLM or TGI at inference. The projector is tiny and runs anywhere; the bottleneck is the LM forward pass.

**2. Multi-page documents**
- Bills of lading often span 3–10 pages. Two architectures:
  - **Concat all pages**: encode each page, concatenate visual tokens, run the LM once. Simple, accurate, but blows the context window past 5–6 pages.
  - **Per-page + aggregator**: encode pages independently, extract per-page JSON, merge in post-processing. Cheaper, but cross-page references (e.g. a summary on page 1 referencing items on page 4) become tricky.
- Donut tops out at ~1200 visual tokens per page at 640×960. Higher resolution or page tiling needed for dense forms.

**3. Layout-aware extraction**
- For BoL with complex tables and many fields, swap Donut for **LayoutLMv3** or **DocFormerV2** — they encode bounding boxes + OCR text + image jointly. Donut is great for moderately structured docs; less great for dense tabular forms.

**4. Schema flexibility**
- Receipt schema is fixed here. BoL schemas vary by carrier, shipper, and document type.
- Option: include the target JSON schema (or a JSON Schema spec) in the prompt. Train the model to extract any schema it's shown.
- Option: per-document-type adapters on the projector (LoRA-style), routed by a lightweight document classifier.

**5. Quality monitoring**
- The reward function is your runtime validator. Score every prediction; flag low-reward outputs for review.
- Format adherence is your SLA. Below X% → fall back to a stronger model or human queue.
- Drift detection: log adherence per day / per carrier / per document type. Retrain when it degrades.

**6. Evaluation infrastructure**
- Test set must mirror production: handwritten variants, low-quality scans, multi-page, different shippers.
- Per-field metrics: not just JSON validity. Per-line precision/recall. Total exact-match. Date-field IoU.
- Adversarial set: near-duplicate items, partially-occluded text, malformed source documents.

**7. Active learning**
- Pipe low-reward production samples back into the training set after human correction. Project labels are cheap if you only label what the model gets wrong.


## Limitations

- **OCR accuracy**: 1.1B LMs struggle with small or handwritten text on noisy scans. The architecture works; the model is small. Swapping in a 7B+ LM would improve substantially with minimal code change.
- **Single GPU budget**: capped batch size and K. Larger K would give better advantage estimates and faster RL convergence.
- **Domain**: CORD is Indonesian-Korean receipts. English-language receipts in production would benefit from continued pre-training on a domain-matched corpus.
- **No reward-hacking guardrails**: the reward function is honest but not adversarial. A determined model could find degenerate outputs that score well — for production, add an adversarial test suite to the reward CI.


## Acknowledgements

- [Donut](https://arxiv.org/abs/2111.15664) (Kim et al., 2022) for the vision encoder.
- [TinyLlama](https://arxiv.org/abs/2401.02385) for the language model.
- [CORD](https://github.com/clovaai/cord) for the receipt dataset.
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) for the group-relative advantage idea. 