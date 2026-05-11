# vlm

Custom Vision-language model for structured data extraction from scanned receipts. Submitted for the Document AI Alignment take-home assignment.

## README Outline

- **Quick Start** — setup and how to run the pipeline
- **Architecture** — vision encoder choice, projector, visual token routing
- **Data** — CORD-v2, target schema, parser decisions
- **Training** — SFT and RL configs and procedures
- **Reward** — components and how they map to the eval metric
- **Evaluation** — what's measured and how
- **Results** — headline numbers and behavior notes
- **Design Decisions** — non-obvious choices, alternatives explored
- **Scaling to Production** — what would change for multi-page bills of lading
- **Limitations** — what's not addressed and why
- **Project Structure** — repo layout

## Requirements

- Single GPU with 8–16 GB VRAM (tested on RTX 3090)
- Python 3.10+
- `uv` for dependency management
- HF token for downloading model weights

## Quick Start

```bash
uv sync
python -m vlm.utils.hf_login

python -m vlm.scripts.train_sft -c tlama_
python -m vlm.scripts.train_rl  -c receipt_base
python -m vlm.scripts.evaluate  -c receipt_base
```

TensorBoard:

```bash
tensorboard --logdir training_runs/
```

A `-c debug` configuration runs the full pipeline on 20 samples in a few minutes for sanity-checking.