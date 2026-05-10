from dataclasses import dataclass, field
from pathlib import Path

from vlm.configs.paths import EXPERIMENTS_DIR


@dataclass
class DataConfig:
    dataset_name: str = "naver-clova-ix/cord-v2"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    train_samples: int = 800
    val_samples: int = 200
    test_samples: int = 50


@dataclass
class VisionConfig:
    model_name: str = "naver-clova-ix/donut-base"
    default_processor: bool = False
    image_height: int = 640
    image_width: int = 960


@dataclass
class ModelConfig:
    lm_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    instruction: str = (
        "Extract the tabular data from this document and output it in JSON format.\nAssistant:"
    )


@dataclass
class SFTConfig:
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_accum_steps: int = 4
    max_length: int = 256
    log_every: int = 10
    sample_every: int = 50


@dataclass
class RLConfig:
    epochs: int = 2
    completions_per_image: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    temperature: float = 0.8
    max_completion_tokens: int = 128
    kl_coef: float = 0.01
    clip_eps: float = 0.2
    log_every: int = 5
    sample_every: int = 20


@dataclass
class EvalConfig:
    num_samples: int = 50
    max_completion_tokens: int = 128
    temperature: float = 0.1




@dataclass
class PathConfig:
    experiment_name: str = "donut-tinyllama-sft-rl"
    sft_checkpoint_dir: Path = EXPERIMENTS_DIR / f"{experiment_name}/checkpoints/sft"
    rl_checkpoint_dir: Path = EXPERIMENTS_DIR / f"{experiment_name}/checkpoints/rl"
    sft_best_checkpoint: Path = EXPERIMENTS_DIR / f"{experiment_name}/checkpoints/sft/best.pt"
    rl_best_checkpoint: Path = EXPERIMENTS_DIR / f"{experiment_name}/checkpoints/rl/best.pt"
    sft_run_dir: Path = EXPERIMENTS_DIR / f"{experiment_name}/runs/sft"
    rl_run_dir: Path = EXPERIMENTS_DIR / f"{experiment_name}/runs/rl"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    paths: PathConfig = field(default_factory=PathConfig)


DEFAULT_CONFIG = Config()