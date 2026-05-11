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
    val_samples: int = 100
    test_samples: int = 100


@dataclass
class VisionConfig:
    model_name: str = "naver-clova-ix/donut-base-finetuned-cord-v2"
    default_processor: bool = False
    image_height: int = 640
    image_width: int = 960

@dataclass
class LMConfig:
    lm_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    instruction: str = (
        "Extract the tabular data from this document and output it in JSON format.\nAssistant:"
    )


@dataclass
class SFTConfig:
    epochs: int = 10
    batch_size: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    grad_accum_steps: int = 4
    grad_clip_norm: float = 0.5
    max_target_length: int = 192
    log_every: int = 10
    sample_every: int = 50


@dataclass
class RLConfig:
    epochs: int = 1
    completions_per_image: int = 4
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    temperature: float = 0.7
    max_completion_tokens: int = 192
    grad_clip_norm: float = 0.5
    kl_coef: float = 0.02
    log_every: int = 5
    sample_every: int = 20


@dataclass
class EvalConfig:
    num_samples: int = 50
    max_completion_tokens: int = 192
    temperature: float = 0.1


@dataclass
class TrainingConfig:
    name: str
    data: DataConfig = field(default_factory=DataConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    model: LMConfig = field(default_factory=LMConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def root_dir(self) -> Path:
        return EXPERIMENTS_DIR / self.name

    @property
    def sft_checkpoint_dir(self) -> Path:
        return self.root_dir / "checkpoints" / "sft"

    @property
    def rl_checkpoint_dir(self) -> Path:
        return self.root_dir / "checkpoints" / "rl"

    @property
    def sft_best_checkpoint(self) -> Path:
        return self.sft_checkpoint_dir / "best.pt"

    @property
    def rl_best_checkpoint(self) -> Path:
        return self.rl_checkpoint_dir / "best.pt"

    @property
    def sft_run_dir(self) -> Path:
        return self.root_dir / "runs" / "sft"

    @property
    def rl_run_dir(self) -> Path:
        return self.root_dir / "runs" / "rl"

    @property
    def results_dir(self) -> Path:
        return self.root_dir / "results"