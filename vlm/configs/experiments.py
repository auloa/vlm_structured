from vlm.configs.schema import ExperimentConfig


def debug() -> ExperimentConfig:
    cfg = ExperimentConfig(name="debug")

    cfg.data.train_samples = 20
    cfg.data.val_samples = 10
    cfg.data.test_samples = 10

    cfg.vision.image_height = 640
    cfg.vision.image_width = 960

    cfg.sft.epochs = 1
    cfg.sft.batch_size = 1

    cfg.rl.epochs = 1
    cfg.rl.completions_per_image = 2

    return cfg


def donut_tinyllama_sft_rl() -> ExperimentConfig:
    cfg = ExperimentConfig(name="donut-tinyllama-sft-rl")

    cfg.data.train_samples = 800
    cfg.data.val_samples = 200
    cfg.data.test_samples = 50

    cfg.vision.image_height = 640
    cfg.vision.image_width = 960

    cfg.sft.epochs = 2
    cfg.sft.batch_size = 4
    cfg.sft.grad_accum_steps = 4

    cfg.rl.epochs = 1
    cfg.rl.completions_per_image = 4

    return cfg


EXPERIMENTS = {
    "debug": debug,
    "donut-tinyllama-sft-rl": donut_tinyllama_sft_rl,
}


def get_experiment(name: str) -> ExperimentConfig:
    try:
        return EXPERIMENTS[name]()
    except KeyError as exc:
        available = ", ".join(EXPERIMENTS)
        raise ValueError(f"Unknown experiment '{name}'. Available: {available}") from exc