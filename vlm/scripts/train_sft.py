import argparse

from vlm.configs.training_configs import TRAINING_CONFIGS, get_training_config
from vlm.training.sft import train_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT training.")

    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default="receipt_base",
        choices=sorted(TRAINING_CONFIGS),
        help="Experiment config name.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_training_config(args.experiment)

    print(f"running SFT experiment: {cfg.name}")
    train_sft(cfg)


if __name__ == "__main__":
    main()