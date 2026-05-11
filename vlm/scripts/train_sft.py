import argparse

from vlm.configs.experiments import EXPERIMENTS, get_experiment
from vlm.training.sft import train_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT training.")

    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default="receipt-base",
        choices=sorted(EXPERIMENTS),
        help="Experiment config name.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_experiment(args.experiment)

    print(f"running SFT experiment: {cfg.name}")
    train_sft(cfg)


if __name__ == "__main__":
    main()