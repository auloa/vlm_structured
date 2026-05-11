import argparse

from vlm.configs.experiments import EXPERIMENTS, get_experiment
from vlm.training.rl import train_rl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL training.")

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

    print(f"running RL experiment: {cfg.name}")
    print(f"loading SFT checkpoint from: {cfg.sft_best_checkpoint}")

    train_rl(cfg)


if __name__ == "__main__":
    main()