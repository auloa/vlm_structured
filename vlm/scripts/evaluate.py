import argparse
from pathlib import Path

from vlm.configs.training_configs import TRAINING_CONFIGS, get_experiment
from vlm.evaluation.evaluate import compare_sft_and_rl, evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate receipt JSON extraction.")

    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default="debug",
        choices=sorted(TRAINING_CONFIGS),
        help="Experiment config name.",
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="both",
        choices=["sft", "rl", "both", "custom"],
        help="Which checkpoint to evaluate.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Required when --stage custom.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of held-out test examples to evaluate.",
    )

    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="Maximum number of generated completion tokens.",
    )

    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sampling during evaluation. Default is deterministic greedy decoding.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_experiment(args.experiment)

    print(f"running evaluation for experiment: {cfg.name}")

    if args.stage == "both":
        compare_sft_and_rl(
            cfg=cfg,
            num_samples=args.num_samples,
            max_completion_tokens=args.max_completion_tokens,
        )
        return

    if args.stage == "sft":
        checkpoint_path = cfg.sft_best_checkpoint
    elif args.stage == "rl":
        checkpoint_path = cfg.rl_best_checkpoint
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --stage custom")
        checkpoint_path = Path(args.checkpoint)

    evaluate_checkpoint(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        stage=args.stage,
        num_samples=args.num_samples,
        max_completion_tokens=args.max_completion_tokens,
        do_sample=args.sample,
    )


if __name__ == "__main__":
    main()