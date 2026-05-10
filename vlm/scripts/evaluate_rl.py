from vlm.config.experiments import donut_tinyllama_sft_rl
from vlm.evaluation.evaluate import evaluate_checkpoint


def main():
    cfg = donut_tinyllama_sft_rl()

    evaluate_checkpoint(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.test_split,
        num_samples=cfg.data.test_samples,
        vision_model_name=cfg.vision.model_name,
        image_height=cfg.vision.image_height,
        image_width=cfg.vision.image_width,
        lm_name=cfg.model.lm_name,
        instruction=cfg.model.instruction,
        checkpoint_path=cfg.rl_best_checkpoint,
        output_dir=cfg.results_dir / "rl_eval",
        max_completion_tokens=cfg.eval.max_completion_tokens,
        temperature=cfg.eval.temperature,
        do_sample=False,
        generation_repetition_penalty=1.0,
    )


if __name__ == "__main__":
    main()