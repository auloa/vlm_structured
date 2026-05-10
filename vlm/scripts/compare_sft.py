from vlm.configs.experiments import finetuned_donut
from vlm.evaluation.visual_compare import build_visual_comparison_report


def main():
    cfg = finetuned_donut()

    build_visual_comparison_report(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.test_split,
        num_samples=20,
        vision_model_name=cfg.vision.model_name,
        image_height=cfg.vision.image_height,
        image_width=cfg.vision.image_width,
        lm_name=cfg.model.lm_name,
        instruction=cfg.model.instruction,
        checkpoint_path=cfg.sft_best_checkpoint,
        output_html_path=cfg.results_dir / "sft_visual_comparison.html",
        max_completion_tokens=cfg.eval.max_completion_tokens,
        temperature=cfg.eval.temperature,
        do_sample=False,
    )


if __name__ == "__main__":
    main()