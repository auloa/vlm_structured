from vlm.configs.experiments import get_experiment
from vlm.training.sft import train_sft
from vlm.utils.training import set_seed

set_seed(42)


EXPERIMENT_NAME = "finetuned-donut"


if __name__ == "__main__":
    cfg = get_experiment(EXPERIMENT_NAME)
    train_sft(
        dataset_name=cfg.data.dataset_name,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        train_samples=cfg.data.train_samples,
        val_samples=cfg.data.val_samples,
        vision_model_name=cfg.vision.model_name,
        image_height=cfg.vision.image_height,
        image_width=cfg.vision.image_width,
        lm_name=cfg.model.lm_name,
        instruction=cfg.model.instruction,
        epochs=cfg.sft.epochs,
        batch_size=cfg.sft.batch_size,
        learning_rate=cfg.sft.learning_rate,
        weight_decay=cfg.sft.weight_decay,
        grad_accum_steps=cfg.sft.grad_accum_steps,
        grad_clip_norm=cfg.sft.grad_clip_norm,
        max_target_length=cfg.sft.max_target_length,
        log_every=cfg.sft.log_every,
        sample_every=cfg.sft.sample_every,
        run_dir=cfg.sft_run_dir,
        checkpoint_dir=cfg.sft_checkpoint_dir,
        best_checkpoint_path=cfg.sft_best_checkpoint,
    )