from vlm.configs.experiments import get_experiment
from vlm.training.rl import train_rl
from vlm.utils.training import set_seed

set_seed(42)


EXPERIMENT_NAME = "debug"
if __name__ == "__main__":
    cfg = get_experiment(EXPERIMENT_NAME)
    train_rl(
        dataset_name=cfg.data.dataset_name,
        train_split=cfg.data.train_split,
        train_samples=cfg.data.train_samples,
        vision_model_name=cfg.vision.model_name,
        image_height=cfg.vision.image_height,
        image_width=cfg.vision.image_width,
        lm_name=cfg.model.lm_name,
        instruction=cfg.model.instruction,
        epochs=cfg.rl.epochs,
        completions_per_image=cfg.rl.completions_per_image,
        learning_rate=cfg.rl.learning_rate,
        weight_decay=cfg.rl.weight_decay,
        temperature=cfg.rl.temperature,
        max_completion_tokens=cfg.rl.max_completion_tokens,
        kl_coef=cfg.rl.kl_coef,
        log_every=cfg.rl.log_every,
        sample_every=cfg.rl.sample_every,
        sft_checkpoint_path=cfg.sft_best_checkpoint,
        run_dir=cfg.rl_run_dir,
        checkpoint_dir=cfg.rl_checkpoint_dir,
        best_checkpoint_path=cfg.rl_best_checkpoint,
    )
