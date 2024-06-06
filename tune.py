import os, logging
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

import ray
from ray import tune, air
from ray.tune.search.hyperopt import HyperOptSearch
# from ray.tune.schedulers import AsyncHyperBandScheduler

import torch
from plmnist.plmnist import build_model, run_training, verify_config, test
from plmnist.config import (
    DATA_PATH,
    LOG_PATH,
    RESULT_PATH,
    NUM_EPOCHS,
    BATCH_SIZE,
    HIDDEN_SIZE,
    LEARNING_RATE,
    DROPOUT_PROB,
    # FGSM_EPSILON,
    SEED,
)

logger = logging.getLogger(__name__)


class MNISTTrainable(tune.Trainable):

    default_config = dict(
        max_epochs=NUM_EPOCHS,
        log_path=LOG_PATH,
        data_dir=DATA_PATH,
        batch_size=BATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        dropout_prob=DROPOUT_PROB,
        seed=SEED,
    )

    def setup(self, config: dict):
        verify_config(config)

        self.seed = config.pop("seed")
        self.config = config
        self.trainer, self.model = build_model(**self.config)

    def step(self):
        run_training(self.trainer, self.model)
        results = test(self.trainer, self.seed)
        return results

    def reset_config(self, new_config):
        self.setup(new_config)

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[dict]:
        # validate_save_restore
        ckpt_path = Path(checkpoint_dir).resolve() / "model.ckpt"
        self.trainer.save_checkpoint(ckpt_path)

        return {
            "model_path": ckpt_path,
            "config": self.config,
        }
    
    def load_checkpoint(self, checkpoint: dict):
        # validate_save_restore
        self.reset_config(checkpoint["config"])
        self.model.load_state_dict(torch.load(checkpoint["model_path"]))


parameter_space = {
    "max_epochs": NUM_EPOCHS,
    "log_path": LOG_PATH,
    "data_dir": DATA_PATH,
    "batch_size": BATCH_SIZE,
    "hidden_size": tune.qlograndint(16, 64, 16),
    "learning_rate": tune.choice([1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
    "dropout_prob": tune.quniform(0.0, 0.7, 0.1),
    "seed": SEED,
}

if __name__ == "__main__":
    now = datetime.now()
    total_hours = 1
    end = now + timedelta(hours=total_hours)
    name = "mnist_" + now.strftime("%Y%m%d-%H%M%S")

    logging.basicConfig(level="INFO")
    logger.info(f"Starting tuning experiment {name}, ending in {total_hours} hours ({end})")

    mode="max"
    metric="test_acc"

    ray.init(address="auto", _redis_password=os.environ.get("RAY_REDIS_PASSWORD", None))

    tuner = tune.Tuner(
        trainable=MNISTTrainable,
        num_samples=1,
        param_space=parameter_space,
        tune_config=tune.TuneConfig(
            mode=mode,
            metric=metric,
            search_alg=HyperOptSearch(points_to_evaluate=[MNISTTrainable.default_config]),
            # scheduler=AsyncHyperBandScheduler(
            #     time_attr="training_iteration",
            #     metric="test_acc",
            #     mode="max",
            #     max_t=NUM_EPOCHS,
            #     grace_period=1,
            # ),
            num_samples=10,
            reuse_actors=True,
            time_budget_s=total_hours * 3600,
        ),
        run_config=air.RunConfig(
            name=name,
            local_dir=RESULT_PATH,
            # stop={"training_iteration": NUM_EPOCHS},
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute=metric,
                checkpoint_score_order=mode,
            ),
            progress_reporter=tune.CLIReporter(
                metric_columns=["test_acc", "test_loss", "training_iteration"],
                parameter_columns=[k for k in parameter_space.keys() if type(k).__module__.startswith("ray.tune")],
                sort_by_metric=True
            )
        )
    )

    analysis = tuner.run()
    analysis.get_dataframe().to_csv(f"{RESULT_PATH}/{name}.csv")

    