import os, sys, logging
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

import ray
from ray import tune, air
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.utils import validate_save_restore
from ray.tune.schedulers import AsyncHyperBandScheduler

from pytorch_lightning.trainer import setup as pl_setup

from plmnist.plmnist import build_model, run_training, verify_config, test
from plmnist.config import (
    DATA_PATH,
    LOG_PATH,
    BATCH_SIZE,
    HIDDEN_SIZE,
    LEARNING_RATE,
    DROPOUT_PROB,
    SEED,
)

logger = logging.getLogger(__name__)

# define the search space for the hyperparameter tuning
parameter_space = dict(
    batch_size=BATCH_SIZE,
    hidden_size=tune.qlograndint(16, 64, 16),
    learning_rate=tune.qloguniform(1e-8, 1e-1, 1e-8),
    dropout_prob=tune.quniform(0.0, 0.7, 0.1),
    seed=SEED,
)

# define the default configuration for the training process
#   note that this must be within the search space defined above
default_config = dict(
    batch_size=BATCH_SIZE,
    hidden_size=HIDDEN_SIZE,
    learning_rate=LEARNING_RATE,
    dropout_prob=DROPOUT_PROB,
    seed=SEED,
)

# configure the number of epochs per tuning step, the maximum number of steps,
#   the time budget, and the number of samples to run
#   note the maximum number of epochs is then `STEP_EPOCHS * MAX_STEPS`
STEP_EPOCHS = 2
MAX_STEPS = 5
MAX_HOURS = 15 / 60  # 15 minutes
NUM_SAMPLES = 1000  # max number of samples to run

RAY_RESULTS_DIR = (Path(LOG_PATH) / "ray_results").resolve()


# now the main part: defining the tune.Trainable class
class MNISTTrainable(tune.Trainable):
    # two methods are required: `setup()` and `step()`
    #   `setup()` is called once, when initializing the trainable (or when resetting it)
    #   `step()` is called at each tuning step

    # this class also implements `save_checkpoint()`, `load_checkpoint()`, and `reset_config()`
    #   to allow for saving and restoring the state of the trainable
    #   this is useful for using fancier search/scheduling algorithms that can do early stopping/resuming.

    _session_name: Optional[str]  # will be set once the experiment is started

    def setup(self, config: dict):
        # first, verify the config is correct. we disable add_defaults since we provide all options here.
        verify_config(config, add_defaults=False)

        pl_setup._log_device_info = lambda trainer: None  # disable device info logging

        # create the model and the pytorch lightning trainer
        self.seed = config["seed"]
        self.config = config
        self.trainer, self.model = build_model(
            max_epochs=STEP_EPOCHS,
            verbose=False,
            log_path=RAY_RESULTS_DIR / self._session_name,
            # log_path=self.logdir,  # if you don't plan to use checkpoints after tuning
            **self.config,
        )
        self.resume_ckpt_path = None

    def step(self):
        # set max_epochs appropriately for this step
        self.trainer.fit_loop.max_epochs = STEP_EPOCHS * (self.training_iteration + 1)

        # run training for STEP_EPOCHS epochs, potentially resuming from a checkpoint
        run_training(self.trainer, self.model, ckpt_path=self.resume_ckpt_path)

        # clear the resume_ckpt_path so we don't accidentally resume from the old checkpoint next step
        self.resume_ckpt_path = None

        # test the model and return the results
        results = test(
            self.trainer, self.seed, verbose=False
        )  # results will be reported in a table
        return results

    def reset_config(self, new_config):
        # to avoid some overhead in creating the Trainable class,
        #   we can define the reset_config() method.
        #   since we don't edit the Trainable's state during step(), we can just call setup() again.
        self.setup(new_config)

        return True  # must return True

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[dict]:
        # save the model checkpoint
        ckpt_path = Path(checkpoint_dir).resolve() / "model.ckpt"
        self.trainer.save_checkpoint(ckpt_path)

        # return a dictionary here that contains a path to the model checkpoint
        #   note you can return the checkpoint itself, but this should be nicer
        #   since the dictionary will be pickled and (potentially) sent over the network.
        #   you can also return `None`, which will instead call `load_checkpoint` using `checkpoint_dir`
        return {
            "model_path": ckpt_path,
            "config": self.config,
        }

    def load_checkpoint(self, checkpoint: dict):
        # load the model checkpoint
        self.reset_config(checkpoint["config"])

        # make sure to set `resume_ckpt_path` so the next step will load the weights from the given checkpoint
        self.resume_ckpt_path = checkpoint["model_path"]


# now we run the tuning process
if __name__ == "__main__":
    # first, initialize the ray worker. Note you have to run `ray start --head` before running this script.
    ray.init(address="auto")

    if "--test" in sys.argv:
        # if "--test" is passed, validate that the class is working as expected
        #   first, test one step of training
        trainable = MNISTTrainable()
        trainable.setup(default_config)
        results = trainable.train()

        # then, validate the save/restore functionality
        validated = validate_save_restore(MNISTTrainable, default_config)
        print("~" * 80 + "\n\t Validated\n" + "~" * 80)
    else:
        # otherwise, run the tuning process
        now = datetime.now()
        end = now + timedelta(hours=MAX_HOURS)
        name = "mnist_" + now.strftime("%Y%m%d-%H%M%S")

        # add the name to the trainable class so we can store logs there
        MNISTTrainable._session_name = name

        logger.info(
            f"Starting tuning experiment {name}, ending in {MAX_HOURS} hours ({end})"
        )

        # we want to maximize the test accuracy
        mode = "max"
        metric = "test_acc"

        # initialize the tuner class. this is where all the configuration is done.
        #   all of these classes have many options, so make sure to check the docstring/documentation for each of them.
        tuner = tune.Tuner(
            # the kwargs in Tuner are the main options: what Trainable to use, the number of samples, etc.
            trainable=MNISTTrainable,
            param_space=parameter_space,
            tune_config=tune.TuneConfig(
                # the TuneConfig class is where the search/scheduling algorithms are defined.
                mode=mode,
                metric=metric,
                search_alg=HyperOptSearch(
                    # HyperOpt is a TPE-based search algorithm
                    points_to_evaluate=[
                        default_config
                    ],  # start with the default config(s)
                    n_initial_points=5,  # and add a few more random points before running the optimization
                ),
                scheduler=AsyncHyperBandScheduler(
                    # AsyncHyperBand will automatically stop poorly performing trials
                    time_attr="training_iteration",
                    max_t=MAX_STEPS,
                    grace_period=1,  # minimum number of steps to train for before stopping a trial
                ),
                num_samples=NUM_SAMPLES,
                reuse_actors=True,  # this saves some overhead but requires defining reset_config()
                time_budget_s=MAX_HOURS * 3600,
            ),
            run_config=air.RunConfig(
                # the RunConfig class is where experiment options are given.
                name=name,
                storage_path=RAY_RESULTS_DIR,  # if not specified, ray will store results in ~/ray_results
                stop=dict(training_iteration=MAX_STEPS),  # stop after MAX_STEPS steps
                checkpoint_config=air.CheckpointConfig(
                    # save the best checkpoint based on the test accuracy
                    num_to_keep=1,
                    checkpoint_score_attribute=metric,
                    checkpoint_score_order=mode,
                ),
                log_to_file=True,
            ),
        )

        # finally, run the tuning process
        analysis = tuner.fit()

        # save the results to a CSV file in the storage_path
        analysis.get_dataframe().to_csv(
            Path(LOG_PATH) / "ray_results" / name / "results.csv"
        )
