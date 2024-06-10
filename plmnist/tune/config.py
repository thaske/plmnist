import os
from pathlib import Path

os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"

from ray import air, tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler

from plmnist.config import LOG_PATH

from plmnist.tune.parameter_space import default_config


# number of epochs per tuning step
STEP_EPOCHS = 2

# maximum number of steps (may be stopped early)
MAX_STEPS = 5

# maximum time budget for tuning
MAX_HOURS = 15 / 60  # 15 minutes

# maximum number of samples to run
NUM_SAMPLES = 1000  # max number of samples to run

# directory to store the ray results
RAY_RESULTS_DIR = (Path(LOG_PATH) / "ray_results").resolve()

# mode and metric for the search algorithm
MODE, METRIC = "max", "test_acc"

# define the search algorithm
search_alg = HyperOptSearch(
    # HyperOpt is a TPE-based search algorithm
    points_to_evaluate=[default_config],  # start with the default config(s)
    n_initial_points=5,  # and add a few more random points before running the optimization
)

# define the scheduler
scheduler = AsyncHyperBandScheduler(
    # AsyncHyperBand will automatically stop poorly performing trials
    max_t=MAX_STEPS,
    grace_period=1,  # minimum number of steps to train for before stopping a trial
)

# populate TuneConfig
tune_config = tune.TuneConfig(
    mode=MODE,
    metric=METRIC,
    search_alg=search_alg,
    scheduler=scheduler,
    num_samples=NUM_SAMPLES,
    time_budget_s=MAX_HOURS * 3600,
    reuse_actors=True,  # this saves some overhead but requires defining reset_config()
)

# configure how many checkpoints to keep
checkpoint_config = air.CheckpointConfig(
    # save the best checkpoint based on the test accuracy
    num_to_keep=1,
    checkpoint_score_attribute=METRIC,
    checkpoint_score_order=MODE,
)


# populate RunConfig, dynamically to set `name` based on the experiment start time
def get_run_config(name: str):

    run_config = air.RunConfig(
        # the RunConfig class is where experiment options are given.
        name=name,
        storage_path=RAY_RESULTS_DIR,  # if not specified, ray will store results in ~/ray_results
        stop=dict(training_iteration=MAX_STEPS),  # stop after MAX_STEPS steps
        checkpoint_config=checkpoint_config,
        log_to_file=True,
    )
    return run_config
