import os

import ray
from ray.tune import Tuner

from plmnist.tune.utils import hide_logs, make_trial_name
from plmnist.tune.trainable import MNISTTrainable
from plmnist.tune.config import (
    parameter_space,
    tune_config,
    get_run_config,
    RAY_RESULTS_DIR,
)


if __name__ == "__main__":
    hide_logs()
    # first, initialize the ray worker. Note you have to run `ray start --head` before running this script.
    ray.init(address="auto", _redis_password=os.environ.get("RAY_redis_password", None))

    # generate a unique name for this tuning experiment and compute the end time `MAX_HOURS`
    name, end = make_trial_name()

    # add the name to the trainable class to store checkpoints persistently
    # otherwise they are stored in a tempdir and deleted upon tuning exit
    # class NamedMNISTTrainable(MNISTTrainable):
    #     _session_name = name

    print(f"Starting tuning experiment {name}, ending at {end}")

    # initialize the tuner with the trainable class, parameter space, and configurations
    tuner = Tuner(
        trainable=MNISTTrainable,
        param_space=parameter_space,
        tune_config=tune_config,
        run_config=get_run_config(name),
    )

    # run the tuning process
    analysis = tuner.fit()

    # save the results to a CSV file
    analysis.get_dataframe().to_csv(RAY_RESULTS_DIR / name / "results.csv")

    print(f"Finished tuning experiment {name}")
