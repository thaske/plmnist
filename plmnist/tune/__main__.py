import os
from pathlib import Path

import ray
from ray.tune import Tuner

from plmnist.config import LOG_PATH

from plmnist.tune.parameter_space import parameter_space
from plmnist.tune.utils import hide_logs, make_trial_name
from plmnist.tune.trainable import MNISTTrainable
from plmnist.tune.config import tune_config, get_run_config


if __name__ == "__main__":
    hide_logs()
    # first, initialize the ray worker. Note you have to run `ray start --head` before running this script.
    ray.init(address="auto", _redis_password=os.environ.get("RAY_redis_password", None))

    # run the tuning process
    name, end = make_trial_name()

    # add the name to the trainable class so we can store logs there
    class NamedMNISTTrainable(MNISTTrainable):
        _session_name = name

    print(f"Starting tuning experiment {name}, ending at {end}")

    # initialize the tuner class. this is where all the configuration is done.
    #   all of these classes have many options, so make sure to check the docstring/documentation for each of them.
    tuner = Tuner(
        # the kwargs in Tuner are the main options: what Trainable to use, the number of samples, etc.
        trainable=NamedMNISTTrainable,
        param_space=parameter_space,
        tune_config=tune_config,
        run_config=get_run_config(name),
    )

    # finally, run the tuning process
    analysis = tuner.fit()

    # save the results to a CSV file in the storage_path
    analysis.get_dataframe().to_csv(
        Path(LOG_PATH) / "ray_results" / name / "results.csv"
    )

    print(f"Finished tuning experiment {name}")
