def hide_logs():
    import logging, warnings

    # needs to be run on each worker
    warnings.filterwarnings("ignore", "Checkpoint directory .* exists and is not empty")
    for module in logging.Logger.manager.loggerDict:
        if module.startswith(
            (
                "lightning",
                "pytorch",
                "ray.train._internal.storage",
                "ray.tune.execution.experiment_state",
            )
        ):
            logging.getLogger(module).setLevel(logging.FATAL)


def test_trainable():
    from plmnist.tune.trainable import MNISTTrainable
    from plmnist.tune.parameter_space import default_config
    from ray.tune.utils import validate_save_restore

    # first, test one step of training
    trainable = MNISTTrainable()
    trainable.setup(default_config)
    results = trainable.train()
    print(results)

    # then, validate the save/restore functionality
    assert validate_save_restore(MNISTTrainable, default_config)


def make_trial_name():
    from datetime import datetime, timedelta
    from plmnist.tune.config import MAX_HOURS

    now = datetime.now()
    end = now + timedelta(hours=MAX_HOURS)
    name = "mnist_" + now.strftime("%Y%m%d-%H%M%S")

    return name, end
