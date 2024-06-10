import os, json, hashlib, inspect

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from plmnist import logger
from plmnist.model import LitMNIST
from plmnist.config import (
    DATA_PATH,
    LOG_PATH,
    RESULT_PATH,
    NUM_EPOCHS,
    BATCH_SIZE,
    HIDDEN_SIZE,
    LEARNING_RATE,
    DROPOUT_PROB,
    SEED
)


def build_model(
    max_epochs: int = NUM_EPOCHS,
    log_path: str = LOG_PATH,
    data_dir: str = DATA_PATH,
    batch_size: int = BATCH_SIZE,
    hidden_size: int = HIDDEN_SIZE,
    learning_rate: float = LEARNING_RATE,
    dropout_prob: float = DROPOUT_PROB,
    seed: int = SEED,
    verbose: bool = True,
):
    pl.seed_everything(seed)
    model = LitMNIST(
        data_dir=data_dir,
        batch_size=batch_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        dropout_prob=dropout_prob,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=max_epochs,
        logger=CSVLogger(save_dir=log_path),
        enable_progress_bar=verbose,
        enable_model_summary=verbose,
    )
    return trainer, model

def run_training(trainer: pl.Trainer, model: pl.LightningModule, **kwargs):
    trainer.fit(model, **kwargs)
    return trainer

def train(
    max_epochs: int = NUM_EPOCHS,
    log_path: str = LOG_PATH,
    data_dir: str = DATA_PATH,
    batch_size: int = BATCH_SIZE,
    hidden_size: int = HIDDEN_SIZE,
    learning_rate: float = LEARNING_RATE,
    dropout_prob: float = DROPOUT_PROB,
    seed: int = SEED,
):
    trainer, model = build_model(
        max_epochs=max_epochs,
        log_path=log_path,
        data_dir=data_dir,
        batch_size=batch_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        dropout_prob=dropout_prob,
        seed=seed,
    )

    trainer = run_training(trainer, model)

    return trainer

def test(trainer: pl.Trainer, seed=None, verbose=True):
    results = dict()
    results["config"] = dict()
    results["config"]["batch_size"] = trainer.model.batch_size
    results["config"]["hidden_size"] = trainer.model.hidden_size
    results["config"]["learning_rate"] = trainer.model.learning_rate
    results["config"]["dropout_prob"] = trainer.model.dropout_prob
    results["config"]["max_epochs"] = trainer.max_epochs
    results["config"]["log_dir"] = trainer.logger.log_dir

    if seed is not None:
        results["config"]["seed"] = seed

    if "val_loss" in trainer.callback_metrics:
        results["val_loss"] = trainer.callback_metrics["val_loss"].item()
    
    if "val_acc" in trainer.callback_metrics:
        results["val_acc"] = trainer.callback_metrics["val_acc"].item()

    trainer.test(ckpt_path="best", verbose=verbose)

    results["test_loss"] = trainer.callback_metrics["test_loss"].item()
    results["test_acc"] = trainer.callback_metrics["test_acc"].item()

    results["epochs"] = trainer.current_epoch

    return results


def write(
    results: dict,
    trainer: pl.Trainer,
    directory: str = RESULT_PATH,
    do_dhash: bool = True,
):
    if do_dhash:
        results["hash"] = hashlib.md5(
            json.dumps(results["config"], sort_keys=True).encode()
        ).hexdigest()

        dhash = "-" + results["hash"]
    else:
        dhash = ""

    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/results{dhash}.json", "w") as f:
        json.dump(results, f)

    trainer.save_checkpoint(f"{directory}/model{dhash}.ckpt")

    return dhash



def verify_config(config: dict, add_defaults: bool=True):
    signature = inspect.signature(train)
    for key in config:
        assert key in signature.parameters, f"Unknown parameter found: {key}"
        if not isinstance(config[key], signature.parameters[key].annotation):
            logger.warn(
                f"Parameter {key} has wrong type: {type(config[key])}, "
                f"expected {signature.parameters[key].annotation}! "
                 "This may cause errors downstream."
            )
    if add_defaults:
        for key in signature.parameters:
            if key not in config:
                logger.info(
                    f"Parameter {key} not found in config, "
                    f"using default value: {signature.parameters[key].default}"
                )
                config[key] = signature.parameters[key].default
    return config