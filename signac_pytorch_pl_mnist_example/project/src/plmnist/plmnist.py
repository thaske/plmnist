import os, json, hashlib

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

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
)


def train(
    max_epochs: int = NUM_EPOCHS,
    log_path: str = LOG_PATH,
    data_dir: str = DATA_PATH,
    batch_size: int = BATCH_SIZE,
    hidden_size: int = HIDDEN_SIZE,
    learning_rate: float = LEARNING_RATE,
    dropout_prob: float = DROPOUT_PROB,
):
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
    )
    trainer.fit(model)

    return trainer, model


def test(trainer: pl.Trainer, seed=None):
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

    results["val_loss"] = trainer.callback_metrics["val_loss"].item()
    results["val_acc"] = trainer.callback_metrics["val_acc"].item()

    trainer.test(ckpt_path="best")

    results["test_loss"] = trainer.callback_metrics["test_loss"].item()
    results["test_acc"] = trainer.callback_metrics["test_acc"].item()

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
