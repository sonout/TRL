import os
import sys
from typing import Dict, List

sys.path.append("..")
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from models.model_abtract import BaseModel
from pipelines.utils import ROOT_DIR


class ModelTrainer:
    def __init__(self, model: pl.LightningModule, config: Dict, do_training: bool = True):
        logger = TensorBoardLogger(
            os.path.join(ROOT_DIR, "training_tb_logs"), name=model.name
        )
        # We can skip training if we want
        if do_training:
            limit_train_batches = None
        else:
            limit_train_batches = 1 # 0 did not work due to other error

        # early stopping to prevent overfitting
        stopping_criterium = None
        if config["early_stopping"]:
            stopping_criterium = EarlyStopping(
                monitor=config["monitor_metric"],
                mode=config["mode"],
                min_delta=config["min_delta"],
                patience=config["patience"],
            )
        # initialize trainer
        self.trainer = pl.Trainer(
            accelerator="gpu",
            devices=config["gpu_device_ids"],
            #auto_lr_find=True, removed for lightly env
            logger=logger,
            max_epochs=config["epochs"],
            max_steps=2000 if config["debug"] else -1,
            enable_checkpointing=True,
            callbacks=stopping_criterium,
            limit_train_batches=limit_train_batches,
        )

        self.model = model

    def train(self, train: DataLoader, val: DataLoader = None):
        self.trainer.fit(self.model, train_dataloaders=train, val_dataloaders=val)

        return self._get_state_dict()

    def evaluate_performance(self, test: DataLoader):
        return self.trainer.test(dataloaders=test)

    def _get_state_dict(self):
        return self.model.state_dict()
