import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import get_scorer


class DestinationTask(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        print(config)
        self._fnn_road = nn.Sequential(
            nn.Linear(config["input_size"], int(config["input_size"] * 2)),
            nn.ReLU(),
            nn.Linear(int(config["input_size"] * 2), config["num_segments"]),
        )
        self._loss = nn.CrossEntropyLoss()
        self.soft = nn.Softmax(dim=-1)

        # load metrics
        self.metrics_single = []
        self.metrics_multi = []
        for metric_str, args in config["metrics"]:
            if metric_str in ["top_k_accuracy"]:
                self.metrics_multi.append((metric_str, get_scorer(metric_str), args))
            else:
                self.metrics_single.append((metric_str, get_scorer(metric_str), args))

        assert len(self.metrics_single) + len(self.metrics_multi) > 0, "No metric provided  - can`t proceed"

    def forward(self, x):
        x = self._fnn_road(x)
        if not self.training:
            x = self.soft(x)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        out = self.forward(X)

        loss = self._loss(out.squeeze(), y)

        self.log("train_loss", loss, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred = self(X)
        pred_single = pred.argmax(dim=-1).squeeze().detach().cpu()
        self.log_dict(
            {
                name: metric._score_func(
                    y_true=y.detach().cpu(),
                    y_pred=pred_single,
                    **args
                )
                for name, metric, args in self.metrics_single
            }, logger=True
        )
        # For faster validation speed, we only do single metrics

    def test_step(self, batch, batch_idx: int):
        X, y = batch
        pred = self(X).detach().cpu()
        pred_single = pred.argmax(dim=-1).squeeze()
        self.log_dict(
            {
                name: metric._score_func(
                    y_true=y.detach().cpu(),
                    y_pred=pred_single,
                    **args
                )
                for name, metric, args in self.metrics_single
            }
        )
        self.log_dict(
            {
                f"{name}@{args['k']}": metric._score_func(
                    y_true=y.detach().cpu(),
                    y_score=pred,
                    labels=np.arange(pred.shape[1]),
                    **args
                )
                for name, metric, args in self.metrics_multi
            }
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer

    @property
    def name(self):
        return self.__class__.__name__
