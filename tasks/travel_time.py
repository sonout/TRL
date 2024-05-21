import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import get_scorer


class TravelTimeTask(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._decoder_fnn = nn.Sequential(
            nn.Linear(config["input_size"], config["input_size"] * 2),
            nn.ReLU(),
            nn.Linear(config["input_size"] * 2, 1),
        )
        self._loss = nn.MSELoss()

        # load metrics
        self.metrics = []
        for metric_str, args in config["metrics"]:
            self.metrics.append((metric_str, get_scorer(metric_str), args))

        assert len(self.metrics) > 0, "No metric provided  - can`t proceed"

    def forward(self, x):
        x = self._decoder_fnn(x)

        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        out = self.forward(X)

        loss = self._loss(out.squeeze(), y.float())

        self.log("train_loss", loss, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred = self(X)
        self.log_dict(
            {
                name: metric._score_func(
                    y_true=y.detach().cpu(),
                    y_pred=pred.squeeze().detach().cpu(),
                    **args
                )
                for name, metric, args in self.metrics
            }, logger=True
        )

    def test_step(self, batch, batch_idx: int):
        X, y = batch
        pred = self(X)
        self.log_dict(
            {
                name: metric._score_func(
                    y_true=y.detach().cpu(),
                    y_pred=pred.squeeze().detach().cpu(),
                    **args
                )
                for name, metric, args in self.metrics
            }
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer

    @property
    def name(self):
        return self.__class__.__name__
