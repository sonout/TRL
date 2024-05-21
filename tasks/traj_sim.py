import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import get_scorer


class TrajSimTask(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._ghost_nn = nn.Linear(1, 1)
        # load metrics
        self.metrics = []
        for metric_str, args in config["metrics"]:
            self.metrics.append((metric_str, get_scorer(metric_str), args))

        assert len(self.metrics) > 0, "No metric provided  - can`t proceed"

    def forward(self, xb, embs):
        # x: bxe embs: nxe
        embs = embs[0, :, :] # embs are the same for all batches, so just take first -> [all_traj, emb_dim]
        sims = (xb[:, None, :] * embs).sum(dim=2)
        return sims

    def training_step(self, batch, batch_idx):
        # Not needed but must match deocoder task model pattern
        return None

    def test_step(self, batch, batch_idx: int):
        X, y, idx = batch # X: [batch, emb_dim], y: [copy*batch, all_traj, emb_dim], idx: [batch]
        preds = self(X, y) # [batch, all_traj] -> similarity score for each traj

        ## Calculate Mean Rank 
        ranks = torch.argsort(preds, dim=1, descending=True)
        rank = torch.where(ranks == idx[:, None])[1]
        # Note: if we have outliers, then the mean_rank gets very skewed. Try remove outliers 
        if self.config['remove_outliers']:
            rank = rank[rank < self.config['remove_outliers_from_rank']]
        self.log_dict({"mean_rank": rank.float().mean()})
        #mean_rank = preds.shape[1] - torch.mean(preds.argsort().argsort().gather(1, idx.view(-1,1)).squeeze().float())

        ## Other Metrics
        self.log_dict(
            {
                f"{name}@{args['k']}": metric._score_func(
                    idx.detach().cpu(),
                    preds.squeeze().detach().cpu(),
                    labels=np.arange(preds.shape[1]),
                    **args,
                )
                for i, (name, metric, args) in enumerate(self.metrics)
            }
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer

    @property
    def name(self):
        return self.__class__.__name__
