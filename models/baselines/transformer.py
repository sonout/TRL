import os
import sys

sys.path.append("../..")

import pytorch_lightning as pl
import torch

from models.model_abtract import BaseModel
from models.proposed.traj_encoders import TransformerEncoder
from models.proposed.contrastive_frameworks import SimCLR_asym



class Transformer(pl.LightningModule, BaseModel):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        self.encoder = TransformerEncoder(config['emb_size'], nhead=4, nlayer=1)

        proj_dim = config['emb_size'] // 2

        self.simclr = SimCLR_asym(self.encoder, self.encoder, 
                        config['emb_size'],
                        proj_dim, 
                        temperature = config['simclr_temperature'])



    def training_step(self, batch, batch_idx):
        road_emb_seq1, len1, road_emb_seq2, len2, orig, lengths = batch

        loss = self.simclr({'x': road_emb_seq1, 'lengths':len1},{'x': road_emb_seq2, 'lengths':len2})

        self.log("train_loss", loss, logger=True, prog_bar=True, on_step=True, on_epoch=True)

        return loss


    def validation_step(self, batch, batch_idx):
        super().val_step(batch, batch_idx)
        
    def on_validation_epoch_end(self):
        acc = super().on_val_end()
        self.log("val_acc", acc, logger=True, prog_bar=True, on_epoch=True)


    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        traj1, len1, traj2, len2, road_emb_seq, lengths = batch

        # generate trajectory embedding        
        z = self.simclr.encode1({'x': road_emb_seq, 'lengths':lengths})
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        # Scheduler?
        return optimizer

    @property
    def name(self):
        return self.__class__.__name__

    def load_model(self, path):
        self.load_state_dict(torch.load(path))







