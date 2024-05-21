import os
import sys

sys.path.append("../..")

import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.model_abtract import BaseModel
from .traj_enc_transformer import Transformer
from .contrastive_frameworks import MoCo_fusion3

class TIGR(pl.LightningModule, BaseModel):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        self.road_emb1_size = config['road_emb1_size']
        self.road_emb2_size = config['road_emb2_size']
        self.cell_emb_size = config['cell_emb_size']
        self.time_emb_size = config['time_emb_size']
        self.road_emb1_size = self.road_emb1_size + self.time_emb_size # We use them together

        self.model_road1 = Transformer(self.road_emb1_size, nlayer=config['n_layers']) 
        self.model_road2 = Transformer(self.road_emb2_size, nlayer=config['n_layers'])
        self.model_cell = Transformer(self.cell_emb_size, nlayer=config['n_layers'])

        moco_proj_dim = 128 #emb_size // 2
        self.moco = MoCo_fusion3(self.model_road1, self.model_road2, self.model_cell,
                        self.road_emb1_size, self.road_emb2_size, self.cell_emb_size,
                        moco_proj_dim, 
                        config['moco_nqueue'],
                        temperature = config['moco_temperature'])
        
        # Cross Attention of Traffic and Time
        self.n_head = 4
        self.cross_att1 = nn.MultiheadAttention(embed_dim=config['road_emb1_size'], num_heads=self.n_head)
        self.cross_att2 = nn.MultiheadAttention(embed_dim=config['time_emb_size'], num_heads=self.n_head)


    def training_step(self, batch, batch_idx):
        road1_trajs1_emb, road1_trajs1_len, road1_trajs2_emb, road1_trajs2_len, _, _, \
            road2_trajs1_emb, road2_trajs1_len, road2_trajs2_emb, road2_trajs2_len, _, _, \
                cell_trajs1_emb, cell_trajs1_len, cell_trajs2_emb, cell_trajs2_len, _, _, \
                time1_embs, time2_embs, _ = batch

        road1_cat = torch.cat([road1_trajs1_emb, time1_embs], dim=-1)

        road2_cat = torch.cat([road1_trajs2_emb, time2_embs], dim=-1)
        
        
        loss = self.moco({'x': road1_cat, 'lengths':road1_trajs1_len},
                         {'x': road2_cat, 'lengths':road1_trajs2_len},
                         {'x': road2_trajs1_emb, 'lengths':road2_trajs1_len},
                         {'x': road2_trajs2_emb, 'lengths':road2_trajs2_len},
                         {'x': cell_trajs1_emb, 'lengths': cell_trajs1_len},
                         {'x': cell_trajs2_emb, 'lengths': cell_trajs2_len})

        self.log("train_loss", loss, logger=True, prog_bar=True, on_step=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        super().val_step(batch, batch_idx)
        
    def on_validation_epoch_end(self):
        acc = super().on_val_end()
        self.log("val_acc", acc, logger=True, prog_bar=True, on_epoch=True)

    
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        _, _, _, _, road1_trajs_emb, road1_trajs_len, \
            _, _, _, _, road2_trajs_emb, road2_trajs_len, \
            _, _, _, _, cell_trajs_emb, cell_trajs_len, \
                _, _, time_emb = batch
        
        road1_cat = torch.cat([road1_trajs_emb, time_emb], dim=-1)

        z = self.moco.encode({'x': road1_cat, 'lengths':road1_trajs_len},{'x': road2_trajs_emb, 'lengths':road2_trajs_len}, {'x': cell_trajs_emb, 'lengths':cell_trajs_len})
        return z

    
    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.config["learning_rate"], weight_decay = 0.0001)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = self.config["trajcl_training_lr_degrade_step"], gamma = self.config["trajcl_training_lr_degrade_gamma"])
        return [optimizer]#, [scheduler]

    @property
    def name(self):
        return self.__class__.__name__







