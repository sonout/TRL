import os
import sys

sys.path.append("../..")

import json
import math
import time
from itertools import chain, combinations

import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.model_abtract import BaseModel
from models.utils import map_trajectory_to_road_embeddings
from pipelines.utils import ROOT_DIR, load_config
from .trajcl_files.dual_attention import DualSTB
#from .trajcl_files.moco import MoCo


class TrajCL(pl.LightningModule, BaseModel):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        self.model = TrajCLModel(config)

    def training_step(self, batch, batch_idx):
        trajs1_emb, trajs1_emb_p, trajs1_len, trajs2_emb, trajs2_emb_p, trajs2_len, _, _, _ = batch

        loss = self.model(trajs1_emb, trajs1_emb_p, trajs1_len, trajs2_emb, trajs2_emb_p, trajs2_len)
        #loss = self.model.loss(*model_rtn)

        self.log("train_loss", loss, logger=True)

        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        # After dataset implementation, we do collate there, so here we get already all outputs after collate
        #trajs_emb, trajs_emb_p, trajs1_len  = collate_for_test(X1, self.cellspace, self.embs)
        #trajs1_emb, trajs1_emb_p, trajs1_len, trajs2_emb, trajs2_emb_p, trajs2_len, X_orig, X_p_orig, X_len = batch
        trajs1_emb, trajs1_emb_p, trajs1_len, trajs2_emb, trajs2_emb_p, trajs2_len, trajs_emb, trajs_emb_p, trajs_len = batch

        trajs_emb = self.model.interpret(trajs_emb, trajs_emb_p, trajs_len)
        return trajs_emb

    def validation_step(self, batch, batch_idx):
        super().val_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        acc = super().on_val_end()
        self.log("val_acc", acc, logger=True, prog_bar=True, on_epoch=True)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.config["learning_rate"], weight_decay = 0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = self.config["trajcl_training_lr_degrade_step"], gamma = self.config["trajcl_training_lr_degrade_gamma"])
        return [optimizer], [scheduler]

    @property
    def name(self):
        return self.__class__.__name__




class TrajCLModel(nn.Module):

    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        encoder_q = DualSTB(config['emb_size'], 
                            config['trans_hidden_dim'], 
                            config['trans_attention_head'], 
                            config['trans_attention_layer'], 
                            config['trans_attention_dropout'], 
                            config['trans_pos_encoder_dropout'],)
        
        encoder_k = DualSTB(config['emb_size'], 
                            config['trans_hidden_dim'], 
                            config['trans_attention_head'], 
                            config['trans_attention_layer'], 
                            config['trans_attention_dropout'], 
                            config['trans_pos_encoder_dropout'],)

        moco_proj_dim = config['emb_size'] // 2

        self.clmodel = MoCo(encoder_q, encoder_k, 
                        config['emb_size'],
                        moco_proj_dim, 
                        config['moco_nqueue'],
                        temperature = config['moco_temperature'])



    def forward(self, trajs1_emb, trajs1_emb_p, trajs1_len, trajs2_emb, trajs2_emb_p, trajs2_len):
        # create kwargs inputs for TransformerEncoder
        
        max_trajs1_len = torch.arange(trajs1_len.max().item()).to(trajs1_len.get_device()) # in essense -- trajs1_len[0]
        max_trajs2_len = torch.arange(trajs2_len.max().item()).to(trajs1_len.get_device()) # in essense -- trajs2_len[0]

        src_padding_mask1 = max_trajs1_len[None, :] >= trajs1_len[:, None]
        src_padding_mask2 = max_trajs2_len[None, :] >= trajs2_len[:, None]
        
        loss = self.clmodel({'src': trajs1_emb, 'attn_mask': None, 'src_padding_mask': src_padding_mask1, 'src_len': trajs1_len, 'srcspatial': trajs1_emb_p},  
                {'src': trajs2_emb, 'attn_mask': None, 'src_padding_mask': src_padding_mask2, 'src_len': trajs2_len, 'srcspatial': trajs2_emb_p})
        return loss

    def interpret(self, trajs1_emb, trajs1_emb_p, trajs1_len):
        max_trajs1_len = trajs1_len.max().item() # trajs1_len[0]
        src_padding_mask1 = torch.arange(max_trajs1_len).to(trajs1_len.get_device())[None, :] >= trajs1_len[:, None]

        max_trajs1_len = torch.arange(trajs1_len.max().item()).to(trajs1_len.get_device()) # trajs1_len[0]
        src_padding_mask2 = max_trajs1_len[None, :] >= trajs1_len[:, None]

        traj_embs = self.clmodel.backbone(**{'src': trajs1_emb, 'attn_mask': None, 'src_padding_mask': src_padding_mask1, 'src_len': trajs1_len, 'srcspatial': trajs1_emb_p})
        return traj_embs


    def loss(self, logits, targets):
        return self.clmodel.loss(logits, targets)


    def load_checkpoint(self):
        return 

import copy
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.utils.scheduler import cosine_schedule



class MoCo(nn.Module):
    def __init__(self, encoder_q, encoder_k, nemb, nout,
                queue_size, mmt = 0.999, temperature = 0.07):
        super(MoCo, self).__init__()
        
        self.queue_size = queue_size
        self.mmt = mmt
        self.temperature = temperature

        self.backbone = encoder_q
        self.projection_head = MoCoProjectionHead(nemb, nemb, nout)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NTXentLoss(temperature=temperature, memory_bank_size=queue_size)

    def forward_normal(self, x):
        query = self.backbone(**x)#.flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(**x)#.flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

    def forward(self, kwargs_q, kwargs_k):
        current_epoch = 1
        momentum = cosine_schedule(current_epoch, 10, self.mmt, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        
        query = self.forward_normal(kwargs_q)
        key = self.forward_momentum(kwargs_k)
        loss = self.criterion(query, key)
        return loss
    
        