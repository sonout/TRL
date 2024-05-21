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
from pipelines.utils import ROOT_DIR
from models.proposed.traj_encoders import TransformerEncoder

from lightly.loss import NTXentLoss


class CLTSim(pl.LightningModule, BaseModel):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        
        self.hidden_size = self.config["emb_size"]
        self.n_layers = 1
        self.bidirectional = False

        # Backbone Encoder 
        self.encoder = LSTMEncoder(self.hidden_size, self.hidden_size, self.bidirectional, self.n_layers)
        # self.encoder = TransformerEncoder(config['emb_size'], nhead=4, nlayer=1)
        
        #self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss = NTXentLoss(temperature=0.05)
    

    def training_step(self, batch, batch_idx):
        trajs1_emb, trajs1_len, trajs2_emb, trajs2_len, _, _ = batch

        # We need to get here X, length
        # X1/X2: (batch_size, padded_length, feat_dim)
        z1 = self.encoder(trajs1_emb, trajs1_len)
        z2 = self.encoder(trajs2_emb, trajs2_len)
        #loss = self.contrastive_loss_simclr(z1, z2)
        loss = self.loss(z1, z2)

        self.log("train_loss", loss, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        super().val_step(batch, batch_idx)
        
    def on_validation_epoch_end(self):
        acc = super().on_val_end()
        self.log("val_acc", acc, logger=True, prog_bar=True, on_epoch=True)
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        trajs1_emb, trajs1_len, trajs2_emb, trajs2_len, trajs_emb, trajs_len = batch
        z = self.encoder(trajs_emb, trajs_len)
        return z
    
    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))

    def configure_optimizers(self):
        self.optim = torch.optim.Adam(
            self.encoder.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"]
        )
        return self.optim

    #def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        #return optimizer

    @property
    def name(self):
        return self.__class__.__name__
    
    def contrastive_loss_simclr(self, z1, z2, similarity="inner", temperature=0.04):
        """

        Args:
            z1(torch.tensor): (batch_size, d_model)
            z2(torch.tensor): (batch_size, d_model)

        Returns:

        """
        assert z1.shape == z2.shape
        batch_size, d_model = z1.shape
        features = torch.cat([z1, z2], dim=0)  # (batch_size * 2, d_model)

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)
        if similarity == 'inner':
            similarity_matrix = torch.matmul(features, features.T)
        elif similarity == 'cosine':
            similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        else:
            similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # [batch_size * 2, 1]

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # [batch_size * 2, 2N-2]

        logits = torch.cat([positives, negatives], dim=1)  # (batch_size * 2, batch_size * 2 - 1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)  # (batch_size * 2, 1)
        logits = logits / temperature

        loss_res = self.criterion(logits, labels)
        return loss_res


    

    


class LSTMEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, bidirectional, n_layers, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first,
                            bidirectional=bidirectional, num_layers=n_layers)

    def forward(self, trajs_hidden, trajs_len):
        outputs, _ = self.lstm(trajs_hidden)
        hn = outputs[torch.arange(trajs_hidden.shape[0]), trajs_len-1]
        return hn


