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
import pickle

from models.model_abtract import BaseModel
from models.utils import map_trajectory_to_road_embeddings
from pipelines.utils import ROOT_DIR


class T2Vec(pl.LightningModule, BaseModel):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        
        cell_embs_filepath = os.path.join(ROOT_DIR, config["model_files_path"], config["embs_file"])
        _cell_emb = pickle.load(open(cell_embs_filepath, 'rb')).to('cpu').detach() # tensor

        self.embedding_layer = nn.Embedding(_cell_emb.shape[0], _cell_emb.shape[1])
        self.embedding_layer = self.embedding_layer.from_pretrained(_cell_emb, freeze=True)


        # define encoder and decoder architecture
        self.encoder = LSTMEncoder1(self.config)
        
        self._decoder = nn.LSTM(
            input_size=self.config["cell_embedding_dim"],
            hidden_size=self.config["emb_size"],
            num_layers=1,
            batch_first=True,
        )
        self._road_decoder = nn.Sequential(
            nn.Linear(self.config["emb_size"], _cell_emb.shape[0]),
            #nn.Softmax(dim=1),
        )

        #self.model = BertModel4Pretrain(self.config)
        #self.model.init_token_embed(_road_emb)
        self.loss_road = torch.nn.CrossEntropyLoss(reduction='mean')
    
    def decode(self, x, dx):
        # decoding step
        # dx_lengths = lengths
        # dx = torch.nn.utils.rnn.pack_padded_sequence(
        #     dx, dx_lengths.detach().cpu(), batch_first=True
        # )

        # hstate = x.unsqueeze(0)
        # cstate = torch.zeros_like(hstate)
        dx, hs = self._decoder(dx, x)

        # dx, plengths = torch.nn.utils.rnn.pad_packed_sequence(
        #     dx, batch_first=True, padding_value=0
        # )
        # dx = dx.contiguous()
        road_pred = self._road_decoder(dx.squeeze())

        return road_pred, hs
    
    def training_step(self, batch, batch_idx):
        traj1, len1, traj2, len2, orig, lengths = batch
        # We need to get here X, length
        # X1/X2: (batch_size, padded_length, feat_dim)
        road_emb_seq = self.embedding_layer(orig)
        decoder_road_emb = self.embedding_layer(orig)

        z, states = self.encoder(road_emb_seq, lengths)

        loss = 0
        seq_len = min(road_emb_seq.shape[1], decoder_road_emb.shape[1])
        for t in range(seq_len - 1):
            # out_r, states = self.decode(
            #     states, decoder_road_emb[:, t, :].unsqueeze(1)
            # )
            dx, _ = self._decoder(decoder_road_emb[:, t, :].unsqueeze(1), states)
            out_r = self._road_decoder(dx.squeeze())
            loss += self.loss_road(out_r, orig[:, t + 1])
        

        self.log("train_loss", loss, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        super().val_step(batch, batch_idx)
        
    def on_validation_epoch_end(self):
        acc = super().on_val_end()
        self.log("val_acc", acc, logger=True, prog_bar=True, on_epoch=True)
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        traj1, len1, traj2, len2, orig, lengths = batch
        road_emb_seq = self.embedding_layer(orig)
        z, (hs, cs) = self.encoder(road_emb_seq, lengths)
        return z
    
    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer

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


class LSTMEncoder1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self._encoder = nn.LSTM(
                input_size=self.config["cell_embedding_dim"],
                hidden_size=self.config["emb_size"],
                num_layers=1,
                batch_first=True,
            )
    
    def forward(self, x, lengths):
        batch_size, seq_len, _ = x.size()
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )

        x, (hs, cs) = self._encoder(x)

        x, plengths = torch.nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True, padding_value=0
        )
        x = x.contiguous()

        x = torch.stack(
            [x[b, plengths[b] - 1] for b in range(batch_size)]
        )  # get last valid item per batch batch x hidden

        return x, (hs, cs)