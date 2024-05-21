import os
import sys

sys.path.append("../..")

import json
import time
from itertools import chain, combinations

import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from scipy.sparse.csgraph import shortest_path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.model_abtract import BaseModel
from models.utils import map_trajectory_to_road_embeddings
from pipelines.utils import ROOT_DIR

from .toast import SkipGramToast


class Trembr(pl.LightningModule, BaseModel):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        # define road segment embedder
        _road_encoder = SkipGramToast(
            self.config["input_size"], self.config["road_emb_size"]
        )
        _road_encoder.load_model(
            path=os.path.join(ROOT_DIR, self.config["road_emb_model_path"])
        )

        _road_emb = _road_encoder.load_emb()

        self.embedding_layer = nn.Embedding(_road_emb.shape[0], _road_emb.shape[1])
        self.embedding_layer = self.embedding_layer.from_pretrained(_road_emb, freeze=True)


        # define encoder and decoder architecture
        self.encoder = LSTMEncoder1(self.config)
        
        self._decoder = nn.LSTM(
            input_size=self.config["road_emb_size"],
            hidden_size=self.config["emb_size"],
            num_layers=1,
            batch_first=True,
        )
        self._road_decoder = nn.Sequential(
            nn.Linear(self.config["emb_size"], self.config["input_size"]),
            #nn.Softmax(dim=1),
        )
        self._time_decoder = nn.Sequential(
            nn.Linear(self.config["emb_size"], int(self.config["emb_size"] / 2)),
            nn.ReLU(),
            nn.Linear(int(self.config["emb_size"] / 2), 1),
        )

        #self.model = BertModel4Pretrain(self.config)
        #self.model.init_token_embed(_road_emb)
        self.loss_road = torch.nn.CrossEntropyLoss(reduction='mean')


    def encode(self, x, lengths):
        x, (hs, cs) = self.encoder(x, lengths)

        return x, (hs, cs)

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
        time_pred = self._time_decoder(dx.squeeze())

        return road_pred, time_pred, hs

    def training_step(self, batch, batch_idx):
        X, lengths = batch
        # We need to get here X, length
        # X1/X2: (batch_size, padded_length, feat_dim)
        road_emb_seq = self.embedding_layer(X)
        decoder_road_emb = self.embedding_layer(X)

        z, states = self.encode(road_emb_seq, lengths)

        loss = 0
        for t in range(road_emb_seq.shape[1] - 1):
            out_r, out_t, states = self.decode(
                states, decoder_road_emb[:, t, :].unsqueeze(1)
            )
            loss += self.loss_road(out_r, X[:, t + 1])        

        self.log("train_loss", loss, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        super().val_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        acc = super().on_val_end()
        self.log("val_acc", acc, logger=True, prog_bar=True, on_epoch=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        X, lengths = batch
        # map trajectory to embeddings and append timestamp
        road_emb_seq = self.embedding_layer(X)
        # road_emb_seq = torch.cat((road_emb_seq, yt.unsqueeze(-1)), dim=-1)

        # generate trajectory embedding
        z, (hs, cs) = self.encode(road_emb_seq, lengths)

        return z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer

    @property
    def name(self):
        return self.__class__.__name__

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


class LSTMEncoder1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self._encoder = nn.LSTM(
                input_size=self.config["road_emb_size"],
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



class SkipGramToast(nn.Module):
    """
    Model for road segment embeddings (uses trained embeddings from master thesis)
    """

    def __init__(self, vocab_size, emb_dim, type_num=13, type_dim=32):
        super(SkipGramToast, self).__init__()
        self.input_emb = nn.Embedding(vocab_size, emb_dim)
        self.output_emb = nn.Embedding(vocab_size, emb_dim + type_num)
        self.type_pred = nn.Linear(emb_dim, type_num, bias=False)
        self.log_sigmoid = nn.LogSigmoid()

        initrange = (2.0 / (vocab_size + emb_dim)) ** 0.5  # Xavier init
        self.input_emb.weight.data.uniform_(-initrange, initrange)
        self.output_emb.weight.data.uniform_(-0, 0)

    def forward(self, target_input, type_input, context, types, neg, type_mask):

        v = self.input_emb(target_input)
        u = self.output_emb(context)

        type_pred = self.type_pred(v)
        type_loss = F.binary_cross_entropy_with_logits(
            type_pred, types, weight=type_mask
        )

        # positive_val: [batch_size]
        v_cat = torch.cat((v, torch.sigmoid(type_pred)), dim=1)
        positive_val = self.log_sigmoid(torch.sum(u * v_cat, dim=1)).squeeze()

        u_hat = self.output_emb(neg)
        neg_vals = torch.bmm(u_hat, v_cat.unsqueeze(2)).squeeze(2)
        neg_val = self.log_sigmoid(-torch.sum(neg_vals, dim=1)).squeeze()

        loss = positive_val + neg_val

        return -loss.mean(), type_loss

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def load_emb(self):
        return self.input_emb.weight.data


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
        # trajs_hidden: batch_size * 2 * n_views, seq_len, hidden_size
        # trajs_len: batch_size * 2 * n_views
        #packed_trajs_hidden = pack_padded_sequence(trajs_hidden, trajs_len.detach().cpu(), batch_first=True, enforce_sorted=False)
        # hn: num_layers * n_direction, batch_size * 2 * n_views, hidden_size
        #_, (hn, _) = self.lstm(packed_trajs_hidden)
        #outputs, _ = self.lstm(packed_trajs_hidden)
        outputs, _ = self.lstm(trajs_hidden)
        #hn = hn.transpose(0, 1).reshape(trajs_hidden.shape[0], -1)
        # outputs: batch_size * 2 * n_views, seq_len, hidden_size * n_direction
        # outputs, _ = self.lstm(trajs_hidden)
        # hn: batch_size * 2 * n_views, hidden_size * n_direction
        hn = outputs[torch.arange(trajs_hidden.shape[0]), trajs_len-1]
        #unpacked_output, hn = pad_packed_sequence(packed_output, batch_first=True)
        #return hn.transpose(0, 1).reshape(trajs_hidden.shape[0], -1)
        return hn