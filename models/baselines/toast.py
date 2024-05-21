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


class Toast(pl.LightningModule, BaseModel):
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

        self.model = BertModel4Pretrain(self.config)
        self.model.init_token_embed(_road_emb)

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )

        self.criterion1 = nn.CrossEntropyLoss(reduction="none")
        self.criterion2 = nn.CrossEntropyLoss()

    def on_train_epoch_start(self) -> None:
        self.trainer.train_dataloader.dataset.gen_new_walks(num_walks=100)

    def training_step(self, batch, batch_idx):
        (
            traj_input,
            input_mask,
            masked_pos,
            length,
            is_traj,
            masked_tokens,
            masked_weights,
        ) = batch

        mask_lm_output, next_sent_output = self.model.forward(
            traj_input, input_mask, masked_pos, length
        )

        next_loss = self.criterion2(next_sent_output, is_traj.long())
        mask_loss = self.criterion1(mask_lm_output.transpose(1, 2), masked_tokens)

        mask_loss = (mask_loss * masked_weights.float()).mean()

        loss = next_loss + mask_loss

        self.log("train_mask_loss", mask_loss, logger=True)
        self.log("train_next_loss", next_loss, logger=True)
        self.log("train_loss", loss, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        super().val_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        acc = super().on_val_end()
        self.log("val_acc", acc, logger=True, prog_bar=True, on_epoch=True)
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        (
            traj_input,
            input_mask,
            masked_pos,
            length,
            is_traj,
            masked_tokens,
            masked_weights,
        ) = batch

        h = self.model.transformer(traj_input, input_mask)  # B x S x D
        traj_h = torch.sum(h * input_mask.unsqueeze(-1).float(), dim=1) / length.float()

        return traj_h

    def load_model(self, path: str):
        ...

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer

    @property
    def name(self):
        return self.__class__.__name__


class BertModel4Pretrain(nn.Module):
    def __init__(self, cfg):
        super(BertModel4Pretrain, self).__init__()
        self.transformer = Transformer(cfg)
        self.fc = nn.Linear(cfg["hidden_size"], cfg["hidden_size"])
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(cfg["hidden_size"], cfg["hidden_size"])
        self.activ2 = gelu
        self.norm = LayerNorm(cfg)
        self.classifier = nn.Linear(cfg["hidden_size"], 2)

        embed_weight = self.transformer.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab)

    def init_token_embed(self, embed):
        token_vocab = self.transformer.embed.tok_embed.weight.shape[0]

        if embed.shape[0] < token_vocab:
            self.transformer.embed.tok_embed.weight.data[
                token_vocab - embed.shape[0] :
            ] = embed
            print(self.transformer.embed.tok_embed.weight.shape)
        else:
            self.transformer.embed.tok_embed.weight.data = embed

    def forward(self, input_ids, input_mask, masked_pos, traj_len):
        h = self.transformer(input_ids, input_mask)  # B x S x D
        # pooled_h = self.activ1(self.fc(h[:, 0]))
        traj_h = (
            torch.sum(h * input_mask.unsqueeze(-1).float(), dim=1) / traj_len.float()
        )
        pooled_h = self.activ1(self.fc(traj_h))

        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))  # B x S x D
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        # logits_lm = self.decoder(h_masked) + self.decoder_bias
        logits_lm = self.decoder(h_masked)
        logits_clsf = self.classifier(pooled_h)

        return logits_lm, logits_clsf


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    def __init__(self, cfg, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(cfg["hidden_size"]))
        self.beta = nn.Parameter(torch.zeros(cfg["hidden_size"]))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    def __init__(self, cfg):
        super(Embeddings, self).__init__()
        self.tok_embed = nn.Embedding(
            cfg["input_size"] + 2, cfg["hidden_size"]
        )  # token embeddind
        self.pos_embed = nn.Embedding(
            cfg["max_len"], cfg["hidden_size"]
        )  # position embedding
        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)  # (S,) -> (B, S)

        e = self.tok_embed(x)  # + self.pos_embed(pos)
        res = self.drop(self.norm(e))
        return res


def split_last(x, shape):
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, cfg):
        super(MultiHeadedSelfAttention, self).__init__()
        self.proj_q = nn.Linear(cfg["hidden_size"], cfg["hidden_size"])
        self.proj_k = nn.Linear(cfg["hidden_size"], cfg["hidden_size"])
        self.proj_v = nn.Linear(cfg["hidden_size"], cfg["hidden_size"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.scores = None  # for visualization
        self.n_heads = cfg["n_heads"]

    def forward(self, x, mask):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])

        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))

        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    def __init__(self, cfg):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(cfg["hidden_size"], cfg["hidden_size_ff"])
        self.fc2 = nn.Linear(cfg["hidden_size_ff"], cfg["hidden_size"])

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super(Block, self).__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg["hidden_size"], cfg["hidden_size"])
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.fc = nn.Linear(cfg["hidden_size"], cfg["hidden_size"])
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layers"])])

    def forward(self, x, mask):
        h = self.fc(self.embed(x))
        for block in self.blocks:
            h = block(h, mask)
        return h


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
