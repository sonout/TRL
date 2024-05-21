import os
import sys

sys.path.append("../..")

import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from scipy.sparse.csgraph import shortest_path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from lightly.loss import NTXentLoss

from models.model_abtract import BaseModel
from models.utils import (contrastive_loss_simclr,
                          map_trajectory_to_road_embeddings)
from pipelines.utils import ROOT_DIR

from ._start_util_models import BERTContrastiveLM


class Start(pl.LightningModule, BaseModel):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        self.mlm_ratio = self.config.get("mlm_ratio", 1.0)
        self.contra_ratio = self.config.get("contra_ratio", 1.0)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.data_argument1 = self.config.get("data_argument1", ["shuffle_position"])
        self.data_argument2 = self.config.get("data_argument2", ["shuffle_position"])

        self.align_w = self.config.get("align_w", 1.0)
        self.unif_w = self.config.get("unif_w", 1.0)
        self.align_alpha = self.config.get("align_alpha", 2)
        self.unif_t = self.config.get("unif_t", 2)
        self.train_align_uniform = self.config.get("train_align_uniform", False)
        self.test_align_uniform = self.config.get("test_align_uniform", True)
        self.norm_align_uniform = self.config.get("norm_align_uniform", False)
        self.clip_grad_norm = self.config.get("clip_grad_norm", True)
        self.max_grad_norm = self.config.get("max_grad_norm", 5)

        self.data_features = {
            "vocab_size": self.config["input_size"],
            "node_fea_dim": self.config["road_features"],
        }

        self.model = BERTContrastiveLM(self.config, self.data_features)
        self.criterion_mask = torch.nn.NLLLoss(ignore_index=0, reduction="none")
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.criterion_lightly = NTXentLoss()

    def training_step(self, batch, batch_idx):
        contra_view1, contra_view2, padding_masks1, padding_masks2, batch_temporal_mat1, batch_temporal_mat2, \
            X, targets, target_masks, padding_masks, batch_temporal_mat, graph_dict = batch
        

        z1, z2, predictions_l = self.model(contra_view1=contra_view1, contra_view2=contra_view2,
                                               argument_methods1=self.data_argument1,
                                               argument_methods2=self.data_argument2,
                                               masked_input=X, padding_masks=padding_masks,
                                               batch_temporal_mat=batch_temporal_mat,
                                               padding_masks1=padding_masks1,
                                               batch_temporal_mat1=batch_temporal_mat1,
                                               padding_masks2=padding_masks2,
                                               batch_temporal_mat2=batch_temporal_mat2,
                                               graph_dict=graph_dict[0]) # with or without [0]?

        targets_l, target_masks_l = targets[..., 0], target_masks[..., 0]
        mean_loss_l, batch_loss_l, num_active_l = self._cal_loss(predictions_l, targets_l, target_masks_l)
        
        logits, labels = contrastive_loss_simclr(z1, z2)
        mean_loss_con = self.criterion(logits, labels)
        #mean_loss_con = self.criterion_lightly(z1, z2)

        mean_loss = self.mlm_ratio * mean_loss_l + self.contra_ratio * mean_loss_con

        self.log("train_loss", mean_loss, logger=True)

        ## Other metrics/losses ##
        #align_uniform_loss, align_loss, uniform_loss = self.align_uniform(z1, z2)
        with torch.no_grad():
            total_correct_l = self._cal_acc(predictions_l, targets_l, target_masks_l)
            total_active_elements_l = num_active_l.item()

        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        ## Logging
        self.log("lr", self.optimizers().param_groups[0]['lr'], logger=True, prog_bar=True)
        self.log("Loc acc(%)", total_correct_l / total_active_elements_l * 100, logger=True, prog_bar=True)
        self.log("MLM loss", mean_loss_l.item(), logger=True, prog_bar=True)
        self.log("Contr. loss", mean_loss_con.item(), logger=True, prog_bar=True)
        #self.log("align_loss", align_loss, logger=True, prog_bar=True)

        # self.lr_scheduler.step_update(num_updates=batches_seen)

        return mean_loss

    def validation_step(self, batch, batch_idx):
        super().val_step(batch, batch_idx)
        
    def on_validation_epoch_end(self):
        acc = super().on_val_end()
        self.log("val_acc", acc, logger=True, prog_bar=True, on_epoch=True)
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        _, _, _, _, _, _, X, targets, _, padding_masks, batch_temporal_mat, graph_dict = batch

        token_emb, hidden_states, _ = self.model.bert(
            x=X,
            padding_masks=padding_masks,
            batch_temporal_mat=batch_temporal_mat,
            graph_dict=graph_dict[0],
            output_hidden_states=False,
            output_attentions=False,
        ) # (B, T, d_model)

        traj_emb = self.model.pooler(
            bert_output=token_emb,
            padding_masks=padding_masks,
            hidden_states=hidden_states,
        )  # (B, d_model)

        # input_mask_expanded = (
        #     padding_masks.unsqueeze(-1).expand(token_emb.size()).float()
        # )  # (batch_size, seq_length, d_model)
        # sum_embeddings = torch.sum(token_emb * input_mask_expanded, 1)
        # sum_mask = input_mask_expanded.sum(1)
        # sum_mask = torch.clamp(sum_mask, min=1e-9)

        return traj_emb #sum_embeddings / sum_mask

    def _cal_loss(self, pred, targets, targets_mask):
        batch_loss_list = self.criterion_mask(pred.transpose(1, 2), targets)
        batch_loss = torch.sum(batch_loss_list)
        num_active = targets_mask.sum()
        mean_loss = (
            batch_loss / num_active
        )  # mean loss (over samples) used for optimization
        return mean_loss, batch_loss, num_active

    def _cal_acc(self, pred, targets, targets_mask):
        mask_label = targets[targets_mask]  # (num_active, )
        lm_output = pred[targets_mask].argmax(dim=-1)  # (num_active, )
        correct_l = mask_label.eq(lm_output).sum().item()
        return correct_l
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer

    @property
    def name(self):
        return self.__class__.__name__

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
