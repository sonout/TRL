import os
import sys

sys.path.append("../..")

import json
import pytorch_lightning as pl
import torch
import torch.nn as nn
import itertools

from models.model_abtract import BaseModel
from pipelines.utils import ROOT_DIR
from .traj_enc_transformer import Transformer, MHA, precompute_freqs_cis
from .contrastive_frameworks import IntraInterContrastive

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

        proj_dim = 128 #emb_size // 2
        self.contrastive = IntraInterContrastive(self.model_road1, self.model_road2, self.model_cell,
                        self.road_emb1_size, self.road_emb2_size, self.cell_emb_size,
                        proj_dim, 
                        config['nqueue'],
                        temperature = config['temperature'])

        
        self.att_fusion = LMA(self.time_emb_size, loc_seq_len = 1)



    def training_step(self, batch, batch_idx):
        road1_trajs1_emb, road1_trajs1_len, road1_trajs2_emb, road1_trajs2_len, _, _, \
            road2_trajs1_emb, road2_trajs1_len, road2_trajs2_emb, road2_trajs2_len, _, _, \
                cell_trajs1_emb, cell_trajs1_len, cell_trajs2_emb, cell_trajs2_len, _, _, \
                time1_embs, time2_embs, _ = batch


        #road1_cat = torch.cat([road1_trajs1_emb, time1_embs], dim=-1)
        #road2_cat = torch.cat([road1_trajs2_emb, time2_embs], dim=-1)

        road1_cat = self.att_fusion(road1_trajs1_emb, time1_embs, road1_trajs1_len)
        road2_cat = self.att_fusion(road1_trajs2_emb, time2_embs, road1_trajs2_len)
        
        
        loss = self.contrastive({'x': road1_cat, 'lengths':road1_trajs1_len},
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
        # After dataset implementation, we do collate there, so here we get already all outputs after collate
        #trajs_emb, trajs_emb_p, trajs1_len  = collate_for_test(X1, self.cellspace, self.embs)
        #trajs1_emb, trajs1_emb_p, trajs1_len, trajs2_emb, trajs2_emb_p, trajs2_len, X_orig, X_p_orig, X_len = batch
        _, _, _, _, road1_trajs_emb, road1_trajs_len, \
            _, _, _, _, road2_trajs_emb, road2_trajs_len, \
            _, _, _, _, cell_trajs_emb, cell_trajs_len, \
                _, _, time_emb = batch

        road1_cat = self.att_fusion(road1_trajs_emb, time_emb, road1_trajs_len)

        z = self.contrastive.encode({'x': road1_cat, 'lengths':road1_trajs_len},{'x': road2_trajs_emb, 'lengths':road2_trajs_len}, {'x': cell_trajs_emb, 'lengths':cell_trajs_len})
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


class LMA(nn.Module):
    def __init__(self, dim, loc_seq_len = None, dropout = 0.1):
        super(LMA, self).__init__()

        if loc_seq_len is None:
            self.loc_seq_len = 1
        else:
            self.loc_seq_len = loc_seq_len
        self.Wq1 = nn.Linear(dim, dim, bias=False)
        self.Wk1 = nn.Linear(dim, dim, bias=False)
        self.Wv1 = nn.Linear(dim, dim, bias=False)
        self.Wq2 = nn.Linear(dim, dim, bias=False)
        self.Wk2 = nn.Linear(dim, dim, bias=False)
        self.Wv2 = nn.Linear(dim, dim, bias=False)

        self.dropout = dropout
        self.FFN1 = nn.Sequential(
            nn.Linear(dim, int(dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(dim*0.5), dim),
            nn.Dropout(0.1)
        )
        self.FFN2 = nn.Sequential(
            nn.Linear(dim, int(dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(dim*0.5), dim),
            nn.Dropout(0.1)
        )
        self.layer_norm = nn.LayerNorm(dim*2, eps=1e-6)

    def forward(self, seq_s, seq_t, len): # seq_s/seq_t shape [N, L, D]
        N, L, D = seq_s.size()
        q1 = self.Wq1(seq_s)
        k1 = self.Wk1(seq_t)
        v1 = self.Wv1(seq_t)

        assert L % self.loc_seq_len == 0, f"Sequence Length {L} should be divisible by loc_seq_len {self.loc_seq_len}"
        n_heads = L // self.loc_seq_len #5

        q1 = q1.view(N, n_heads, self.loc_seq_len, D) # [N, Heads, L_loc, D]
        k1 = k1.view(N, n_heads, self.loc_seq_len, D) # [N, Heads, L_loc, D]
        v1 = v1.view(N, n_heads, self.loc_seq_len, D) # [N, Heads, L_loc, D]

        output1 = torch.nn.functional.scaled_dot_product_attention(q1, k1, v1, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=False)

        # restore orig shape
        output1 = output1.view(N, L, D)
        output1 = self.FFN1(output1) + output1

        q2 = self.Wq2(seq_t)
        k2 = self.Wk2(seq_s)
        v2 = self.Wv2(seq_s)

        q2 = q2.view(N, n_heads, self.loc_seq_len, D) # [N, Heads, L_loc, D]
        k2 = k2.view(N, n_heads, self.loc_seq_len, D)
        v2 = v2.view(N, n_heads, self.loc_seq_len, D)

        output2 = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        # restore orig shape
        output2 = output2.view(N, L, D)
        output2 = self.FFN2(output2) + output2

        out = torch.cat([output1, output2], dim=-1) # [N, L, 2*D]
        out = self.layer_norm(out)

        return out


    

