import os
import sys

sys.path.append("../..")

import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


from models.model_abtract import BaseModel
from pipelines.utils import ROOT_DIR
from .traj_enc_transformer import Transformer, MHA, precompute_freqs_cis
from .contrastive_frameworks import IntraInterContrastive
from models.token_embs.time.time2vec import Time2Vec

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
        self.moco = IntraInterContrastive(self.model_road1, self.model_road2, self.model_cell,
                        self.road_emb1_size, self.road_emb2_size, self.cell_emb_size,
                        proj_dim, 
                        config['nqueue'],
                        temperature = config['temperature'])

        self.time2vec = Time2Vec(k = self.time_emb_size, act = "cos", in_feats = 4)
        state_dict_path = os.path.join(ROOT_DIR, config["time2vec_path"])
        state_dict = torch.load(state_dict_path, map_location=self.device)
        self.time2vec.load_state_dict(state_dict, strict=False)

        self.att_fusion = LMA(self.time_emb_size, loc_seq_len = config["lma_seq_len"])


    def training_step(self, batch, batch_idx):
        road1_trajs1_emb, road1_trajs1_len, road1_trajs2_emb, road1_trajs2_len, _, _, \
            road2_trajs1_emb, road2_trajs1_len, road2_trajs2_emb, road2_trajs2_len, _, _, \
                cell_trajs1_emb, cell_trajs1_len, cell_trajs2_emb, cell_trajs2_len, _, _, \
                time1_feats, time2_feats, _ = batch

        # encode time
        time1_embs = self.time2vec.encode(time1_feats)
        time2_embs = self.time2vec.encode(time2_feats)


        road1_cat = self.att_fusion(road1_trajs1_emb, time1_embs, road1_trajs1_len)
        road2_cat = self.att_fusion(road1_trajs2_emb, time2_embs, road1_trajs2_len)
        
        
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
        
        # encode time
        time_emb = self.time2vec.encode(time_emb)

        road1_cat = self.att_fusion(road1_trajs_emb, time_emb, road1_trajs_len)

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



class LMA(nn.Module):
    def __init__(self, dim, loc_seq_len = 1, dropout = 0.1):
        super(LMA, self).__init__()

        # Ensure loc_seq_len is at least 1
        self.loc_seq_len = max(1, loc_seq_len if loc_seq_len is not None else 1)

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

    def forward(self, seq_s, seq_t, seq_lengths): # seq_s/seq_t shape [N, L, D]
        N, L_orig, D = seq_s.size()
        _N_t, _L_t, _D_t = seq_t.size()

        # Basic input validation
        assert L_orig == _L_t and D == _D_t, "Input sequences seq_s and seq_t must have same L and D dimensions."
        assert N == len(seq_lengths), "Batch size mismatch between input tensors and seq_lengths."
        assert N == _N_t, "Batch size mismatch between seq_s and seq_t."

        pad_amount = 0
        if L_orig % self.loc_seq_len != 0:
            target_len = math.ceil(L_orig / self.loc_seq_len) * self.loc_seq_len
            pad_amount = target_len - L_orig
            L_padded = target_len
        else:
            L_padded = L_orig # No padding needed

        if pad_amount > 0:
            # Pad tensors on the right of the sequence dimension (dim 1)
            # Pad format: (pad_left, pad_right, pad_top, pad_bottom, ...)
            seq_s_padded = F.pad(seq_s, (0, 0, 0, pad_amount), mode='constant', value=0.0)
            seq_t_padded = F.pad(seq_t, (0, 0, 0, pad_amount), mode='constant', value=0.0)
        else:
            seq_s_padded = seq_s
            seq_t_padded = seq_t
        seq_s = seq_s_padded
        seq_t = seq_t_padded

        ##########
        q1 = self.Wq1(seq_s)
        k1 = self.Wk1(seq_t)
        v1 = self.Wv1(seq_t)

        assert L_padded % self.loc_seq_len == 0, f"Sequence Length {L_padded} should be divisible by loc_seq_len {self.loc_seq_len}"
        n_heads = L_padded // self.loc_seq_len #5

        q1 = q1.view(N, n_heads, self.loc_seq_len, D) # [N, Heads, L_loc, D]
        k1 = k1.view(N, n_heads, self.loc_seq_len, D) # [N, Heads, L_loc, D]
        v1 = v1.view(N, n_heads, self.loc_seq_len, D) # [N, Heads, L_loc, D]

        output1 = torch.nn.functional.scaled_dot_product_attention(q1, k1, v1, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=False)

        # restore orig shape
        output1 = output1.reshape(N, L_padded, D)
        output1 = self.FFN1(output1) + output1

        q2 = self.Wq2(seq_t)
        k2 = self.Wk2(seq_s)
        v2 = self.Wv2(seq_s)

        q2 = q2.view(N, n_heads, self.loc_seq_len, D) # [N, Heads, L_loc, D]
        k2 = k2.view(N, n_heads, self.loc_seq_len, D)
        v2 = v2.view(N, n_heads, self.loc_seq_len, D)

        output2 = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        # restore orig shape
        output2 = output2.reshape(N, L_padded, D)
        output2 = self.FFN2(output2) + output2

        out = torch.cat([output1, output2], dim=-1) # [N, L_padded, 2*D]
        out = self.layer_norm(out[:, :L_orig, :]) # [N, L_orig, 2*D]

        return out
