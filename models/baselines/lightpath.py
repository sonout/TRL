import os
import sys

sys.path.append("../..")

import pickle
import math
import pytorch_lightning as pl
import torch
from torch import nn

from models.model_abtract import BaseModel
from models.proposed.traj_encoders import TransformerLightpath, _PositionalEncoding

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from pipelines.utils import ROOT_DIR

class LightPath(pl.LightningModule, BaseModel):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        self.compress_ratio = config["compress_ratio"] # Set to 0.6
        self.seq_len = config["seq_len"]
        embed_dim = self.config['emb_size']
        decoder_embed_dim = self.config['decoder_emb_size']
        self.train_type = config["train_type"] # 0: MAE, 1: Contrastive, 2: Both

        #### ROAD EMBEDDINGS ####
        _road_emb = pickle.load(open(os.path.join(ROOT_DIR, self.config["road_emb_path"]), 'rb'))
        self.embedding_layer = nn.Embedding(_road_emb.shape[0], _road_emb.shape[1])
        self.embedding_layer = self.embedding_layer.from_pretrained(_road_emb, freeze=True)

        ###### MAE Encoder ######
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = self.get_positional_encoding(emb_size=embed_dim, maxlen=self.seq_len + 1)

        # Transformer Encoder
        self.encoder = TransformerLightpath(ninput=embed_dim, nhidden=config["nhidden"], nhead=config["nhead"], nlayer=config["nlayer"], attn_dropout=config["droppout"])

        ###### MAE Decoder ######
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = self.get_positional_encoding(emb_size=decoder_embed_dim, maxlen=self.seq_len + 1)


        self.decoder = TransformerLightpath(ninput=decoder_embed_dim, nhidden=config["nhidden"], nhead=config["nhead"], nlayer=config["nlayer"], attn_dropout=config["droppout"])

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, _road_emb.shape[1], bias=True) # decoder to traj emb


        ###### Contrastive Framework ######
        moco_proj_dim = embed_dim // 2
        self.moco = MoCo(self.encoder, self.encoder, 
                        config['emb_size'],
                        moco_proj_dim, 
                        config['moco_nqueue'],
                        temperature = config['moco_temperature'])


    def training_step(self, batch, batch_idx):
        x, lengths, x1, x1_len, x2, x2_len  = batch

        mae_loss = 0
        contr_loss = 0
        if self.train_type == 0 or self.train_type == 2:
            # MAE Training
            # Embedd traj
            traj = self.embedding_layer(x)

            x, mask, ids_restore = self.forward_mask(traj, lengths, self.compress_ratio) # Adds CLS & PE as well
            latent = self.encoder(x, lengths)
            pred = self.forward_decoder(latent, lengths, ids_restore) 
            mae_loss = self.forward_loss(traj, pred, mask)
            self.log("mae_loss", mae_loss, logger=True, prog_bar=True, on_step=True, on_epoch=True)
        if self.train_type == 1 or self.train_type == 2:
            # Contrastive Training
            traj1 = self.embedding_layer(x1)
            traj2 = self.embedding_layer(x2)
            contr_loss = self.moco({'x': traj1, 'lengths':x1_len},{'x': traj2, 'lengths':x2_len})
            self.log("contr_loss", contr_loss, logger=True, prog_bar=True, on_step=True, on_epoch=True)


        loss = 1 * mae_loss + contr_loss
        return loss


    def validation_step(self, batch, batch_idx):
        super().val_step(batch, batch_idx)
        
    def on_validation_epoch_end(self):
        acc = super().on_val_end()
        self.log("val_acc", acc, logger=True, prog_bar=True, on_epoch=True)


    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, lengths, x1, x1_len, x2, x2_len  = batch

        # Embedd traj
        x = self.embedding_layer(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:x.shape[1]+1, :].to(x.device)
        lengths = lengths + 1

        # masking: length -> length * mask_ratio
        #x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :].to(x.device)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # generate trajectory embedding        
        z = self.encoder(x, lengths)[:, 0, :]
        #z = self.moco.encode({'x': x, 'lengths':lengths})
        return z # Return only CLS token

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        # Scheduler?
        return optimizer

    @property
    def name(self):
        return self.__class__.__name__

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


    
    def forward_mask(self, x, lengths, mask_ratio):

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:x.shape[1]+1, :].to(x.device)
        lengths = lengths + 1

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :].to(x.device)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        return x, mask, ids_restore


    def forward_decoder(self, x, lengths, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # we have CLS so length + 1
        lengths = lengths + 1

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed[:, :x.shape[1], :].to(x.device)

        # apply Transformer blocks
        x = self.decoder(x, lengths)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


    def forward_loss(self, traj, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = traj

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def get_positional_encoding(self, emb_size: int, maxlen: int = 201):
        den = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000)) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        return pos_embedding.transpose(0, 1)
    



import copy
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
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
        
        query = self.forward_normal(kwargs_q)[:, 0, :]
        key = self.forward_momentum(kwargs_k)[:, 0, :]
        loss = self.criterion(query, key)
        return loss
    
    def encode(self, x):
        return self.backbone(**x)[:, 0, :]