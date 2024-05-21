import os
import sys

sys.path.append("../..")

import math
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import networkx as nx

from models.model_abtract import BaseModel
from pipelines.utils import ROOT_DIR, load_road_network
from models.baselines.jclm_files.jcml_utils import node_node_loss, seq_seq_loss, node_seq_loss


class JCLM(pl.LightningModule, BaseModel):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        # Create PYG Edge Index
        edge_df, _, _, LG = load_road_network(city_name=config["city"])
        map_id = {j: i for i, j in enumerate(LG.nodes)}
        edge_list = nx.to_pandas_edgelist(LG)
        edge_list["sidx"] = edge_list["source"].map(map_id)
        edge_list["tidx"] = edge_list["target"].map(map_id)

        edge_index = np.array(edge_list[["sidx", "tidx"]].values).T
        edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
        pyg_data = Data(edge_index=edge_index)

        # transition matrix
        self.trans_mx = np.load(os.path.join(ROOT_DIR,config["trans_mx_path"]))
        # Preprocess trans_mx
        trans_mat_b = (self.trans_mx > 0.6) 
        aug_edges = [(i // self.trans_mx.shape[0] , i % self.trans_mx.shape[0]) for i, n in enumerate(trans_mat_b.flatten()) if n]
        self.aug_edge_index = torch.tensor(np.array(aug_edges).transpose())

        self.vocab_size = len(edge_df)  

        graph_encoder1 = GraphEncoder(config["emb_size"], config["hidden_size"], GATConv, 2, nn.ReLU())
        graph_encoder2 = GraphEncoder(config["emb_size"], config["hidden_size"], GATConv, 2, nn.ReLU())
        seq_encoder = TransformerModel(config["hidden_size"], 4, config["hidden_size"], 2, 0.2)

        self.model = CLMEncoder(
            vocab_size=self.vocab_size,
            embed_size=config["emb_size"],
            hidden_size=config["hidden_size"],
            edge_index1=pyg_data.edge_index,
            edge_index2=self.aug_edge_index,
            graph_encoder1=graph_encoder1,
            graph_encoder2=graph_encoder2,
            seq_encoder=seq_encoder,
        )
        self.model
        #self.model_parallel = nn.DataParallel(self.model)

        self.l_st = 0.8
        self.l_ss = self.l_tt = 0.5 * (1 - self.l_st)



    def training_step(self, batch, batch_idx):

        node_rep1, node_rep2, seq_rep1, seq_rep2 = self.model(batch)

        loss_ss = node_node_loss(node_rep1, node_rep2, "jsd")
        loss_tt = seq_seq_loss(seq_rep1, seq_rep2, "jsd")
        loss_st1 = node_seq_loss(node_rep1, seq_rep2, batch, "jsd")
        loss_st2 = node_seq_loss(node_rep2, seq_rep1, batch, "jsd")
        loss_st = (loss_st1 + loss_st2) / 2
        loss = self.l_ss * loss_ss + self.l_tt * loss_tt + self.l_st * loss_st

        self.log("train_loss", loss, logger=True, prog_bar=True, on_step=True, on_epoch=True)

        return loss


    def validation_step(self, batch, batch_idx):
        super().val_step(batch, batch_idx)
        
    def on_validation_epoch_end(self):
        acc = super().on_val_end()
        self.log("val_acc", acc, logger=True, prog_bar=True, on_epoch=True)


    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        _, node_enc1, node_enc2 = self.model.encode_graph()
        seq_rep_combined, seq_pooled1, seq_pooled2 = self.model.encode_sequence(batch, node_enc1, node_enc2)
        return seq_rep_combined

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"], weight_decay=1e-6)
        # Scheduler?
        return optimizer

    @property
    def name(self):
        return self.__class__.__name__

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    @staticmethod
    def create_trans_mx(cpath, traj_map, num_nodes):
     # calculate transition matrix 
        trans_mat = np.zeros((num_nodes, num_nodes))
        for seq in tqdm(cpath):
            for i, id1 in enumerate(seq):
                for id2 in seq[i:]:
                    node_id1, node_id2 = traj_map[id1], traj_map[id2]
                    trans_mat[node_id1, node_id2] += 1
        return trans_mat



class CLMEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        hidden_size,
        edge_index1,
        edge_index2,
        graph_encoder1,
        graph_encoder2,
        seq_encoder,
    ):
        super(CLMEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.node_embedding = nn.Embedding(vocab_size, embed_size)
        self.padding = torch.zeros(1, hidden_size, requires_grad=False)
        self.edge_index1 = edge_index1.cuda()
        self.edge_index2 = edge_index2.cuda()
        self.graph_encoder1 = graph_encoder1
        self.graph_encoder2 = graph_encoder2
        self.seq_encoder = seq_encoder

    def encode_graph(self):
        node_emb = self.node_embedding.weight
        node_enc1 = self.graph_encoder1(node_emb, self.edge_index1)
        node_enc2 = self.graph_encoder2(node_emb, self.edge_index2)
        return node_enc1 + node_enc2, node_enc1, node_enc2

    def encode_sequence(self, sequences, node_enc1, node_enc2):

        batch_size, max_seq_len = sequences.size()
        src_key_padding_mask = sequences == self.vocab_size
        pool_mask = (1 - src_key_padding_mask.int()).transpose(0, 1).unsqueeze(-1)

        lookup_table1 = torch.cat([node_enc1, self.padding.cuda()], 0)
        seq_emb1 = (
            torch.index_select(lookup_table1, 0, sequences.view(-1))
            .view(batch_size, max_seq_len, -1)
            .transpose(0, 1)
        )
        seq_enc1 = self.seq_encoder(seq_emb1, None, src_key_padding_mask)
        seq_pooled1 = (seq_enc1 * pool_mask).sum(0) / pool_mask.sum(0)

        lookup_table2 = torch.cat([node_enc2, self.padding.cuda()], 0)
        seq_emb2 = (
            torch.index_select(lookup_table2, 0, sequences.view(-1))
            .view(batch_size, max_seq_len, -1)
            .transpose(0, 1)
        )
        seq_enc2 = self.seq_encoder(seq_emb2, None, src_key_padding_mask)
        seq_pooled2 = (seq_enc2 * pool_mask).sum(0) / pool_mask.sum(0)
        return seq_pooled1 + seq_pooled2, seq_pooled1, seq_pooled2

    def forward(self, sequences):
        _, node_enc1, node_enc2 = self.encode_graph()
        _, seq_pooled1, seq_pooled2 = self.encode_sequence(sequences, node_enc1, node_enc2)
        return node_enc1, node_enc2, seq_pooled1, seq_pooled2



class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, num_layers, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            input_size, num_heads, hidden_size, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src, src_mask, src_key_padding_mask):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        return output


class GraphEncoder(nn.Module):
    def __init__(self, input_size, output_size, encoder_layer, num_layers, activation):
        super(GraphEncoder, self).__init__()

        self.num_layers = num_layers
        self.activation = activation

        self.layers = [encoder_layer(input_size, output_size)]
        for _ in range(1, num_layers):
            self.layers.append(encoder_layer(output_size, output_size))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.activation(self.layers[i](x, edge_index))
        return x