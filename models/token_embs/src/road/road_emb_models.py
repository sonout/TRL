import os
import sys

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import pandas as pd

from sklearn.impute import KNNImputer

from torch_geometric.nn import GAE, GATConv
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import GAE, GATConv, GCNConv, InnerProductDecoder
from torch_geometric.utils import (add_self_loops, from_networkx,
                                   negative_sampling, remove_self_loops)


class GAEModel():
    def __init__(self, data, device, encoder, emb_dim=128, layers=2):
        self.model = GAE(
            encoder(data.x.shape[1], emb_dim)
        )  # feature dim, emb dim
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.model = self.model.to(device)
        self.device = device
        self.train_data = data
        self.train_data = self.train_data.to(device)

    def train(self, epochs: int = 1000):
        avg_loss = 0
        for e in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            z = self.model.encode(self.train_data.x, self.train_data.edge_index)
            loss = self.model.recon_loss(z, self.train_data.edge_index)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()

            if e > 0 and e % 500 == 0:
                print("Epoch: {}, avg_loss: {}".format(e, avg_loss / e))

    def save_model(self, path="save/"):
        torch.save(self.model.state_dict(), os.path.join(path + "/model.pt"))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save_emb(self, path):
        np.savetxt(
            os.path.join(path + "/embedding.out"),
            X=self.model.encode(self.train_data.x, self.train_data.edge_index)
            .detach()
            .cpu()
            .numpy(),
        )

    def load_emb(self, path=None):
        if path:
            return np.loadtxt(path)
        return (
            self.model.encode(self.train_data.x, self.train_data.edge_index)
            .detach()
            .cpu()
            .numpy()
        )


class GATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, 2 * out_channels)
        self.conv2 = GATConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x.float(), edge_index).relu()
        return self.conv2(x, edge_index)
    


class Node2VecModel():
    def __init__(
        self,
        data,
        device,
        emb_dim=128,
        walk_length=30, #50?
        walks_per_node=25, # 10?
        context_size=5, #10?
        q=1,
        p=1,
    ):
        self.model = Node2Vec(
            data.edge_index,
            embedding_dim=emb_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=10,
            p=p,
            q=q,
            sparse=True,
        ).to(device)
        self.loader = self.model.loader(batch_size=128, shuffle=True, num_workers=4)
        self.device = device
        self.data = data
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)

    def train(self, epochs):
        avg_loss = 0
        for e in range(epochs):
            self.model.train()
            total_loss = 0
            for pos_rw, neg_rw in self.loader:
                self.optimizer.zero_grad()
                loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss += total_loss / len(self.loader)
            if e > 0 and e % 20 == 0:
                print("Epoch: {}, avg_loss: {}".format(e, avg_loss / e))

    def save_model(self, path="save/"):
        torch.save(self.model.state_dict(), path + "model.pt")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save_emb(self, path):
        np.savetxt(path + "embedding.out", X=self.model().detach().cpu().numpy())

    def load_emb(self, path=None):
        if path:
            self.emb = np.loadtxt(path)
        return self.model().detach().cpu().numpy()



######################### Transition Probability based GCN Road Embedding ############################

class SFCModel():
    def __init__(self, feats_in, adj, device, emb_dim=64, layers=2):
        self.device = device
        self.adj = adj
        
        G = nx.from_numpy_array(adj.T, create_using=nx.DiGraph)
        self.data_adj = from_networkx(G).to(device)
        self.model = GCNEncoder(feats_in, emb_dim, layers=layers).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
    
    def train(self, x, epochs: int = 1000):
        x = x.to(self.device)
        avg_loss = 0
        print("Training SFC")
        for e in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Process each time step
            z_all = []
            for t in range(24):
                z_t = self.model(
                    x[:, t, :],
                    self.data_adj.edge_index,
                    self.data_adj.weight,
                )
                z_all.append(z_t)
            
            # Aggregate embeddings across time steps (e.g., mean)
            z = torch.stack(z_all, dim=1).mean(dim=1)
            
            loss = self.recon_loss(z, self.data_adj.edge_index)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()
            
            if e > 0 and e % 50 == 0:
                print(f"Epoch: {e}, avg_loss: {avg_loss / e}")

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        decoder = InnerProductDecoder()
        EPS = 1e-15
        pos_loss = -torch.log(decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), os.path.join(filepath))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save_emb(self, path):
        # Assuming self.train_data is defined elsewhere
        np.savetxt(
            os.path.join(path, "embedding.out"),
            X=self.model.encode(self.train_data.x, self.train_data.edge_index)
            .detach()
            .cpu()
            .numpy(),
        )

    def embed(self, x):
        x = x.to(self.device)
        # Process each time step
        z_all = []
        for t in range(24):
            z_t = self.model(x[:, t, :], self.data_adj.edge_index, self.data_adj.weight)
            z_all.append(z_t)
        
        # Aggregate embeddings across time steps (e.g., mean)
        return torch.stack(z_all, dim=1)

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, layers=2):
        super().__init__()
        self.layers = nn.Sequential()
        if layers == 2:
            self.layers.append(GCNConv(in_channels, 2 * out_channels))
            self.layers.append(GCNConv(2 * out_channels, out_channels))
        else:
            self.layers.append(GCNConv(in_channels, out_channels))

    def forward(self, x, edge_index, edge_weight):
        x = x.float()
        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_weight).relu()
        return self.layers[-1](x, edge_index, edge_weight)

class InnerProductDecoder(nn.Module):
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return torch.sigmoid(value) if sigmoid else value
    

#### ROAD EMBEDDING MODEL ####

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor
from _walker import random_walks as _random_walks
from scipy import sparse
from torch.utils.data import DataLoader
from operator import itemgetter
from tqdm import tqdm


class Traj2VecModel():
    def __init__(
        self,
        edge_index,
        adj,
        device,
        emb_dim=128,
        walk_length=30,
        walks_per_node=25,
        context_size=5,
        num_neg=10,
    ):
        self.model = Traj2Vec(
            edge_index,
            adj,
            embedding_dim=emb_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_neg,
            sparse=True,
        ).to(device)
        self.loader = self.model.loader(batch_size=128, shuffle=True, num_workers=4)
        self.device = device
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)

    def train(self, epochs):
        avg_loss = 0
        for e in range(epochs):
            self.model.train()
            total_loss = 0
            for pos_rw, neg_rw in self.loader:
                self.optimizer.zero_grad()
                loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                print("loss: {}".format(loss.item()))
            avg_loss += total_loss / len(self.loader)
            if e > 0 and e % 1 == 0:
                print("Epoch: {}, avg_loss: {}".format(e, avg_loss / e))

    def save_model(self, path="save/"):
        torch.save(self.model.state_dict(), path + "model.pt")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save_emb(self, path):
        np.savetxt(path + "embedding.out", X=self.model().detach().cpu().numpy())

    def load_emb(self, path=None):
        if path:
            self.emb = np.loadtxt(path)
        return self.model().detach().cpu().numpy()



class Traj2Vec(nn.Module):
    def __init__(
        self,
        edge_index,
        adj,
        embedding_dim,
        walk_length,
        context_size,
        walks_per_node=1,
        num_negative_samples=1,
        num_nodes=None,
        sparse=True,
    ):
        super().__init__()

        N = maybe_num_nodes(edge_index, num_nodes)
        self.adj = adj  # SparseTensor(row=row, col=col, sparse_sizes=(N, N))
        # self.adj = self.adj.to("cpu")
        # self.traj_data = Traj2Vec.map_traj_to_node_ids(traj_data, network)
        self.EPS = 1e-15

        assert walk_length >= context_size

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples

        self.embedding = nn.Embedding(N, embedding_dim, sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, batch=None):
        """Returns the embeddings for the nodes in :obj:`batch`."""
        emb = self.embedding.weight
        return emb if batch is None else emb.index_select(0, batch)

    def loader(self, **kwargs):
        return DataLoader(range(self.adj.shape[0]), collate_fn=self.sample, **kwargs) # Here we obtain a random walk sample

    @staticmethod
    def traj_walk(adj, walk_length, start, walks_per_node):

        A = sparse.csr_matrix(adj)
        indptr = A.indptr.astype(np.uint32)
        indices = A.indices.astype(np.uint32)
        data = A.data.astype(np.float32)

        walks = _random_walks(
            indptr, indices, data, start, walks_per_node, walk_length + 1
        )

        return walks.astype(int)

    @staticmethod
    def generate_traj_static_walks(traj_data, network, walk_length):
        # create map
        tmap = {}
        nodes = list(network.line_graph.nodes)
        for index, id in zip(network.gdf_edges.index, network.gdf_edges.fid):
            tmap[id] = nodes.index(index)

        # map traj id sequences to graph node id sequences
        mapped_traj = []
        for traj in traj_data:
            mapped_traj.append(itemgetter(*traj)(tmap))

        # generate matrix with walk length columns
        traj_matrix = np.zeros(
            shape=(
                sum(len(x) for x in mapped_traj)
                - (len(mapped_traj) * (walk_length))
                + 1,
                walk_length,
            )
        )
        run_idx = 0
        for j, traj in tqdm(enumerate(mapped_traj)):
            for i in range(len(traj)):
                if walk_length > len(traj) - i:
                    break
                window = i + walk_length
                traj_matrix[run_idx + i, :] = traj[i:window]
            run_idx += len(traj) - walk_length

        return traj_matrix

    def pos_sample(self, batch):
        # batch = batch.repeat(self.walks_per_node)
        # rowptr, col, _ = self.adj.csr()
        rw = torch.tensor(
            Traj2Vec.traj_walk(
                self.adj,
                self.walk_length,
                start=batch,
                walks_per_node=self.walks_per_node,
            ),
            dtype=int,
        )
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j : j + self.context_size])
        return torch.cat(walks, dim=0)

    def neg_sample(self, batch):
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = torch.randint(self.adj.shape[0], (batch.size(0), self.walk_length))
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j : j + self.context_size])
        return torch.cat(walks, dim=0)

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    def loss(self, pos_rw, neg_rw):
        r"""Computes the loss given positive and negative random walks."""

        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(
            pos_rw.size(0), -1, self.embedding_dim
        )

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(
            neg_rw.size(0), -1, self.embedding_dim
        )

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss
