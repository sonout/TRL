import math
import os
import sys
from pathlib import Path
import time

import networkx as nx
import numpy as np
from scipy import sparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from _walker import random_walks as _random_walks

import torch_geometric.transforms as T
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import GAE, GATConv, GCNConv, InnerProductDecoder
from torch_geometric.utils import (add_self_loops, from_networkx,
                                   negative_sampling, remove_self_loops)

sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent))
from pipelines.utils import ROOT_DIR


class SkipGramModel():
    def __init__(
        self,
        data,
        adj,
        device,
        emb_dim=128,
        walk_length=30,
        walks_per_node=25,
        context_size=5,
        num_neg=10,
    ):
        self.model = SkipGram(
            data.edge_index,
            adj,
            embedding_dim=emb_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_neg,
            sparse=True,
        ).to(device)
        self.loader = self.model.loader(batch_size=384, shuffle=True, num_workers=4)
        self.device = device
        self.data = data
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)

    def train(self, epochs):
        avg_loss = 0
        print("start training")
        for e in range(epochs):
            self.model.train()
            total_loss = 0
            for pos_rw, neg_rw in self.loader:
                self.optimizer.zero_grad()
                loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                #print("loss: {}".format(loss.item()))
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


class SkipGram(nn.Module):
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


    def pos_sample(self, batch):
        # batch = batch.repeat(self.walks_per_node)
        # rowptr, col, _ = self.adj.csr()
        rw = torch.tensor(
            SkipGram.traj_walk(
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


class SFCModel():
    def __init__(self, data, device, adj=None, emb_dim=128, layers=2, add_edge_degree=True):
        data.validate(raise_on_error=True)
        self.device = device
        self.adj = adj        
        
        #self.train_data = data
        self.train_data = self.transform_data(data, self.adj, add_edge_degree=add_edge_degree) # Do we need to do that? what does it do?
        self.train_data = self.train_data.to(device)

        self.model = GCNEncoder(data.x.shape[1], emb_dim, layers=layers).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train(self, epochs: int = 1000):
        avg_loss = 0
        for e in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            #z = self.model.encode(self.train_data.x, self.train_data.edge_index)
            z = self.model(
                self.train_data.x,
                self.train_data.edge_traj_index,
                self.train_data.edge_weight,
            )

            #loss = self.model.recon_loss(z, self.train_data.edge_index)
            loss = self.recon_loss(z, self.train_data.edge_traj_index)
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()

            if e > 0 and e % 500 == 0:
                print("Epoch: {}, avg_loss: {}".format(e, avg_loss / e))

    
    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        decoder = InnerProductDecoder()
        EPS = 1e-15

        pos_loss = -torch.log(decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        #pos_edge_index, _ = remove_self_loops(pos_edge_index)
        #pos_edge_index, _ = add_self_loops(pos_edge_index)

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss
    
    
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
            self.model(
                self.train_data.x,
                self.train_data.edge_traj_index,
                self.train_data.edge_weight,
            )
            .detach()
            .cpu()
        )
    
    def transform_data(self, data, adj, add_edge_degree = True):
        G = nx.from_numpy_array(adj.T, create_using=nx.DiGraph)
        data_traj = from_networkx(G)
        data.edge_traj_index = data_traj.edge_index
        data.edge_weight = data_traj.weight

        if add_edge_degree:
            transform = T.Compose([
                T.OneHotDegree(9), # training without features
                #T.ToDevice(device),
            ])
            data = transform(data)
        return data




class GAEModel():
    def __init__(self, data, device, encoder, emb_dim=128, layers=2):
        data.validate(raise_on_error=True)
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
        )


class GATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, 2 * out_channels)
        self.conv2 = GATConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x.float(), edge_index).relu()
        return self.conv2(x, edge_index)



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









####### Node2vec
from torch_geometric.nn import Node2Vec


class Node2VecModel():
    def __init__(
        self,
        data,
        device,
        emb_dim=128,
        walk_length=50, #30
        walks_per_node=10, #25
        context_size=10, #5
        q=1,
        p=1,
    ):
        self.model = Node2Vec(data.edge_index, 
                              embedding_dim=emb_dim, 
                              walk_length=walk_length,
                              context_size=context_size,
                              walks_per_node=walks_per_node,
                              num_negative_samples=10,
                              p=p, q=q, sparse=True).to(device)
        
        self.loader = self.model.loader(batch_size=32, shuffle=True, num_workers=4)

        self.device = device
        self.data = data
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.001) #0.01
        self.checkpoint_file = os.path.join(ROOT_DIR, "models/road_embs", 'node2vec_best.pt')

    def train(self, epochs):
        
        avg_loss = 0
        epoch_train_loss_best = 10000000.0
        for e in range(epochs):
            time_ep = time.time()
            self.model.train()
            total_loss = 0
            for pos_rw, neg_rw in self.loader:
                self.optimizer.zero_grad()
                loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.loader)
            
            if avg_loss < epoch_train_loss_best:
                epoch_best = e
                epoch_train_loss_best = avg_loss
                self.save_model(filepath=self.checkpoint_file)

            print("Epoch: {}, avg_loss: {}, time: {}".format(e, avg_loss, time.time()-time_ep))
        print("Best epoch: {}, best loss: {}".format(epoch_best, epoch_train_loss_best))
        self.load_model(filepath=self.checkpoint_file)

    def save_model(self, filepath="save/model.pt"):
        torch.save({'model_state_dict': self.model.state_dict()}, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath,  map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def save_emb(self, path):
        np.savetxt(path + "embedding.out", X=self.model().detach().cpu().numpy())

    def load_emb(self, path=None):
        if path:
            self.emb = np.loadtxt(path)
        return self.model()
