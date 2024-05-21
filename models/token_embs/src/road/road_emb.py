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
