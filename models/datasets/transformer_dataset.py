import sys
import os
import pickle
import pandas as pd
import numpy as np
import random
import swifter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from operator import itemgetter


sys.path.append("../..")
from . import transforms as T
from pipelines.utils import ROOT_DIR, load_config
from models.utils import merc2cell2, lonlat2meters
from models.token_embs.src.cell.init_cs import init_cs


class TransformerDataset(Dataset):
    def __init__(self, data, edge_df, line_graph, config=None, train=True):
        self.edge_df = edge_df
        self.line_graph = line_graph
        self.seq_len = 200
        city = config['city']

        config = config['model_args']
        self.cell_or_road = config['cell_or_road']

        if self.cell_or_road == 'cell':
            data['merc_seq'] = data.coord_seq.swifter.apply(lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj])
            self.trajs = data["merc_seq"].values  # trajectory
            self.cell_road_emb = pickle.load(open(os.path.join(ROOT_DIR, config["cellroad_emb_path"]), 'rb')).to('cpu').detach() 
            data_config = load_config(name=city, ctype="dataset")
            self.cellspace = init_cs(data_config['min_lon'], data_config['min_lat'], data_config['max_lon'], data_config['max_lat'], data_config['cellspace_buffer'], data_config['cell_size'])


        elif self.cell_or_road == 'road':
            self.trajs = data["cpath"].values  # trajectory
            self.traj_map = self._create_edge_emb_mapping()
            self.cell_road_emb = pickle.load(open(os.path.join(ROOT_DIR, config["cellroad_emb_path"]), 'rb'))
            self.cellspace = None


        transform = [
            #T.Simplify(p=0.3),
            #T.Shift(p=0.3),
            T.Mask(p=0.7),
            T.Subset(p=0.7),
        ]
        self.transform = T.Compose(transform)
        
        self.collate_custom = CustomCollateFunction(self.cell_or_road, self.cell_road_emb, self.cellspace, self.transform, self.transform, config)

    def __len__(self):
        return self.trajs.shape[0]

    def __getitem__(self, idx):
        # This is the cpath values/edge_df idx
        traj = self.trajs[idx]

        if len(traj) > self.seq_len:
            traj = self.cut_traj(traj)
        
        if self.cell_or_road == 'road':
            # This are the nodes in the line_graph
            traj = list(itemgetter(*traj)(self.traj_map))
        return traj

    # tested index mapping is correct
    def _create_edge_emb_mapping(self):
        map = {}
        nodes = list(self.line_graph.nodes)
        for index, id in zip(self.edge_df.index, self.edge_df.fid):
            map[id] = nodes.index(index)
        # print(map == map2) # yields true

        return map

    def cut_traj(self, traj):
        start_idx = int((len(traj) - self.seq_len) * random.random())
        return traj[start_idx : start_idx + self.seq_len]
    
class CustomCollateFunction(nn.Module):
    def __init__(self, cell_or_road, road_emb, cellspace, augfn1, augfn2, config) :

        super(CustomCollateFunction, self).__init__()
        self.cell_or_road = cell_or_road
        self.road_emb = road_emb
        self.cellspace = cellspace
        self.augfn1 = augfn1
        self.augfn2 = augfn2
        self.config = config

    def forward(self, batch):
        trajs = batch

        trajs1 = [self.augfn1(t) for t in trajs]
        trajs2 = [self.augfn2(t) for t in trajs]

        if self.cell_or_road == 'cell':
            trajs, trajs_p = zip(*[merc2cell2(t, self.cellspace) for t in trajs])
            trajs1, trajs1_p = zip(*[merc2cell2(t, self.cellspace) for t in trajs1])
            trajs2, trajs2_p = zip(*[merc2cell2(t, self.cellspace) for t in trajs2])

        trajs_emb_road = [self.road_emb[list(t)] for t in trajs]
        trajs1_emb_road = [self.road_emb[list(t)] for t in trajs1]
        trajs2_emb_road = [self.road_emb[list(t)] for t in trajs2]

        trajs_emb_road = pad_sequence(trajs_emb_road, batch_first = True) # [seq_len, batch_size, emb_dim]
        trajs1_emb_road = pad_sequence(trajs1_emb_road, batch_first = True) # [seq_len, batch_size, emb_dim]
        trajs2_emb_road = pad_sequence(trajs2_emb_road, batch_first = True) # [seq_len, batch_size, emb_dim]

        trajs_len = torch.tensor(list(map(len, trajs)), dtype = torch.long)
        trajs1_len = torch.tensor(list(map(len, trajs1)), dtype = torch.long)
        trajs2_len = torch.tensor(list(map(len, trajs2)), dtype = torch.long)

        # return: two padded tensors and their lengths
        return trajs1_emb_road, trajs1_len, trajs2_emb_road, trajs2_len, trajs_emb_road, trajs_len