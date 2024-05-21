import random
import sys
import os

import pickle
import numpy as np
import swifter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


sys.path.append("../..")
sys.path.append('models/baselines/trajcl_files')
from models.baselines.trajcl_files.traj import *
from models.token_embs.src.cell.cellspace import CellSpace
from . import transforms as T
from pipelines.utils import ROOT_DIR, load_config
from models.token_embs.src.cell.init_cs import init_cs

from models.utils import lonlat2meters

def merc2cell2(src, cs: CellSpace):
    tgt = [cs.get_cellid_by_point(*p) for p in src]
    return tgt


class CSTTEDataset(Dataset):
    def __init__(self, data, edge_df, line_graph, config=None, train=True):
        self.edge_df = edge_df
        self.line_graph = line_graph
        city = config['city']

        #mapped = data["cpath"].swifter.apply(lambda x: [self.map[l] + 1 for l in x]).values
        #self.X = mapped  # trajectory
        #self.yt = data["road_timestamps"].values  # timestamps
        #self.yr = mapped  # road segment labels mapped to road network ids

        # Create Mercator Seq.
        data['merc_seq'] = data.coord_seq.swifter.apply(lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj])

        # We only need gps traj and merc_seq
        self.data = data[['coord_seq','merc_seq', 'timestamps']]
        #self.data['merc_seq'] = self.data['merc_seq'].apply(lambda x: [list(y) for y in x])

        config = config['model_args']
        grid_length = config['grid_length']

        data_config = load_config(name=city, ctype="dataset")
        self.cellspace = init_cs(data_config['min_lon'], data_config['min_lat'], data_config['max_lon'], data_config['max_lat'], data_config['cellspace_buffer'], grid_length)


                
        self.collate_custom = CustomCollateFunction(self.cellspace, config)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data.iloc[idx].merc_seq
        t = self.data.iloc[idx].timestamps

        # if t longer than x then cut
        if len(t) > len(x):
            t = t[:len(x)]
        assert len(x) == len(t), f"len(x) != len(t), {len(x)} != {len(t)}"

        # check if len is less than 200 else cut
        if len(x) > 200:
            x = x[:200]
            t = t[:200]
        return x, t


class CustomCollateFunction(nn.Module):
    def __init__(self, cellspace, config) :

        super(CustomCollateFunction, self).__init__()
        self.cellspace = cellspace
        self.config = config

    def forward(self, batch):
        trajs, t = zip(*batch)

        # For Augmentation, split into even and uneven
        uneven_points = [traj[::2] for traj in trajs]
        even_points = [traj[1::2] for traj in trajs]
        uneven_t = [t[::2] for t in t]
        even_t = [t[1::2] for t in t]

        trajs_cell = [merc2cell2(t, self.cellspace) for t in trajs]
        trajs1_cell = [merc2cell2(t, self.cellspace) for t in uneven_points]
        trajs2_cell = [merc2cell2(t, self.cellspace) for t in even_points]

        trajs_len = torch.tensor(list(map(len, trajs_cell)), dtype = torch.long)
        trajs1_len = torch.tensor(list(map(len, trajs1_cell)), dtype = torch.long)
        trajs2_len = torch.tensor(list(map(len, trajs2_cell)), dtype = torch.long)

        # Pad sequence
        trajs_cell = pad_sequence([torch.tensor(t) for t in trajs_cell], batch_first=True)
        trajs1_cell = pad_sequence([torch.tensor(t) for t in trajs1_cell], batch_first=True)
        trajs2_cell = pad_sequence([torch.tensor(t) for t in trajs2_cell], batch_first=True)
        # Pad points
        trajs = pad_sequence([torch.tensor(t) for t in trajs], batch_first=True)
        uneven_points = pad_sequence([torch.tensor(t) for t in uneven_points], batch_first=True)
        even_points = pad_sequence([torch.tensor(t) for t in even_points], batch_first=True)
        # Pad time
        t = pad_sequence([torch.tensor(t) for t in t], batch_first=True)
        uneven_t = pad_sequence([torch.tensor(t) for t in uneven_t], batch_first=True)
        even_t = pad_sequence([torch.tensor(t) for t in even_t], batch_first=True)

        # return: two padded tensors and their lengths
        return trajs, t, trajs_cell, trajs_len, \
                uneven_points, uneven_t, trajs1_cell, trajs1_len, \
                even_points, even_t, trajs2_cell, trajs2_len
    