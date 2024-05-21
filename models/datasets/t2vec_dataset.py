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

from models.utils import lonlat2meters
from models.token_embs.src.cell.init_cs import init_cs


# pair-wise conversion -- structural features and spatial feasures
def merc2cell2(src, cs: CellSpace):
    tgt = []
    for p in src:
        x,y = p[0],p[1]
        if  cs.x_min <= x <= cs.x_max \
                and cs.y_min <= y <= cs.y_max:
            cell_id = cs.get_cellid_by_point(*p)
            tgt.append((cell_id, p))
    # remove consecutive duplicates
    tgt = [v for i, v in enumerate(tgt) if i == 0 or v[0] != tgt[i-1][0]]
    if len(tgt) == 0:
        print("1 Outside of CellSpace")
        tgt = [(0, src[-1])]
    tgt, tgt_p = zip(*tgt)
    return tgt, tgt_p

class T2VecDataset(Dataset):
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
        self.data = data[['coord_seq','merc_seq']]
        #self.data['merc_seq'] = self.data['merc_seq'].apply(lambda x: [list(y) for y in x])

        config = config['model_args']

        #self.aug1 = get_aug_fn(config['trajcl_aug1'])
        #self.aug2 = get_aug_fn(config['trajcl_aug2'])
        transform1 = [
            #T.Simplify(p=0.3),
            #T.Shift(p=0.3),
            T.Mask(p=1),
            T.Subset(p=0.4),
        ]
        self.transform1 = T.Compose(transform1)


        # cellspace_filepath = os.path.join(ROOT_DIR, config["model_files_path"], config["dataset_cell_file"])
        data_config = load_config(name=city, ctype="dataset") 
        self.cellspace = init_cs(data_config['min_lon'], data_config['min_lat'], data_config['max_lon'], data_config['max_lat'], data_config['cellspace_buffer'], data_config['cell_size'])
        
        self.collate_custom = CustomCollateFunction(self.cellspace, self.transform1, self.transform1, config)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data.iloc[idx].merc_seq
        # check if len is less than 200 else cut
        if len(x) > 200:
            x = x[:200]
        return x
    
    
class CustomCollateFunction(nn.Module):
    def __init__(self, cellspace, augfn1, augfn2, config) :

        super(CustomCollateFunction, self).__init__()
        self.cellspace = cellspace
        self.augfn1 = augfn1
        self.augfn2 = augfn2
        self.config = config

    def forward(self, batch):
        trajs = batch

        trajs1 = [self.augfn1(t) for t in trajs]
        trajs2 = [self.augfn2(t) for t in trajs]

        trajs_cell, trajs_p = zip(*[merc2cell2(t, self.cellspace) for t in trajs])
        trajs1_cell, trajs1_p = zip(*[merc2cell2(t, self.cellspace) for t in trajs1])
        trajs2_cell, trajs2_p = zip(*[merc2cell2(t, self.cellspace) for t in trajs2])

        # to tensor long
        trajs_cell = [torch.tensor(t, dtype = torch.long) for t in trajs_cell]
        trajs1_cell = [torch.tensor(t, dtype = torch.long) for t in trajs1_cell]
        trajs2_cell = [torch.tensor(t, dtype = torch.long) for t in trajs2_cell]
        
        trajs_cell = pad_sequence(trajs_cell, batch_first = True) # [seq_len, batch_size, emb_dim]
        trajs1_cell = pad_sequence(trajs1_cell, batch_first = True) # [seq_len, batch_size, emb_dim]
        trajs2_cell = pad_sequence(trajs2_cell, batch_first = True) # [seq_len, batch_size, emb_dim]

        trajs_len = torch.tensor(list(map(len, trajs_cell)), dtype = torch.long)
        trajs1_len = torch.tensor(list(map(len, trajs1_cell)), dtype = torch.long)
        trajs2_len = torch.tensor(list(map(len, trajs2_cell)), dtype = torch.long)

        # return: two padded tensors and their lengths
        return trajs1_cell, trajs1_len, trajs2_cell, trajs2_len, trajs_cell, trajs_len
