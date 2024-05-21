import random
import sys
import os
from operator import itemgetter

import pickle
import numpy as np
import swifter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from . import transforms as T

sys.path.append("../..")
sys.path.append('models/baselines/trajcl_files')
from models.baselines.trajcl_files.traj import *
from pipelines.utils import ROOT_DIR, load_config
from models.token_embs.src.cell.init_cs import init_cs

# ref: TrjSR
def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))

class TrajCLDataset(Dataset):
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

        self.aug1 = get_aug_fn(config['trajcl_aug1'])
        self.aug2 = get_aug_fn(config['trajcl_aug2'])

        cell_embs_filepath = os.path.join(ROOT_DIR, config["model_files_path"], config["embs_file"])
        #cellspace_filepath = os.path.join(ROOT_DIR, config["model_files_path"], config["dataset_cell_file"])

        self.embs = pickle.load(open(cell_embs_filepath, 'rb')).to('cpu').detach() # tensor
        data_config = load_config(name=city, ctype="dataset")
        self.cellspace = init_cs(data_config['min_lon'], data_config['min_lat'], data_config['max_lon'], data_config['max_lat'], data_config['cellspace_buffer'], data_config['cell_size'])

        
        self.collate_custom = CustomCollateFunction(self.cellspace, self.embs, self.aug1, self.aug2, config)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data.iloc[idx].merc_seq    
    
    
class CustomCollateFunction(nn.Module):
    def __init__(self, cellspace, embs, augfn1, augfn2, config) :

        super(CustomCollateFunction, self).__init__()
        self.cellspace = cellspace
        self.embs = embs
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

        trajs_emb_p = [torch.tensor(generate_spatial_features(t, self.cellspace, self.config['trajcl_local_mask_sidelen'])) for t in trajs_p]
        trajs1_emb_p = [torch.tensor(generate_spatial_features(t, self.cellspace, self.config['trajcl_local_mask_sidelen'])) for t in trajs1_p]
        trajs2_emb_p = [torch.tensor(generate_spatial_features(t, self.cellspace, self.config['trajcl_local_mask_sidelen'])) for t in trajs2_p]

        trajs_emb_p = pad_sequence(trajs_emb_p, batch_first = False).float()
        trajs1_emb_p = pad_sequence(trajs1_emb_p, batch_first = False).float()
        trajs2_emb_p = pad_sequence(trajs2_emb_p, batch_first = False).float()

        trajs_emb_cell = [self.embs[list(t)] for t in trajs_cell]
        trajs1_emb_cell = [self.embs[list(t)] for t in trajs1_cell]
        trajs2_emb_cell = [self.embs[list(t)] for t in trajs2_cell]

        trajs_emb_cell = pad_sequence(trajs_emb_cell, batch_first = False) # [seq_len, batch_size, emb_dim]
        trajs1_emb_cell = pad_sequence(trajs1_emb_cell, batch_first = False) # [seq_len, batch_size, emb_dim]
        trajs2_emb_cell = pad_sequence(trajs2_emb_cell, batch_first = False) # [seq_len, batch_size, emb_dim]

        trajs_len = torch.tensor(list(map(len, trajs_cell)), dtype = torch.long)
        trajs1_len = torch.tensor(list(map(len, trajs1_cell)), dtype = torch.long)
        trajs2_len = torch.tensor(list(map(len, trajs2_cell)), dtype = torch.long)

        # return: two padded tensors and their lengths
        return trajs1_emb_cell, trajs1_emb_p, trajs1_len, trajs2_emb_cell, trajs2_emb_p, trajs2_len, trajs_emb_cell, trajs_emb_p, trajs_len
