import sys
import os

import numpy as np
from tqdm import tqdm
import random
import torch
from torch.utils.data import DataLoader, Dataset
from operator import itemgetter


sys.path.append("../..")

from models.utils import create_edge_emb_mapping, cut_traj
from pipelines.utils import ROOT_DIR

def _create_trans_mx(cpath, traj_map, num_nodes):
     # calculate transition matrix 
    trans_mat = np.zeros((num_nodes, num_nodes))
    for seq in tqdm(cpath):
        for i, id1 in enumerate(seq):
            for id2 in seq[i:]:
                node_id1, node_id2 = traj_map[id1], traj_map[id2]
                trans_mat[node_id1, node_id2] += 1
    trans_mat = trans_mat / (trans_mat.max(axis=1, keepdims=True, initial=0.) + 1e-9)
    row, col = np.diag_indices_from(trans_mat)
    trans_mat[row, col] = 0
    return trans_mat

class JCLMDataset(Dataset):
    def __init__(self, data, edge_df, line_graph, config=None, train=True):
        self.edge_df = edge_df
        self.line_graph = line_graph

        self.num_nodes = len(edge_df)
        self.data = data
        self.trajs = data["cpath"].values  # trajectory
        self.traj_map = create_edge_emb_mapping(edge_df, line_graph)

        # transition matrix
        # check if it exists under config["trans_mx_path"] if not create it
        
        if os.path.isfile(os.path.join(ROOT_DIR, config["model_args"]["trans_mx_path"])):
            self.trans_mx = np.load(os.path.join(ROOT_DIR, config["model_args"]["trans_mx_path"]))
        else:
            self.trans_mx = _create_trans_mx(data.cpath, self.traj_map, len(edge_df))
            np.save(os.path.join(ROOT_DIR, config["model_args"]["trans_mx_path"]), self.trans_mx)

        self.data_train = self.train_data_loader(data, self.num_nodes, self.traj_map)


    def __len__(self):
        return self.data_train.shape[0]

    def __getitem__(self, idx):
        return self.data_train[idx]
    
    @staticmethod
    def train_data_loader(df, padding_id, traj_map):
        min_len = 10
        max_len = 100
        num_samples = len(df) # 250000  # 500000

        #df = df.loc[(df["path_len"] > min_len) & (df["path_len"] < max_len)]
        df['cpath'] = df['cpath'].apply(lambda x: x[:max_len])
        df["path_len"] = df["cpath"].map(len)
        if len(df) > num_samples:
            df = df.sample(n=num_samples, replace=False, random_state=1)

        arr = np.full([num_samples, max_len], padding_id, dtype=np.int32)
        for i in tqdm(range(len(df))):
            row = df.iloc[i]
            path_arr = np.array(row["cpath"], dtype=np.int32)
            traj = list(itemgetter(*path_arr)(traj_map))
            arr[i, : row["path_len"]] = traj

        return torch.LongTensor(arr)
    
    @staticmethod
    def collate_custom(batch):
        return torch.stack(batch)
    