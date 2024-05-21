import numpy as np
import random
import swifter
import torch
from torch.utils.data import DataLoader, Dataset
from operator import itemgetter


class Trembr2Dataset(Dataset):
    def __init__(self, data, edge_df, line_graph, config=None, train=True):
        self.edge_df = edge_df
        self.line_graph = line_graph

        #mapped = data["cpath"].swifter.apply(lambda x: [self.map[l] + 1 for l in x]).values
        #self.X = mapped  # trajectory
        #self.yt = data["road_timestamps"].values  # timestamps
        #self.yr = mapped  # road segment labels mapped to road network ids

        self.seq_len = 150
        self.trajs = data["cpath"].values  # trajectory
        self.traj_map = self._create_edge_emb_mapping()

    def __len__(self):
        return self.trajs.shape[0]

    def __getitem__(self, idx):
        traj = list(itemgetter(*self.trajs[idx])(self.traj_map))
        if len(traj) > self.seq_len:
                traj = self.cut_traj(traj)
        length = len(traj)
        padding = [0 for _ in range(self.seq_len - len(traj))]
        traj.extend(padding)

        return (
            traj,
            length,
            #torch.tensor(np.diff(self.yt[idx]), dtype=int),
            #torch.tensor(self.yr[idx], dtype=int),
            #self.map,
        )

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
    
    @staticmethod
    def collate_custom(batch):
        (
            traj,
            length,
        ) = zip(*batch)

        return (
            torch.tensor(traj),
            torch.tensor(length),
        )