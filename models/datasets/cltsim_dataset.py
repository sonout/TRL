import random
from operator import itemgetter

import numpy as np
import swifter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import transforms as T


class CLTSimDataset(Dataset):
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

        # Augmentations/Transformations
        masking_p = 1
        cropping_p = 1
        flipping_p = 0.3
        transform1 = [
            #T.RandomFlip(p=flipping_p),
            T.RandomMasking(p=masking_p),
            #T.RandomCropping(p=cropping_p),
        ]
        transform2 = [
            #T.RandomFlip(p=flipping_p),
            #T.RandomMasking(p=masking_p),
            T.RandomCropping(p=cropping_p),
        ]
        self.transform1 = T.Compose(transform1)
        self.transform2 = T.Compose(transform2)

        self.collate_custom = CustomCollateFunction(self.transform1, self.transform2, self.seq_len)

    def __len__(self):
        return self.trajs.shape[0]

    def __getitem__(self, idx):
        traj = list(itemgetter(*self.trajs[idx])(self.traj_map))
        if len(traj) > self.seq_len:
                traj = self.cut_traj(traj)
        length = len(traj)

        return (
            torch.tensor(traj, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )

    # tested index mapping is correct
    # Stefan: I think this is for mapping the cpath values (aka idx of edge_df) to linegraph nodes
    #           Because the Roadembeddings are based on them?
    def _create_edge_emb_mapping(self):
        map = {}
        nodes = list(self.line_graph.nodes)
        for index, id in zip(self.edge_df.index, self.edge_df.fid):
            map[id] = nodes.index(index) # index is e.g. (25503936, 4722746638, 0)
        # print(map == map2) # yields true
        return map # Maps from idx edge_id/cpath value to linegraph node

    def cut_traj(self, traj):
        start_idx = int((len(traj) - self.seq_len) * random.random())
        return traj[start_idx : start_idx + self.seq_len]
    



class CustomCollateFunction(nn.Module):
    def __init__(self, transform1, transform2, seq_len) :

        super(CustomCollateFunction, self).__init__()
        self.transform1 = transform1
        self.transform2 = transform2
        self.seq_len = seq_len

    def forward(self, batch):
        trajs, lengths = zip(*batch)

        batch_size = len(batch)

        # list of transformed images
        # Transform then pad
        #transforms = [
        #    F.pad(input=self.transform(batch[i % batch_size][0]).unsqueeze_(0), pad=(0, self.seq_len - len(batch[i % batch_size][0])), mode='constant', value=0)
        #    for i in range(2 * batch_size)
        #]

        transforms1 = []
        transform1_lengths = []
        transforms2 = []
        transform2_lengths = []
        for i in range(batch_size):
            t = trajs[i]
            # 1. Transform trajectory
            trajs_t = self.transform1(t)#.unsqueeze_(0)
            transform1_lengths.append(len(trajs_t))
            # Pad trajectory
            padding_length = self.seq_len - len(trajs_t)
            trajs_padded = F.pad(input=trajs_t, pad=(0, padding_length), mode='constant', value=0).unsqueeze_(0)
            transforms1.append(trajs_padded)

            # 2. Transform trajectory
            trajs_t = self.transform2(t)#.unsqueeze_(0)
            transform2_lengths.append(len(trajs_t))
            # Pad trajectory
            padding_length = self.seq_len - len(trajs_t)
            trajs_padded = F.pad(input=trajs_t, pad=(0, padding_length), mode='constant', value=0).unsqueeze_(0)
            transforms2.append(trajs_padded)


        # Pad trajs
        trajs = [F.pad(input=t, pad=(0, self.seq_len - len(t)), mode='constant', value=0).unsqueeze_(0)  for t in trajs]


        # Get both X
        #X1 = transforms[:batch_size]
        #X2 = transforms[batch_size:]
        # tuple of transforms


        X1 = (torch.cat(transforms1, 0))
        X2 = (torch.cat(transforms2, 0))
        X1_len = torch.tensor(transform1_lengths)
        X2_len = torch.tensor(transform2_lengths)
        trajs = (torch.cat(trajs, 0))
        lengths = torch.tensor(lengths)

        #transforms = (
        #    torch.cat(transforms[:batch_size], 0),
        #    torch.cat(transforms[batch_size:], 0),
        #) 
        
        return (
            X1,
            X2,
            trajs,
            X1_len,
            X2_len,
            lengths,
        )
