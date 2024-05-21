import datetime
import random
from operator import itemgetter

import os
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import swifter
import torch
import torch_geometric.transforms as T
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm
from . import transforms as transforms

from pipelines.utils import ROOT_DIR

class Start2Dataset(Dataset):
    def __init__(self, data, edge_df, line_graph, config=None, train=True):
        self.edge_df = edge_df
        self.line_graph = line_graph
        self.map = self._create_edge_emb_mapping()
        data["cpath"] = (
            data["cpath"].swifter.apply(lambda x: [self.map[l] + 3 for l in x]).values
        )

        self.seq_len = 150
        self.add_cls = True
        self.pad_index = 0
        self.unk_index = 1
        self.sos_index = 2

        self.masking_ratio = 0.2
        self.masking_mode = "together"
        self.distribution = "geometric" #"random"
        self.avg_mask_len = 2 #3
        self.exclude_feats = None

        trans_matrix = np.load(os.path.join(ROOT_DIR ,config["transition_matrix_path"]))

        traj = data["cpath"].tolist()
        timestamps = data["road_timestamps"].tolist()
        timestamps = [array.tolist() for array in timestamps]

        # Apply Augmentation: Truncate
        aug1 = transforms.Subset_with_time(p=1, traj_subset_ratio=0.8)
        traj1, timestamps1 = zip(
            *[
                aug1(traj[i], timestamps[i])
                for i in range(len(traj))
            ]
        )
        # Second Augmentation use masking
        aug2 = transforms.Mask_with_time_list(p=0.8, traj_mask_ratio = 0.3)
        traj2, timestamps2 = zip(
            *[
                aug2(traj[i], timestamps[i])
                for i in range(len(traj))
            ]
        )
        saved_files_path = config["saved_files_path"]
        self.traj_list, self.temporal_mat_list = self.data_processing(traj, timestamps) #self._load_data(traj, timestamps, traj_list_path=f"{saved_files_path}/traj_list_{len(traj)}.pkl", temporal_mat_path=f"{saved_files_path}/temporal_mat_{len(traj)}.pkl")
        self.traj_list1, self.temporal_mat_list1 = self.data_processing(traj1, timestamps1) #self._load_data(traj1, timestamps1, traj_list_path=f"{saved_files_path}/traj_list1_{len(traj1)}.pkl", temporal_mat_path=f"{saved_files_path}/temporal_mat1_{len(traj1)}.pkl")
        # Use masked or orig?
        self.traj_list2, self.temporal_mat_list2 = self.data_processing(traj2, timestamps2) #self._load_data(traj2, timestamps2, traj_list_path=f"{saved_files_path}/traj_list2_{len(traj2)}.pkl", temporal_mat_path=f"{saved_files_path}/temporal_mat2_{len(traj2)}.pkl")

        self.features, self.edge_index, self.trans_list = self._create_graph_data(
            trans_matrix
        )
        
        # Are they going to be used?
        self.data_argument1 = []
        self.data_argument2 = []

        print(
            f"Data Shapes --> features: ({self.features.shape}), edge index: ({self.edge_index.shape}), trans list: ({self.trans_list.shape})"
        )

    def _cal_mat(self, tim_list):
        # calculate the temporal relation matrix
        seq_len = len(tim_list)
        mat = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                off = abs(tim_list[i] - tim_list[j])
                mat[i][j] = off
        return mat  # (seq_len, seq_len)

    def _load_data(self, traj, timestamps, traj_list_path=None, temporal_mat_path=None):
        if os.path.exists(traj_list_path) and os.path.exists(temporal_mat_path):
            traj_list = pickle.load(open(traj_list_path, 'rb'))
            temporal_mat_list = pickle.load(open(temporal_mat_path, 'rb'))
            print(f"Loaded preprocessed data: {traj_list_path}, {temporal_mat_path}")
        else:
            traj_list, temporal_mat_list = self.data_processing(traj, timestamps)
            if traj_list_path is not None and temporal_mat_path is not None:
                pickle.dump(traj_list, open(traj_list_path, 'wb'))
                pickle.dump(temporal_mat_list, open(temporal_mat_path, 'wb'))
        return traj_list, temporal_mat_list
    
    def data_processing(self, traj, timestamps):
        traj_list = []
        temporal_mat_list = []
        for i in tqdm(range(len(traj))):
            loc_list = traj[i]
            tim_list = timestamps[i]#.tolist() # sometimes we get lists after augs, sometimes numpy arrays
            if len(loc_list) != len(tim_list):   # tim_list should be one more than loc_list
                diff = len(loc_list) - len(tim_list)
                tim_list = tim_list[:diff]
            new_tim_list = [datetime.datetime.utcfromtimestamp(tim) for tim in tim_list]
            minutes = [
                new_tim.hour * 60 + new_tim.minute + 1 for new_tim in new_tim_list
            ]
            weeks = [new_tim.weekday() + 1 for new_tim in new_tim_list]
            if self.add_cls:
                new_loc_list = [self.sos_index] + loc_list
                minutes = [self.pad_index] + minutes
                weeks = [self.pad_index] + weeks
                tim_list = [tim_list[0]] + tim_list
            temporal_mat = self._cal_mat(tim_list)
            temporal_mat_list.append(temporal_mat)
            traj_fea = np.array([new_loc_list, tim_list, minutes, weeks]).transpose(
                (1, 0)
            )
            traj_list.append(traj_fea)

        return traj_list, temporal_mat_list

    def _create_edge_emb_mapping(self):
        map = {}
        nodes = list(self.line_graph.nodes)
        for index, id in zip(self.edge_df.index, self.edge_df.fid):
            map[id] = nodes.index(index)

        return map

    def __len__(self):
        return len(self.traj_list)

    def __getitem__(self, idx):
        traj_ind = self.traj_list[idx]  # (seq_length, feat_dim)
        temporal_mat = self.temporal_mat_list[idx]  # (seq_length, seq_length)

        traj_ind1 = self.traj_list1[idx]
        temporal_mat1 = self.temporal_mat_list1[idx]

        traj_ind2 = self.traj_list2[idx]
        temporal_mat2 = self.temporal_mat_list2[idx]


        mask = Start2Dataset.noise_mask(
            traj_ind,
            self.masking_ratio,
            self.avg_mask_len,
            self.masking_mode,
            self.distribution,
            self.exclude_feats,
            self.add_cls,
        )  # (seq_length, feat_dim) boolean array

        mask1 = None
        if 'mask' in self.data_argument1:
            mask1 = Start2Dataset.noise_mask(
                traj_ind1,
                self.masking_ratio,
                self.avg_mask_len,
                self.masking_mode,
                self.distribution,
                self.exclude_feats,
                self.add_cls,
            )
        mask2 = None
        if 'mask' in self.data_argument2:
            mask2 = Start2Dataset.noise_mask(
                traj_ind2,
                self.masking_ratio,
                self.avg_mask_len,
                self.masking_mode,
                self.distribution,
                self.exclude_feats,
                self.add_cls,
            )

        graph_dict = {
            "node_features": self.features,  # (N, 27)
            "edge_index": self.edge_index,  # (2, E)
            "loc_trans_prob": self.trans_list,  # (E, 1)
        }

        return torch.LongTensor(traj_ind), torch.LongTensor(mask), torch.LongTensor(temporal_mat), \
               torch.LongTensor(traj_ind1), torch.LongTensor(traj_ind2), \
               torch.LongTensor(temporal_mat1), torch.LongTensor(temporal_mat2), \
               torch.LongTensor(mask1) if mask1 is not None else None, \
               torch.LongTensor(mask2) if mask2 is not None else None, \
                graph_dict

    def _create_graph_data(self, trans_matrix: np.array):
        # create edge_index
        map_id = {j: i for i, j in enumerate(self.line_graph)}
        edge_list = nx.to_pandas_edgelist(self.line_graph)
        edge_list["sidx"] = edge_list["source"].map(map_id)
        edge_list["tidx"] = edge_list["target"].map(map_id)

        edge_probs = []
        for s, t in zip(edge_list["sidx"], edge_list["tidx"]):
            edge_probs.append(trans_matrix[s, t])

        edge_probs = torch.from_numpy(np.array(edge_probs)).unsqueeze(1).float()

        edge_index = np.array(edge_list[["sidx", "tidx"]].values).T
        edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

        # create feature matrix
        df = self.edge_df.copy()
        df["idx"] = df.index.map(map_id)
        df.sort_values(by="idx", axis=0, inplace=True)

        df.rename(columns={"fid": "id"}, inplace=True)

        df = df[["highway_enc", "lanes", "maxspeed", "length"]]

        df["lanes"] = df["lanes"].str.extract(r"(\d+)")
        df["maxspeed"] = df["maxspeed"].str.extract(r"(\d+)")

        # normalize continiuos features
        df["length"] = (df["length"] - df["length"].min()) / (
            df["length"].max() - df["length"].min()
        )

        cats = ["highway_enc", "maxspeed", "lanes"]
        df = pd.get_dummies(
            df,
            columns=cats,
            drop_first=True,
        )
        features = torch.tensor(df.values.astype(float), dtype=torch.double)
        data = Data(x=features, edge_index=edge_index)
        transform = T.Compose(
            [
                T.NormalizeFeatures(),
            ]
        )
        data = transform(data)

        return data.x.float(), data.edge_index, edge_probs
    
    @staticmethod
    # collate_unsuperv_contrastive_split_lm
    def collate_custom(batch):
        max_len = 150

        features, masks, temporal_mat, features1, features2, temporal_mat1, temporal_mat2, mask1, mask2, graph_dict = zip(*batch)
        data_for_mask = list(zip(features, masks, temporal_mat))
        dara_for_contra = list(zip(features1, features2, temporal_mat1, temporal_mat2, mask1, mask2))

        X1, X2, padding_masks1, padding_masks2, batch_temporal_mat1, batch_temporal_mat2 \
            = Start2Dataset.collate_unsuperv_contrastive_split(data=dara_for_contra, max_len=max_len)

        masked_x, targets, target_masks, padding_masks, batch_temporal_mat = Start2Dataset.collate_unsuperv_mask(
            data=data_for_mask, max_len=max_len)
        return X1, X2, padding_masks1, padding_masks2, batch_temporal_mat1, batch_temporal_mat2, \
            masked_x, targets, target_masks, padding_masks, batch_temporal_mat, graph_dict

    @staticmethod
    def collate_unsuperv_contrastive_split(data, max_len=None):
        batch_size = len(data)
        features1, features2, temporal_mat1, temporal_mat2, mask1, mask2 = zip(*data)  # list of (seq_length, feat_dim)
        X1, batch_temporal_mat1, padding_masks1 = Start2Dataset._inner_slove_data(
            features1, temporal_mat1, batch_size, max_len, mask1)
        X2, batch_temporal_mat2, padding_masks2 = Start2Dataset._inner_slove_data(
            features2, temporal_mat2, batch_size, max_len, mask2)
        return X1.long(), X2.long(), padding_masks1, padding_masks2, \
            batch_temporal_mat1.long(), batch_temporal_mat2.long()
    
    
    @staticmethod
    def _inner_slove_data(features, temporal_mat, batch_size, max_len, mask=None):
        # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
        lengths = [X.shape[0] for X in features]  # original sequence length for each time series
        if max_len is None:
            max_len = max(lengths)
        X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)
        batch_temporal_mat = torch.zeros(batch_size, max_len, max_len,
                                        dtype=torch.long)  # (batch_size, padded_length, padded_length)

        target_masks = torch.zeros_like(X, dtype=torch.bool)  # (batch_size, padded_length, feat_dim)
        for i in range(batch_size):
            end = min(lengths[i], max_len)
            X[i, :end, :] = features[i][:end, :]
            batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]
            if mask[i] is not None:
                target_masks[i, :end, :] = mask[i][:end, :]

        padding_masks = Start2Dataset.padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)

        target_masks = ~target_masks  # (batch_size, padded_length, feat_dim)
        target_masks = target_masks * padding_masks.unsqueeze(-1)

        if mask[0] is not None:
            X[..., 0:1].masked_fill_(target_masks[..., 0:1] == 1, 1)  # loc -> mask_index
            X[..., 1:].masked_fill_(target_masks[..., 1:] == 1, 0)  # others -> pad_index
        return X, batch_temporal_mat, padding_masks
    
    @staticmethod
    def noise_mask(
        X,
        masking_ratio,
        lm=3,
        mode="together",
        distribution="random",
        exclude_feats=None,
        add_cls=True,
    ):
        if exclude_feats is not None:
            exclude_feats = set(exclude_feats)

        if distribution == "geometric":  # stateful (Markov chain)
            if mode == "separate":  # each variable (feature) is independent
                mask = np.ones(X.shape, dtype=bool)
                for m in range(X.shape[1]):  # feature dimension
                    if exclude_feats is None or m not in exclude_feats:
                        mask[:, m] = Start2Dataset.geom_noise_mask_single(
                            X.shape[0], lm, masking_ratio
                        )  # time dimension
            else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
                mask = np.tile(
                    np.expand_dims(
                        Start2Dataset.geom_noise_mask_single(
                            X.shape[0], lm, masking_ratio
                        ),
                        1,
                    ),
                    X.shape[1],
                )
        elif (
            distribution == "random"
        ):  # each position is independent Bernoulli with p = 1 - masking_ratio
            if mode == "separate":
                mask = np.random.choice(
                    np.array([True, False]),
                    size=X.shape,
                    replace=True,
                    p=(1 - masking_ratio, masking_ratio),
                )
            else:
                mask = np.tile(
                    np.random.choice(
                        np.array([True, False]),
                        size=(X.shape[0], 1),
                        replace=True,
                        p=(1 - masking_ratio, masking_ratio),
                    ),
                    X.shape[1],
                )
        else:
            mask = np.ones(X.shape, dtype=bool)
        if add_cls:
            mask[0] = True  # CLS at 0, set mask=1
        return mask

    @staticmethod
    def padding_mask(lengths, max_len=None):
        batch_size = lengths.numel()
        max_len = (
            max_len or lengths.max_val()
        )  # trick works because of overloading of 'or' operator for non-boolean types
        return (
            torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1))
        )

    @staticmethod
    def collate_unsuperv_mask(data, max_len=None):
        batch_size = len(data)
        features, masks, temporal_mat = zip(*data)  # list of (seq_length, feat_dim)

        # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
        lengths = [
            X.shape[0] for X in features
        ]  # original sequence length for each time series
        if max_len is None:
            max_len = max(lengths)
        X = torch.zeros(
            batch_size, max_len, features[0].shape[-1], dtype=torch.long
        )  # (batch_size, padded_length, feat_dim)
        batch_temporal_mat = torch.zeros(
            batch_size, max_len, max_len, dtype=torch.long
        )  # (batch_size, padded_length, padded_length)

        # masks related to objective
        target_masks = torch.zeros_like(
            X, dtype=torch.bool
        )  # (batch_size, padded_length, feat_dim)
        for i in range(batch_size):
            end = min(lengths[i], max_len)
            X[i, :end, :] = features[i][:end, :]
            target_masks[i, :end, :] = masks[i][:end, :]
            batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]

        padding_masks = Start2Dataset.padding_mask(
            torch.tensor(lengths, dtype=torch.int16), max_len=max_len
        )

        target_masks = ~target_masks  # (batch_size, padded_length, feat_dim)
        target_masks = target_masks * padding_masks.unsqueeze(-1)

        targets = X.clone()
        targets = targets.masked_fill_(target_masks == 0, 0)

        X[..., 0:1].masked_fill_(target_masks[..., 0:1] == 1, 1)  # loc -> mask_index
        X[..., 1:].masked_fill_(target_masks[..., 1:] == 1, 0)  # others -> pad_index
        return (
            X.long(),
            targets.long(),
            target_masks,
            padding_masks,
            batch_temporal_mat.long(),
        )

    @staticmethod
    def geom_noise_mask_single(L, lm, masking_ratio):
        keep_mask = np.ones(L, dtype=bool)
        p_m = (
            1 / lm
        )  # probability of each masking sequence stopping. parameter of geometric distribution.
        p_u = (
            p_m * masking_ratio / (1 - masking_ratio)
        )  # probability of each unmasked sequence stopping. parameter of geometric distribution.
        p = [p_m, p_u]

        # Start in state 0 with masking_ratio probability
        state = int(
            np.random.rand() > masking_ratio
        )  # state 0 means masking, 1 means not masking
        for i in range(L):
            keep_mask[
                i
            ] = state  # here it happens that state and masking value corresponding to state are identical
            if np.random.rand() < p[state]:
                state = 1 - state

        return keep_mask
