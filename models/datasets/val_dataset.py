import random
from operator import itemgetter
import sys

import numpy as np
import pandas as pd
import swifter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset



from models.utils import DATASETS_PACKAGE_NAME, MODELS_PACKAGE_NAME



class ValDataset(Dataset):
    def __init__(self, data, edge_df, line_graph, config=None, train=True):
        self.edge_df = edge_df
        self.line_graph = line_graph

        # Preprocess for traj_sim
        self.data = self.preprocess_dataset(data)

        data_even, data_uneven = data.copy(), data.copy()
        data_even["cpath"], data_even["road_timestamps"], data_even["coord_seq"], data_even["timestamps"] = (
            data_even["even_cpath"],
            data_even["even_road_timestamps"],
            data_even["even_coord_seq"],
            data_even["even_timestamps"],
        )
        data_uneven["cpath"], data_uneven["road_timestamps"], data_uneven["coord_seq"], data_uneven["timestamps"] = (
            data_uneven["uneven_cpath"],
            data_uneven["uneven_road_timestamps"],
            data_uneven["uneven_coord_seq"],
            data_uneven["uneven_timestamps"],
        )



        # Use model_datasets to preprocess for model
        self.val_dataset_even = getattr(sys.modules[DATASETS_PACKAGE_NAME], config["dataset_class"])(
            data=data_even, edge_df=edge_df, line_graph=line_graph, config=config
        )
        self.val_dataset_uneven = getattr(sys.modules[DATASETS_PACKAGE_NAME], config["dataset_class"])(
            data=data_uneven, edge_df=edge_df, line_graph=line_graph, config=config
        )



    def collate_custom(self, batch):
        even, uneven, idx = zip(*batch)

        even = self.val_dataset_even.collate_custom(even)
        uneven = self.val_dataset_uneven.collate_custom(uneven)
        
        return (
            even,
            uneven,
            idx
        )
    
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        even = self.val_dataset_even.__getitem__(idx)
        uneven = self.val_dataset_uneven.__getitem__(idx)
        data_tuple = (even, uneven, idx)
        return data_tuple

    def preprocess_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        # split trajectories in $$D_a$$ and $$D_b$$ datasets
        trajs = zip(data["cpath"].values, data["road_timestamps"].values)

        # Method from trembr paper: Split Trajs in even and uneven
        d_a, d_b, t_a, t_b = [], [], [], []
        for traj, time in trajs:
            even_points, even_stamps = traj[0::2], time[1::2]
            uneven_points, uneven_stamps = traj[1::2], time[2::2]
            even_stamps, uneven_stamps = np.insert(even_stamps, 0, time[0]), np.insert(
                uneven_stamps, 0, time[0]
            )
            d_a.append(even_points)
            d_b.append(uneven_points)
            t_a.append(even_stamps)
            t_b.append(uneven_stamps)

        data.loc[:, "even_cpath"] = d_a
        data.loc[:, "uneven_cpath"] = d_b
        data.loc[:, "even_road_timestamps"] = t_a
        data.loc[:, "uneven_road_timestamps"] = t_b

        # Do the traj split in even and uneven for coords_seq and timestamps 
        trajs = zip(data["coord_seq"].values, data["timestamps"].values)
        d_a, d_b, t_a, t_b = [], [], [], []
        for traj, time in trajs:
            even_points, even_stamps = traj[0::2], time[1::2]
            uneven_points, uneven_stamps = traj[1::2], time[2::2]
            even_stamps, uneven_stamps = np.insert(even_stamps, 0, time[0]), np.insert(
                uneven_stamps, 0, time[0]
            )
            d_a.append(even_points)
            d_b.append(uneven_points)
            t_a.append(even_stamps)
            t_b.append(uneven_stamps)
        
        data.loc[:, "even_coord_seq"] = d_a
        data.loc[:, "uneven_coord_seq"] = d_b
        data.loc[:, "even_timestamps"] = t_a
        data.loc[:, "uneven_timestamps"] = t_b

        return data