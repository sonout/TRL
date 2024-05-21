import numpy as np
import pandas as pd
import pytorch_lightning as pl
import swifter
import torch
from torch.utils.data import DataLoader, Dataset



class TravelTimeDataset(Dataset):
    def __init__(self, data, edge_df, line_graph, task_config):
        self.X = data["embs"].values
        #self.y = data["road_timestamps"].apply(lambda x: x[-1] - x[0]).values
        self.y = data["timestamps"].apply(lambda x: x[-1] - x[0]).values


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.Tensor(self.X[idx]),
            torch.tensor(self.y[idx], dtype=int),
        )

    @classmethod
    def preprocess_dataset(cls, data: pd.DataFrame):
        return data

    @classmethod
    def generate_embed(
        cls,
        data,
        trainer: pl.Trainer,
        model: pl.LightningDataModule,
        model_dataset_class,
        dataset_args: dict,
    ):
        dataset = model_dataset_class(data, **dataset_args)
        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_custom,
            batch_size=128,
            shuffle=False,
            num_workers=12,
        )

        emb_dataset = trainer.predict(model, dataloader)
        emb_dataset = torch.cat(emb_dataset).detach().cpu().tolist()
        data["embs"] = emb_dataset

        return data


class DestinationDataset(Dataset):
    def __init__(self, data, edge_df, line_graph, task_config):
        self.edge_df = edge_df
        self.line_graph = line_graph
        self.map = self._create_edge_emb_mapping()
        self.X = data["embs"].values
        self.y = data["y"].apply(lambda x: self.map[x]).values

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.Tensor(self.X[idx]),
            torch.tensor(self.y[idx], dtype=int),
        )

    def _create_edge_emb_mapping(self):
        map = {}
        nodes = list(self.line_graph.nodes)
        for index, id in zip(self.edge_df.index, self.edge_df.fid):
            map[id] = nodes.index(index)
        # print(map == map2) # yields true

        return map

    @classmethod
    def preprocess_dataset(cls, data: pd.DataFrame):
        data["y"] = data["cpath"].apply(lambda x: x[-1]).values
        data["cpath"] = data["cpath"].apply(lambda x: x[: int(len(x) * 0.9)]).values
        data["road_timestamps"] = (
            data["road_timestamps"]
            .apply(lambda x: x[: int((len(x) - 1) * 0.9) + 1])
            .values
        )
        data["coord_seq"] = data["coord_seq"].apply(lambda x: x[: int(len(x) * 0.9)]).values
        data["timestamps"] = (
            data["timestamps"]
            .apply(lambda x: x[: int((len(x) - 1) * 0.9) + 1])
            .values
        )
        # Also preprocess coord_seq and timestamps
        #data["coord_seq"] = data.apply(lambda row: [row['coord_seq'][i] for i in range(len(row['opath'])) if row['opath'][i] in row['cpath']]).values
        #coord_seq_new = []
        #for index, row in data.iterrows():
        #    opath = row['opath']
        #    cpath = row['cpath']
        #    coord_seq = row['coord_seq']
        #    coord_seq_i = [coord_seq[i] for i in range(len(opath)) if opath[i] in cpath]
        #    # Check if len of traj is less than 10, if so add last point x times to make it 10
        #    if len(coord_seq_i) < 10:
        #        for i in range(10 - len(coord_seq_i)):
        #            coord_seq_i.append(coord_seq_i[-1])
        #    coord_seq_new.append(coord_seq_i)

        #data['coord_seq'] = coord_seq_new
        #data['timestamps'] = data.apply(lambda row: row['timestamps'][:len(row['coord_seq'])], axis=1)


        return data

    @classmethod
    def generate_embed(
        cls,
        data,
        trainer: pl.Trainer,
        model: pl.LightningDataModule,
        model_dataset_class,
        dataset_args: dict,
    ):
        dataset = model_dataset_class(data, **dataset_args)
        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_custom,
            batch_size=128,
            shuffle=False,
            num_workers=12,
        )

        emb_dataset = trainer.predict(model, dataloader)
        emb_dataset = torch.cat(emb_dataset).detach().cpu().tolist()
        data["embs"] = emb_dataset

        return data


class TrajSimDataset(Dataset):
    def __init__(self, data, edge_df, line_graph, task_config):
        query_size = task_config["db_query_size"]
        db_neg_size = task_config["db_neg_size"]

        self.X = data["embs_even"].values[:query_size]
        self.y = np.array(data["embs_uneven"].values.tolist())[:(db_neg_size+query_size)]
        print(f"Using {len(self.X)} query and {len(self.y)} query & db_neg samples")


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (torch.Tensor(self.X[idx]), torch.Tensor(self.y), idx)

    @classmethod
    def preprocess_dataset(cls, data: pd.DataFrame) -> pd.DataFrame:
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

    @classmethod
    def generate_embed(
        cls,
        data,
        trainer: pl.Trainer,
        model: pl.LightningDataModule,
        model_dataset_class,
        dataset_args: dict,
    ):
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


        dataset_even, dataset_uneven = model_dataset_class(
            data_even, **dataset_args
        ), model_dataset_class(data_uneven, **dataset_args)

        dataloader_even = DataLoader(
            dataset_even,
            collate_fn=dataset_even.collate_custom,
            batch_size=128,
            shuffle=False,
            num_workers=12,
        )
        dataloader_uneven = DataLoader(
            dataset_uneven,
            collate_fn=dataset_uneven.collate_custom,
            batch_size=128,
            shuffle=False,
            num_workers=12,
        )

        emb_datasets = trainer.predict(model, [dataloader_even, dataloader_uneven])
        even_emb = torch.cat(emb_datasets[0]).detach().cpu().tolist()
        uneven_emb = torch.cat(emb_datasets[1]).detach().cpu().tolist()

        data["embs_even"] = even_emb
        data["embs_uneven"] = uneven_emb

        return data
