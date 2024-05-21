import os
import sys

sys.path.append("..")

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import ModelTrainer
from models.utils import DATASETS_PACKAGE_NAME, MODELS_PACKAGE_NAME

from .utils import DATASET_NAME, ROOT_DIR, TRAIN_CONFIG_KEYS, load_road_network, seed_everything


class TrainPipeline:
    def __init__(self, config: dict):
        seed_everything() # if no seed is given we take a random seed each time        

        # check config
        if not all(k in config for k in TRAIN_CONFIG_KEYS):
            raise ValueError("Config missing necessary information.")

        # Train dataset
        data = pd.read_parquet(
            os.path.join(
                ROOT_DIR,
                "datasets/trajectory",
                config["city"],
                "train/train_{}.parquet".format(config["seed"]),
            )
        )
        data = data[data["cpath"].str.len() > 3].copy()
        if config["debug"]:
            data = data.iloc[:1000, :]
        edge_df, _, _, LG = load_road_network(city_name=config["city"])
        dataset = getattr(sys.modules[DATASETS_PACKAGE_NAME], config["dataset_class"])(
            data=data, edge_df=edge_df, line_graph=LG, config=config
        )

        self.dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_custom,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=12,
        )

        # Val dataset
        val_data = pd.read_parquet(
            os.path.join(
                ROOT_DIR,
                "datasets/trajectory",
                config["city"],
                "val/val_{}.parquet".format(config["seed"]),
            )
        )
        val_data = val_data[val_data["cpath"].str.len() > 3].copy()
        if config["debug"]:
            val_data = val_data.iloc[:1000, :]
        val_data = val_data.iloc[:config["val_sample_size"], :]
        val_dataset = getattr(sys.modules[DATASETS_PACKAGE_NAME], "ValDataset")(
            data=val_data, edge_df=edge_df, line_graph=LG, config=config
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            collate_fn=val_dataset.collate_custom,
            batch_size=1024, # Set it to 1024 to avoid OOM, best would be all
            shuffle=False,
            num_workers=12,
        )


        # initilaize model
        config["model_args"]["input_size"] = edge_df.shape[0]
        model = getattr(sys.modules[MODELS_PACKAGE_NAME], config["model_name"])(
            config=config["model_args"]
        )

        # initialize trainer
        self.trainer = ModelTrainer(
            model=model,
            config=config,
        )

        self.config = config

    def run(self, save=True):
        state = self.trainer.train(train=self.dataloader, val=self.val_dataloader)

        if save:
            torch.save(state, os.path.join(ROOT_DIR, self.config["state_dict_path"]))

        # Possible future: safe automatically to path based on city and seed (would need to change in evalate.py as well)
        # os.path.join(ROOT_DIR, self.config["state_dict_path"], self.config["city"], f"{self.config['model_name']}_{self.config['seed']}.pt")
        return state
