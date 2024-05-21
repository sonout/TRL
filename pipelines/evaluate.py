import os
import sys

sys.path.append("..")

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from models import ModelTrainer
from models.utils import DATASETS_PACKAGE_NAME, MODELS_PACKAGE_NAME
from tasks.utils import TASKS_PACKAGE_NAME

from .utils import DATASET_NAME, EVAL_CONFIG_KEYS, ROOT_DIR, load_road_network, seed_everything


class EvaluatePipeline:
    """
    Pipeline for evaluating a single model on a single task
    """

    def __init__(self, model_config: dict, task_config: dict, state_dict = None):
        seed_everything() # if no seed is given we take a random seed each time        

        # check config
        if not all(k in task_config for k in EVAL_CONFIG_KEYS):
            raise ValueError("Config missing necessary information.")
        
        self.task_name = task_config["name"]

        # get task dataset class
        task_dataset_class = getattr(
            sys.modules[TASKS_PACKAGE_NAME], task_config["dataset_class"]
        )

        # Get Test data
        data = pd.read_parquet(
            os.path.join(
                ROOT_DIR,
                "datasets/trajectory",
                task_config["city"],
                "test/test_{}.parquet".format(task_config["seed"]),
            )
        )
        data = data[data["cpath"].str.len() > 3].copy()
        data = data.iloc[: task_config["sample_size"] :]

        # do task specific preprocessing regarding the trajectory sequeneces before embedding them
        data = task_dataset_class.preprocess_dataset(data)

        # Load Road Network
        edge_df, _, _, LG = load_road_network(city_name=task_config["city"])
        # Get Model Dataset
        dataset = getattr(
            sys.modules[DATASETS_PACKAGE_NAME], model_config["dataset_class"]
        )

        # Load Model
        model_config["model_args"]["input_size"] = edge_df.shape[0]
        model = getattr(sys.modules[MODELS_PACKAGE_NAME], model_config["model_name"])(
            config=model_config["model_args"]
        )
        if state_dict is not None:
            model.load_state_dict(state_dict)
        else:
            model.load_model(path=os.path.join(ROOT_DIR, model_config["state_dict_path"]))

        # generate train test dataset by embedding trajectories
        pred_trainer = pl.Trainer(
            accelerator="gpu", devices=task_config["gpu_device_ids"]
        )

        data = task_dataset_class.generate_embed(
            data,
            pred_trainer,
            model,
            dataset,
            {"edge_df": edge_df, "line_graph": LG, "train": False, "config": model_config},
        )

        # split train, val, test
        # Using those splits we roughly get 70% train, 10% val, 20% test
        test_val_size = 0.3
        test_size = 0.66
        # For traj_sim we do not need train and val
        if self.task_name == "TrajSim":
            test_val_size = 0.999
            test_size = 0.999


        train, val_test = train_test_split(
            data,
            test_size=test_val_size,
            random_state=task_config["seed"],
        )

        val, test = train_test_split(
            val_test,
            test_size=test_size,
            random_state=task_config["seed"],
        )

        train_dataset = task_dataset_class(data=train, edge_df=edge_df, line_graph=LG, task_config=task_config)
        val_dataset = task_dataset_class(data=val, edge_df=edge_df, line_graph=LG, task_config=task_config)
        test_dataset = task_dataset_class(data=test, edge_df=edge_df, line_graph=LG, task_config=task_config)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=task_config["batch_size"],
            shuffle=True,
            num_workers=12,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=task_config["batch_size"],
            shuffle=False,
            num_workers=12,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=task_config["batch_size"],
            shuffle=False,
            num_workers=12,
        )

        # load task
        task_config["task_args"]["input_size"] = model_config["model_args"]["emb_size"]
        task_config["task_args"]["num_segments"] = edge_df.shape[0]
        task = getattr(sys.modules[TASKS_PACKAGE_NAME], task_config["task"])(
            config=task_config["task_args"]
        )

        do_training = True
        if self.task_name == "TrajSim": # We do not need to finetune for traj sim
            do_training = False

        # initialize trainer
        self.trainer = ModelTrainer(
            model=task,
            config=task_config,
            do_training=do_training,
        )

    def run(self, save=True):
        self.trainer.train(train=self.train_loader, val=self.val_loader)
        p_results = self.trainer.evaluate_performance(test=self.test_loader)

        if save:
            ...

        return p_results
