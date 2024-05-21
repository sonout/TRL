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


sys.path.append("../..")

from models.utils import DATASETS_PACKAGE_NAME, MODELS_PACKAGE_NAME
from pipelines.utils import load_config, ROOT_DIR



class ConcateDataset(Dataset):
    def __init__(self, data, edge_df, line_graph, config=None, train=True):
        self.edge_df = edge_df
        self.line_graph = line_graph
        self.data = data

        # We need the right configs as well
        self.model_config1 = load_config(name=config["model_args"]["concate_model1"], ctype="model")
        self.model_config2 = load_config(name=config["model_args"]["concate_model2"], ctype="model")


        # Use model_datasets to preprocess for model
        self.dataset_model1 = getattr(sys.modules[DATASETS_PACKAGE_NAME], config["dataset_model1"])(
            data=data, edge_df=edge_df, line_graph=line_graph, config=self.model_config1
        )
        self.dataset_model2 = getattr(sys.modules[DATASETS_PACKAGE_NAME], config["dataset_model2"])(
            data=data, edge_df=edge_df, line_graph=line_graph, config=self.model_config2
        )

        self.use_3_models = False  
        if config["model_args"].get("concate_model3", None) not in [None, "None"]:
            self.use_3_models = True
            self.model_config3 = load_config(name=config["model_args"]["concate_model3"], ctype="model")
            self.dataset_model3 = getattr(sys.modules[DATASETS_PACKAGE_NAME], config["dataset_model3"])(
                data=data, edge_df=edge_df, line_graph=line_graph, config=self.model_config3
            )

    def collate_custom(self, batch):
        if self.use_3_models:
            batch_model1, batch_model2, batch_model3 = zip(*batch)
            batch_model1 = self.dataset_model1.collate_custom(batch_model1)
            batch_model2 = self.dataset_model2.collate_custom(batch_model2)
            batch_model3 = self.dataset_model3.collate_custom(batch_model3)
            return (
                batch_model1,
                batch_model2,
                batch_model3
            )
        else:
            batch_model1, batch_model2 = zip(*batch)
            batch_model1 = self.dataset_model1.collate_custom(batch_model1)
            batch_model2 = self.dataset_model2.collate_custom(batch_model2)
            return (
                batch_model1,
                batch_model2
            )
    
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        batch1 = self.dataset_model1.__getitem__(idx)
        batch2 = self.dataset_model2.__getitem__(idx)

        if self.use_3_models:
            batch3 = self.dataset_model3.__getitem__(idx)
            data_tuple = (batch1, batch2, batch3)
        else:
            data_tuple = (batch1, batch2)
        return data_tuple
