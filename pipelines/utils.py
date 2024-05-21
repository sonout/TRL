import os
from ast import literal_eval
from pathlib import Path
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import yaml
import torch
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATASET_NAME = "trajectories_mapped_road_stamps.parquet"
PLAIN_DATASET_NAME = "plain_dataset.parquet"
PEPROCESSED_DATASET_NAME = "trajectory_preprocessed.parquet"
PRE_MAP_DATASET_NAME = "pre_map_dataset.csv"
# Non-optional train config information keys
TRAIN_CONFIG_KEYS = [
    "model_name",
    "model_args",
    "gpu_device_ids",
    "city",
    "dataset_class",
]
EVAL_CONFIG_KEYS = []



def seed_everything(seed=None):
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False


def load_config(name: str, ctype: str) -> Dict[str, Any]:
    """
    Load a config from a file.

    Args:
        name (str): Name of the config file.

    Returns:
        Dict[str, Any]: Loaded config.
    """
    return yaml.safe_load(
        Path(os.path.join(ROOT_DIR, "configs", ctype, f"{name}.yaml")).read_text()
    )


def load_road_network(
    city_name: str,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, nx.Graph, nx.Graph]:
    """
    Load graph from edges and nodes shape file
    """
    gdf_nodes = gpd.read_file(
        os.path.join(ROOT_DIR, f"datasets/osm/{city_name}/nodes.shp")
    )
    gdf_edges = gpd.read_file(
        os.path.join(ROOT_DIR, f"datasets/osm/{city_name}/edges.shp")
    )
    gdf_nodes.set_index("osmid", inplace=True)
    gdf_edges.set_index(["u", "v", "key"], inplace=True)

    # encode highway column
    gdf_edges["highway"] = gdf_edges["highway"].str.extract(r"(\w+)")
    le = LabelEncoder()
    gdf_edges["highway_enc"] = le.fit_transform(gdf_edges["highway"])

    G = ox.graph_from_gdfs(gdf_nodes, gdf_edges)
    line_G = nx.line_graph(G, create_using=nx.DiGraph)

    return gdf_edges, gdf_nodes, G, line_G


def generate_train_test_split(
    city_name: str, seed: int, test_size=0.3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_parquet(
        os.path.join(ROOT_DIR, "datasets/trajectory", city_name, DATASET_NAME)
    )

    train, test = train_test_split(data, test_size=test_size, random_state=seed)
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    return train, test

def generate_train_val_test_split(
    city_name: str, seed: int, test_size=0.2, val_size=0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_parquet(
        os.path.join(ROOT_DIR, "datasets/trajectory", city_name, DATASET_NAME)
    )
    train_size = 1 - test_size - val_size
    # train is now 75% of the entire data set
    train, test_val = train_test_split(data, test_size=1 - train_size, random_state=seed)
    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    test, val = train_test_split(test_val, test_size=test_size/(test_size + val_size), random_state=seed) 

    train, test = train_test_split(data, test_size=0.3, random_state=seed)
    print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")
    return train, val, test


def preprocess_trajectory_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses mapped trajectory dataset to get timestamp for each traversed road segment.

    Args:
        df (pd.DataFrame): input dataset

    Returns:
        pd.DataFrame: transformed dataset
    """
    df["cpath"] = (
        df["cpath"]
        .swifter.progress_bar(False)
        .apply(
            lambda x: np.fromstring(
                x.replace("\n", "").replace("(", "").replace(")", "").replace(" ", ""),
                sep=",",
                dtype=np.int,
            )
        )
    )

    df = df[["id", "cpath", "duration"]].copy()
    df["duration"] = df["duration"].apply(literal_eval)
    df["travel_time"] = df["duration"].apply(np.sum)

    return df
