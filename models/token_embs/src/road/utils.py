
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import pandas as pd

from sklearn.impute import KNNImputer

from torch_geometric.data import Data
import torch_geometric.transforms as T

def create_pyg_data(line_graph, edge_df, calc_traveltime=True):
    ## edge_index ##
    map_id = {j: i for i, j in enumerate(line_graph)}
    edge_list = nx.to_pandas_edgelist(line_graph)
    edge_list["sidx"] = edge_list["source"].map(map_id)
    edge_list["tidx"] = edge_list["target"].map(map_id)
    edge_index = np.array(edge_list[["sidx", "tidx"]].values).T
    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

    ## Features ##
    df = edge_df.copy()
    df["idx"] = df.index.map(map_id)
    df.sort_values(by="idx", axis=0, inplace=True)
    df.rename(columns={"fid": "id"}, inplace=True)

    # Define possible columns and their types
    possible_columns = {
        "highway_enc": "categorical",
        "lanes": "numerical",
        "maxspeed": "numerical",
        "length": "continuous",
        "avg_speed": "continuous",
        "util": "continuous"
    }

    # Filter columns that are present in the dataframe
    present_columns = [col for col in possible_columns if col in df.columns]
    df = df[present_columns]

    # Add travel time if required and possible
    if calc_traveltime and "length" in df.columns and "avg_speed" in df.columns:
        df["travel_time"] = df["length"] / (df["avg_speed"] * (1000/3600))
        df["travel_time"] = (df["travel_time"] - df["travel_time"].min()) / (
            df["travel_time"].max() - df["travel_time"].min()
        )
        present_columns.append("travel_time")
        possible_columns["travel_time"] = "continuous"

    # Process each column based on its type
    for col in present_columns:
        if possible_columns[col] == "numerical":
            df[col] = df[col].str.extract(r"(\d+)")
        elif possible_columns[col] == "continuous":
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # Categorical: one-hot-encode
    cats = [col for col in present_columns if possible_columns[col] == "categorical"]
    if cats:
        df = pd.get_dummies(df, columns=cats, drop_first=True)

    # Impute
    imputer = KNNImputer(n_neighbors=1)
    imputed = imputer.fit_transform(df)
    df = pd.DataFrame(imputed, columns=df.columns)

    # Convert numerical columns to int
    for col in present_columns:
        if possible_columns[col] == "numerical":
            df[col] = df[col].astype(int)

    # Convert to PYG Data
    features = torch.tensor(df.astype(float).values, dtype=torch.float)
    data = Data(x=features, edge_index=edge_index)

    return data


from operator import itemgetter
import networkx as nx
from tqdm import tqdm   

def generate_node_traj_adj(
        LG, traj_data, traj_to_node , k: int = np.inf, bidirectional=True, add_self_loops=True
    ):
        nodes = list(LG)
        adj = nx.to_numpy_array(LG)
        np.fill_diagonal(adj, 0)

        if add_self_loops:
            adj += np.eye(len(nodes), len(nodes))

        for traj in tqdm(traj_data):
            # print(traj)
            for i, traj_node in enumerate(traj):
                if k == -1:
                    k = len(traj)
                left_slice, right_slice = min(k, i) if bidirectional else 0, min(
                    k + 1, len(traj) - i
                )
                traj_nodes = traj[(i - left_slice) : (i + right_slice)]
                # convert traj_nodes to graph_nodes
                target = itemgetter(traj_node)(traj_to_node)
                context = itemgetter(*traj_nodes)(traj_to_node)
                adj[target, context] += 1
        # remove self weighting if no self loops should be allowed
        if not add_self_loops:
            np.fill_diagonal(adj, 0)
            zero_rows = np.where(~adj.any(axis=1))[0]
            for idx in zero_rows:
                adj[idx, idx] = 1

        # norm adj row wise to get probs
        rowsum = adj.sum(axis=1, keepdims=True)
        adj = adj / rowsum

        # convert to edge_index

        return adj

def generate_trajid_to_nodeid(edge_df, LG):
    map = {}
    nodes = list(LG.nodes)
    for index, id in zip(edge_df.index, edge_df.fid):
        map[id] = nodes.index(index)

    return map