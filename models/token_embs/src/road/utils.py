
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

    df = df[["highway_enc", "lanes", "maxspeed", "length", "avg_speed", "util"]]
    # Add travel time
    if calc_traveltime:
        df["travel_time"] = df["length"] / (df["avg_speed"] * (1000/3600))
        df["travel_time"] = (df["travel_time"] - df["travel_time"].min()) / (
            df["travel_time"].max() - df["travel_time"].min()
        ) 

    # Some values like [2,1] or ["50","40"] need to be extracted to a single Integer
    df["lanes"] = df["lanes"].str.extract(r"(\d+)")
    df["maxspeed"] = df["maxspeed"].str.extract(r"(\d+)")

    # Continuous: normalize
    # Features: length, avg_speed, util, travel_time
    df["length"] = (df["length"] - df["length"].min()) / (
        df["length"].max() - df["length"].min()
    )
    df["avg_speed"] = (df["avg_speed"] - df["avg_speed"].min()) / (
        df["avg_speed"].max() - df["avg_speed"].min()
    )
    df["util"] = (df["util"] - df["util"].min()) / (
        df["util"].max() - df["util"].min()
    )
           

    # Categorical: one-hot-encode
    cats = ["highway_enc", "maxspeed", "lanes"]
    df = pd.get_dummies(
        df,
        columns=cats,
        drop_first=True,
    )

    # Impute    
    imputer = KNNImputer(n_neighbors=1)
    imputed = imputer.fit_transform(df)
    df["lanes"] = imputed[:, 1].astype(int)
    df["maxspeed"] = imputed[:, 2].astype(int)

    # Convert to PYG Data
    features = torch.tensor(df.astype(float).values, dtype=torch.float)
    data = Data(x=features, edge_index=edge_index)

    # I do not think we need to do that.
    #transform = T.Compose(
    #    [
    #        T.NormalizeFeatures(),
    #    ]
    #)
    #data = transform(data)
    return data