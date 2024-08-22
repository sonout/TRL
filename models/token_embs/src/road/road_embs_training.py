import os
import sys

import json
import torch

import pandas as pd
import numpy as np
import networkx as nx


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from pipelines.utils import DATASET_NAME, ROOT_DIR, TRAIN_CONFIG_KEYS, load_road_network
from road_emb_models import Traj2VecModel
from utils import create_pyg_data, generate_node_traj_adj, generate_trajid_to_nodeid
import argparse
import pickle




def main(args):

    device = args.device
    city = args.city
    edge_df, _, _, LG = load_road_network(city_name=city)

    
    #### Features 
    if args.calc_adj:
        # load train data
        train = pd.read_parquet(os.path.join(ROOT_DIR,"datasets/trajectory",city,"train/train_{}.parquet".format(123),))
        val = pd.read_parquet(os.path.join(ROOT_DIR,"datasets/trajectory",city,"val/val_{}.parquet".format(123),))
        traj_df = pd.concat([train, val])

        traj_to_node = generate_trajid_to_nodeid(edge_df, LG)
        traj_data = traj_df.cpath.tolist()
        print("Generating adjacency matrix")
        adj = generate_node_traj_adj(LG, traj_data, traj_to_node, k=1, bidirectional=False, add_self_loops=False)
        np.savetxt(os.path.join(ROOT_DIR,"datasets/transition",city, f"tran_prob_mx_noselfloops.gz" ), X=adj)
    else:

        adj = np.loadtxt(os.path.join(ROOT_DIR,"datasets/transition",city, f"tran_prob_mx_noselfloops.gz" ))

    map_id = {j: i for i, j in enumerate(LG.nodes)}
    edge_list = nx.to_pandas_edgelist(LG)
    edge_list["sidx"] = edge_list["source"].map(map_id)
    edge_list["tidx"] = edge_list["target"].map(map_id)

    edge_index = np.array(edge_list[["sidx", "tidx"]].values).T
    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

    traj2vec = Traj2VecModel(
            edge_index,
            adj=adj,
            device=device,
            emb_dim=64,
            walk_length=30,
            context_size=5,
            walks_per_node=25,
            num_neg=10,
        )
    traj2vec.train(args.epochs)
    #model.save_model(filepathpath=os.path.join(ROOT_DIR,"models/states/other",f"{city}_sfc_{args.emb_dim}.gz" ))

    road_embs = traj2vec.load_emb()
    road_embs = torch.from_numpy(road_embs)
    embs_file = f"{city}_road_embs_traj_skipgram_{args.emb_dim}.pkl"
    embs_file = os.path.join(ROOT_DIR, "models/token_embs/road", embs_file)

    with open(embs_file, 'wb') as fh:
        pickle.dump(road_embs, fh, protocol = pickle.HIGHEST_PROTOCOL)
        print("Saved to: ", embs_file)

    
def normalize_column(df, column_name):
    df[column_name] = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train road embedding models")
    parser.add_argument("--city", type=str, default="sf", help="City name")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device for training")
    parser.add_argument("--calc_adj", type=bool, default=True, help="Calculate adjacency matrix")
    parser.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=10000, help="Epochs for training")
    args = parser.parse_args()

    main(args)


