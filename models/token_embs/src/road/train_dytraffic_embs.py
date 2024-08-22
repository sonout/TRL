import os
import sys

import json
import torch

import pandas as pd
import numpy as np
import networkx as nx


#sys.path.append("../../../..")
# Append root of project to PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from pipelines.utils import DATASET_NAME, ROOT_DIR, TRAIN_CONFIG_KEYS, load_road_network
from road_emb_models import GAEModel, GATEncoder, Node2VecModel, SFCModel
from utils import create_pyg_data, generate_node_traj_adj, generate_trajid_to_nodeid
import argparse





def main(args):

    device = args.device
    city = args.city
    edge_df, _, _, LG = load_road_network(city_name=city)

    
    #### Features 

    #########################################################################
    # Load Dynamic Traffic Mx
    traffic_mx = pd.read_parquet(os.path.join(ROOT_DIR,"datasets/transition",city,"traffic_mx.parquet"))
    additional_cols =  ["length"]
    traffic_mx = pd.concat([traffic_mx, edge_df[additional_cols]], axis=1)

    # Create train_data tensor
    train_data_list = []
    for hour in range(1, 25):
        hour_col = f'avg_speed_{hour}'
        temp_df = traffic_mx[[hour_col] + additional_cols]
        for col in [hour_col] + additional_cols:
            temp_df = normalize_column(temp_df, col)
        train_data_list.append(temp_df.astype(float).values)
    train_data = torch.tensor(np.stack(train_data_list, axis=1), dtype=torch.float)

    if args.calc_adj:
        # load train data
        train = pd.read_parquet(os.path.join(ROOT_DIR,"datasets/trajectory",city,"train/train_{}.parquet".format(123),))
        val = pd.read_parquet(os.path.join(ROOT_DIR,"datasets/trajectory",city,"val/val_{}.parquet".format(123),))
        traj_df = pd.concat([train, val])

        traj_to_node = generate_trajid_to_nodeid(edge_df, LG)
        traj_data = traj_df.cpath.tolist()
        print("Generating adjacency matrix")
        adj = generate_node_traj_adj(LG, traj_data, traj_to_node, k=2, bidirectional=False, add_self_loops=True)
        np.savetxt(os.path.join(ROOT_DIR,"datasets/transition",city, f"tran_prob_mx.gz" ), X=adj)
    else:

        adj = np.loadtxt(os.path.join(ROOT_DIR,"datasets/transition",city, f"tran_prob_mx.gz" ))


    model = SFCModel(train_data.shape[-1], adj=adj, device=device, emb_dim=args.emb_dim, layers=1)
    model.train(train_data, epochs=args.epochs)
    #model.save_model(filepathpath=os.path.join(ROOT_DIR,"models/states/other",f"{city}_sfc_{args.emb_dim}.gz" ))

    dytraffic_embs = model.embed(train_data).detach().to('cpu')
    print(dytraffic_embs.shape)
    # store embeddings save space
    torch.save(dytraffic_embs, os.path.join(ROOT_DIR, "models/token_embs/road", f"{city}_sfc24_{args.emb_dim}.pt"))
    #np.savetxt(os.path.join(ROOT_DIR,"models/token_embs/road",f"{city}_sfc24_{args.emb_dim}.gz"), X=dytraffic_embs)

    
def normalize_column(df, column_name):
    df[column_name] = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train road embedding models")
    parser.add_argument("--city", type=str, default="sf", help="City name")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device for training")
    parser.add_argument("--calc_adj", type=bool, default=False, help="Calculate adjacency matrix")
    parser.add_argument("--emb_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=10000, help="Epochs for training")
    args = parser.parse_args()

    main(args)


