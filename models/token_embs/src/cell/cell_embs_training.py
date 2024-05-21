import os
import sys
from pathlib import Path
sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent))

import torch
import pandas as pd
import numpy as np
import pickle

from model_classes import SkipGramModel
from pipelines.utils import ROOT_DIR, load_config
from models.utils import meters2lonlat, lonlat2meters


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')



def main():
    data_config = load_config(name='porto', ctype="dataset")

    # Load CellSpace
    dataset_cell_file = f"{data_config['city']}_cell{int(data_config['cell_size'])}_cellspace.pkl"
    file_path = os.path.join(ROOT_DIR, "models/road_embs/", dataset_cell_file)
    with open(file_path, 'rb') as fh:
        cs = pickle.load(fh)

    # Load Feats Matrix
    feats_mx_file = f"{data_config['city']}_cell{int(data_config['cell_size'])}_feats_mx.pkl"
    file_path = os.path.join(ROOT_DIR, "models/road_embs/", feats_mx_file)
    with open(file_path, 'rb') as fh:
        feats_mx = pickle.load(fh)

    # Create Graph From Grid Cells
    _, edge_index = cs.all_neighbour_cell_pairs_permutated_optmized()

    # To torch tensor
    feats_mx_torch = torch.tensor(feats_mx, dtype = torch.long, device = device)
    edge_index_torch = torch.tensor(edge_index, dtype = torch.long, device = device).T

    # Create PYG Data Set
    from torch_geometric.data import Data
    data = Data(x=feats_mx_torch, edge_index=edge_index_torch)

    # Create a fake weighted adj mx
    import networkx as nx
    G = nx.Graph(edge_index)
    adj = nx.adjacency_matrix(G)
    adj = adj.toarray()
    row_sums = adj.sum(axis=1)
    adj = adj / row_sums[:, np.newaxis]

    traj2vec = SkipGramModel(
            data,
            adj=adj,
            device=device,
            emb_dim=128,
            walk_length=50,#30
            context_size=10, #5
            walks_per_node=10, #25
            num_neg=10,
        )

    traj2vec.train(epochs=30)

    model_emb = traj2vec.load_emb()

    print(model_emb.shape)

    # Save it for trajectory embeddings
    save_path ="trajemb/models/road_embs"

    # Safe embeddings
    model_emb = torch.from_numpy(model_emb)
    model_name = "skipgram"
    city = "porto"
    add = f""
    embs_file = f"{city}_cell_embs_{model_name}{add}.pkl"
    embs_file = os.path.join(save_path, embs_file)
    with open(embs_file, 'wb') as fh:
        pickle.dump(model_emb, fh, protocol = pickle.HIGHEST_PROTOCOL)
        print("Saved to: ", embs_file)




if __name__ == '__main__':
    main()