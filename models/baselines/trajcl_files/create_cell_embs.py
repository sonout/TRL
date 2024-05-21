import os
import sys
import pickle
import torch
import math

sys.path.append("../../..")

from cellspace import CellSpace
from node2vec import train_node2vec

from pipelines.utils import ROOT_DIR, load_config

# ref: TrjSR
def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))

def init_cellspace(cell_emb_dim, model_files_path, data_config: dict, device: torch.device, cellspace_path: str = None):
        # 1. create cellspase
        # 2. initialize cell embeddings (create graph, train, and dump to file)
        if cellspace_path is not None:
            print("Loading existing cellspace...")
            with open(cellspace_path, 'rb') as fh:
                cs = pickle.load(fh)
            print("Loaded cellspace from file: ", cellspace_path)
        else:
            print("No existing cellspace found. Creating new cellspace...")
            x_min, y_min = lonlat2meters(data_config['min_lon'], data_config['min_lat'])
            x_max, y_max = lonlat2meters(data_config['max_lon'], data_config['max_lat'])
            x_min -= data_config['cellspace_buffer']
            y_min -= data_config['cellspace_buffer']
            x_max += data_config['cellspace_buffer']
            y_max += data_config['cellspace_buffer']

            cell_size = int(data_config['cell_size'])
            cs = CellSpace(cell_size, cell_size, x_min, y_min, x_max, y_max)

            #dataset_cell_file = f"{data_config['city']}_cell{int(data_config['cell_size'])}_{config['dataset_cell_file']}"
            file_path = os.path.join(ROOT_DIR, model_files_path, f"{data_config['city']}_cell{int(data_config['cell_size'])}_cellspace.pkl")
            with open(file_path, 'wb') as fh:
                pickle.dump(cs, fh, protocol = pickle.HIGHEST_PROTOCOL)
            print("Saved cellspace to file: ", file_path)

        _, edge_index = cs.all_neighbour_cell_pairs_permutated_optmized()
        edge_index = torch.tensor(edge_index, dtype = torch.long, device = device).T
        train_node2vec(edge_index, cell_emb_dim, model_files_path, data_config, device)
        return


if __name__ == '__main__':
     data_config = load_config(name='sf', ctype="dataset")
     data_config['ROOT_DIR'] = ROOT_DIR
     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

     model_files_path = "models/token_embs/cell"
     cs_path = "/projects/trajemb/models/token_embs/cell/sf_cell100_cellspace.pkl"
     init_cellspace(cell_emb_dim=64,model_files_path=model_files_path, data_config=data_config, device=device, cellspace_path=cs_path)