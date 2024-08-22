import sys
import os
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import swifter
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from operator import itemgetter


sys.path.append("../..")
 
from . import transforms as T
from pipelines.utils import ROOT_DIR, load_config
from models.utils import lonlat2meters, merc2cell2
from models.proposed.time2vec.t2v import Time2VecConvert
from models.token_embs.src.cell.init_cs import init_cs

class TIGRDataset(Dataset):
    def __init__(self, data, edge_df, line_graph, config=None, train=True):
        self.edge_df = edge_df
        self.line_graph = line_graph
        self.seq_len = 200 # For cells we filter >200 in preprocessing, I would like remove that and do this here as well
        gpu_id = config['gpu_device_ids'][0]
        city = config['city']
        config = config['model_args']

        #### ROAD 1 ####
        self.road_trajs = data["cpath"].values  # trajectory
        self.traj_map = self._create_edge_emb_mapping()
        #self.road_emb1 = pickle.load(open(os.path.join(ROOT_DIR, config["road_emb_path1"]), 'rb'))

        self.dytraffic_embs = torch.load(os.path.join(ROOT_DIR, config["dytraffic_embs_path"]), map_location='cpu')

        #### ROAD 2 ####
        self.road_emb2 = pickle.load(open(os.path.join(ROOT_DIR, config["road_emb_path2"]), 'rb'))

        #### CELL ####
        # Create Mercator Seq.
        data['merc_seq'] = data.coord_seq.swifter.apply(lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj])

        # We only need gps traj and merc_seq
        self.cell_data = data[['coord_seq','merc_seq']]
        #self.data['merc_seq'] = self.data['merc_seq'].apply(lambda x: [list(y) for y in x])

        cell_embs_filepath = os.path.join(ROOT_DIR, config["model_files_path"], config["cell_embs_file"])
        self.cell_embs = pickle.load(open(cell_embs_filepath, 'rb')).to('cpu').detach() # tensor

        data_config = load_config(name=city, ctype="dataset")
        self.cellspace = init_cs(data_config['min_lon'], data_config['min_lat'], data_config['max_lon'], data_config['max_lat'], data_config['cellspace_buffer'], data_config['cell_size'])


        ### TIME ###
        #time_embs_filepath = os.path.join(ROOT_DIR, config["time_emb_path"], f"{city}_time_embs_{len(data)}.pkl")
        if data.road_timestamps.values[0][0] == data.road_timestamps.values[0][-1]:
            print("WE ESTIMEATE TIMESTAMPS!!")
            start_times = [seq[0] for seq in data['road_timestamps'].values]
            travel_times = edge_df['traveltime'].values

            avg_road_timestamps = []
            # Iterate over each sequence in data["cpath"].values
            for i, traj_idxs in tqdm(enumerate(self.road_trajs)):
                segment_times = travel_times[list(traj_idxs)]
                future_timestamps = start_times[i] + np.cumsum(np.concatenate(([0], segment_times)))
                avg_road_timestamps.append(future_timestamps)
            
            data['road_timestamps'] = avg_road_timestamps 

        # Transform 
        vec_dt = np.vectorize(datetime.datetime.fromtimestamp)
        self.time_stamps = data.road_timestamps.values
        self.time_vec_list = [] # List of time vectors
        for time_stamps_traj in tqdm(self.time_stamps):
            time_dt = vec_dt(time_stamps_traj[:-1]) # Note: road_timestamps are one more than cpath
            time_feats_list = [[t.hour, t.minute, t.second, t.weekday()] for t in time_dt]
            self.time_vec_list.append(time_feats_list)
        
        
        
        ### Transforms & Collate ###
        view1 = config["view1"]
        view2 = config["view2"] 
        p = config["aug_prob"]

        transform_dict = {
            "mask": [T.Mask(p=p), T.Mask_with_time(p=p)],
            "trim": [T.Subset(p=p), T.Subset_with_time(p=p)],
            "cutout": [T.ConsecutiveMasking(p=p), T.ConsecutiveMaskingWithTime(p=p)],
            "cut": [T.ConsecutiveMasking(p=p), T.ConsecutiveMaskingWithTime(p=p)],
        }

        self.transform1 = T.Compose([transform_dict[aug][0] for aug in view1])
        self.transform1_w_time = T.Compose_with_time([transform_dict[aug][1] for aug in view1])
        self.transform2 = T.Compose([transform_dict[aug][0] for aug in view2])
        self.transform2_w_time = T.Compose_with_time([transform_dict[aug][1] for aug in view2])

        self.collate_custom = CustomCollateFunction(self.dytraffic_embs, self.road_emb2, self.cellspace, self.cell_embs, self.transform1, self.transform2, self.transform1_w_time, self.transform2_w_time, config)

    def __len__(self):
        return self.road_trajs.shape[0]

    def __getitem__(self, idx):
        ## ROAD ##
        # This is the cpath values/edge_df idx
        traj_idxs = self.road_trajs[idx]
        traj_time_vec = self.time_vec_list[idx]

        if len(traj_idxs) > self.seq_len:
            traj_idxs, traj_time_vec = self.cut_traj_time(traj_idxs, traj_time_vec)
            
        # This are the nodes in the line_graph
        road_traj = list(itemgetter(*traj_idxs)(self.traj_map))

        ## CELL ##
        cell_traj = self.cell_data.iloc[idx].merc_seq
        return road_traj, cell_traj, traj_time_vec

    # tested index mapping is correct
    def _create_edge_emb_mapping(self):
        map = {}
        nodes = list(self.line_graph.nodes)
        for index, id in zip(self.edge_df.index, self.edge_df.fid):
            map[id] = nodes.index(index)
        # print(map == map2) # yields true

        return map

    def cut_traj(self, traj):
        start_idx = int((len(traj) - self.seq_len) * random.random())
        return traj[start_idx : start_idx + self.seq_len]

    def cut_traj_time(self, traj, time_embs):
        start_idx = int((len(traj) - self.seq_len) * random.random())
        return traj[start_idx : start_idx + self.seq_len], time_embs[start_idx : start_idx + self.seq_len]
    
class CustomCollateFunction(nn.Module):
    def __init__(self, dytraffic_embs, road_emb2, cellspace, cell_embs, augfn1, augfn2, augfntime1, augfntime2, config) :

        super(CustomCollateFunction, self).__init__()
        self.dytraffic_embs = dytraffic_embs
        self.road_emb2 = road_emb2
        self.cellspace = cellspace
        self.cell_embs = cell_embs
        self.augfn1 = augfn1
        self.augfn2 = augfn2
        self.augfntime1 = augfntime1
        self.augfntime2 = augfntime2
        self.config = config

    def forward(self, batch):
        road_trajs, cell_trajs, time_embs = zip(*batch)

        ### ROAD 1 + Time ###
        road_trajs1, times1 = zip(*[self.augfntime1(t, time) for t, time in zip(road_trajs, time_embs)])
        road_trajs2, times2 = zip(*[self.augfntime2(t, time) for t, time in zip(road_trajs, time_embs)])


        road_trajs_emb = [self.dytraffic_embs[list(t), time_embs[i][0][0]] for i, t in enumerate(road_trajs)]
        road_trajs1_emb = [self.dytraffic_embs[list(t), times1[i][0][0]] for i, t in enumerate(road_trajs1)]
        road_trajs2_emb = [self.dytraffic_embs[list(t), times2[i][0][0]] for i, t in enumerate(road_trajs2)]

        road_trajs_emb = pad_sequence(road_trajs_emb, batch_first = True) # [seq_len, batch_size, emb_dim]
        road_trajs1_emb = pad_sequence(road_trajs1_emb, batch_first = True) # [seq_len, batch_size, emb_dim]
        road_trajs2_emb = pad_sequence(road_trajs2_emb, batch_first = True) # [seq_len, batch_size, emb_dim]

        road_trajs_len = torch.tensor(list(map(len, road_trajs)), dtype = torch.long)
        road_trajs1_len = torch.tensor(list(map(len, road_trajs1)), dtype = torch.long)
        road_trajs2_len = torch.tensor(list(map(len, road_trajs2)), dtype = torch.long)

        # time
        time_embs = [torch.as_tensor(t) for t in time_embs]
        time_embs = pad_sequence(time_embs, batch_first = True)
        times1 = [torch.as_tensor(t) for t in times1]
        times2 = [torch.as_tensor(t) for t in times2]
        times1 = pad_sequence(times1, batch_first = True)
        times2 = pad_sequence(times2, batch_first = True)

        ### ROAD 2 ###

        road2_trajsa = [self.augfn1(t) for t in road_trajs]
        road2_trajsb = [self.augfn2(t) for t in road_trajs]

        road2_trajs2_emb = [self.road_emb2[list(t)] for t in road_trajs]
        road2_trajs2a_emb = [self.road_emb2[list(t)] for t in road2_trajsa]
        road2_trajs2b_emb = [self.road_emb2[list(t)] for t in road2_trajsb]

        road2_trajs_emb = pad_sequence(road2_trajs2_emb, batch_first = True) # [seq_len, batch_size, emb_dim]
        road2_trajsa_emb = pad_sequence(road2_trajs2a_emb, batch_first = True) # [seq_len, batch_size, emb_dim]
        road2_trajsb_emb = pad_sequence(road2_trajs2b_emb, batch_first = True) # [seq_len, batch_size, emb_dim]

        road2_trajs_len = torch.tensor(list(map(len, road_trajs)), dtype = torch.long)
        road2_trajsa_len = torch.tensor(list(map(len, road2_trajsa)), dtype = torch.long)
        road2_trajsb_len = torch.tensor(list(map(len, road2_trajsb)), dtype = torch.long)

        ### CELL ###
        cell_trajs1 = [self.augfn1(t) for t in cell_trajs]
        cell_trajs2 = [self.augfn2(t) for t in cell_trajs]

        cell_trajs, trajs_p = zip(*[merc2cell2(t, self.cellspace) for t in cell_trajs])
        cell_trajs1, trajs1_p = zip(*[merc2cell2(t, self.cellspace) for t in cell_trajs1])
        cell_trajs2, trajs2_p = zip(*[merc2cell2(t, self.cellspace) for t in cell_trajs2])

        cell_trajs_emb = [self.cell_embs[list(t)] for t in cell_trajs]
        cell_trajs1_emb = [self.cell_embs[list(t)] for t in cell_trajs1]
        cell_trajs2_emb = [self.cell_embs[list(t)] for t in cell_trajs2]

        cell_trajs_emb = pad_sequence(cell_trajs_emb, batch_first = True) # [seq_len, batch_size, emb_dim]
        cell_trajs1_emb = pad_sequence(cell_trajs1_emb, batch_first = True) # [seq_len, batch_size, emb_dim]
        cell_trajs2_emb = pad_sequence(cell_trajs2_emb, batch_first = True) # [seq_len, batch_size, emb_dim]

        cell_trajs_len = torch.tensor(list(map(len, cell_trajs)), dtype = torch.long)
        cell_trajs1_len = torch.tensor(list(map(len, cell_trajs1)), dtype = torch.long)
        cell_trajs2_len = torch.tensor(list(map(len, cell_trajs2)), dtype = torch.long)

        # return: two padded tensors and their lengths
        return road_trajs1_emb, road_trajs1_len, road_trajs2_emb, road_trajs2_len, road_trajs_emb, road_trajs_len, \
            road2_trajsa_emb, road2_trajsa_len, road2_trajsb_emb, road2_trajsb_len, road2_trajs_emb, road2_trajs_len, \
                cell_trajs1_emb, cell_trajs1_len, cell_trajs2_emb, cell_trajs2_len, cell_trajs_emb, cell_trajs_len, \
                    times1, times2, time_embs, 