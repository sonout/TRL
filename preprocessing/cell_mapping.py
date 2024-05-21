import os
import sys

sys.path.append("..")

from ast import literal_eval

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import pickle
import time
import math
import swifter


from pipelines.utils import PLAIN_DATASET_NAME, PRE_MAP_DATASET_NAME, ROOT_DIR
from models.baselines.trajcl_files.cellspace import CellSpace
from models.baselines.trajcl_files.node2vec import train_node2vec

from .utils import PREPROCESS_MAP


# ref: TrjSR
def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))

def inrange(lon, lat, config):
    if lon <= config['min_lon'] or lon >= config['max_lon'] \
            or lat <= config['min_lat'] or lat >= config['max_lat']:
        return False
    return True

def clean_and_output_data(dfraw, config):
    _time = time.time()

    #dfraw = pd.read_csv(ROOT_DIR + '/data/porto.csv')
    #dfraw = dfraw.rename(columns = {"POLYLINE": "coord_seq"})
    print('Full length. #traj={}'.format(dfraw.shape[0]))

    if config['city'] == 'porto':
        dfraw = dfraw[dfraw.MISSING_DATA == False]
        print('Removed trajs with MISSING DATA = True. #traj={}'.format(dfraw.shape[0]))

    dfraw['coord_seq'] = dfraw.coords.apply(literal_eval if isinstance(dfraw.coords[0], str) else lambda x: x) 


    # Edit 22.10.23: Problem for SF too many traj where outside of range (~40%).
    #               Looked like there were a lot of outliers, which were removed by the range requirement.
    #               However, after removing outliers still a lot of trajectories have been removed (~20-30%)
    # dfraw['coord_seq'] = dfraw.coord_seq.map(lambda traj: [p for p in traj if inrange(p[0], p[1], config)])
    #               Found out, that the lon/lat where swapped for some trajectories/ points within trajectories
    #               Thus we check for each point if it is correct, else swap lon/lat
    dfraw['coord_seq'] = dfraw.coord_seq.map(lambda traj: [p if ((p[1] > 0) & (p[1] < 90)) else (p[1], p[0]) for p in traj])
    # range requirement
    dfraw['inrange'] = dfraw.coord_seq.map(lambda traj: sum([inrange(p[0], p[1], config) for p in traj]) == len(traj) ) # True: valid
    dfraw = dfraw[dfraw.inrange == True]
    print('Preprocessed-rm range. #traj={}'.format(dfraw.shape[0]))

    # length requirement
    dfraw.loc[:, 'trajlen'] = dfraw.coord_seq.swifter.apply(lambda traj: len(traj))
    dfraw = dfraw[(dfraw.trajlen >= config['min_traj_len']) & (dfraw.trajlen <= config['max_traj_len'])]
    print('Preprocessed-rm length. #traj={}'.format(dfraw.shape[0]))


    # convert to Mercator
    dfraw.loc[:, 'merc_seq'] = dfraw.coord_seq.swifter.apply(lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj])

    print('Preprocessed-output. #traj={}'.format(dfraw.shape[0]))
    

    if config['city'] == 'sf':
        dfraw = dfraw[['TRIP_ID','TAXI_ID','POLYLINE', 'timestamps','trajlen', 'coord_seq', 'merc_seq', 'occupied']].reset_index(drop = True)
    else:
        dfraw = dfraw[['TRIP_ID','TAXI_ID','POLYLINE', 'timestamps','trajlen', 'coord_seq', 'merc_seq']].reset_index(drop = True)

    # timestamps column, string to list
    dfraw.loc[:, 'timestamps'] = dfraw.timestamps.apply(literal_eval if isinstance(dfraw.timestamps[0], str) else lambda x: x)
    
    print('Preprocess end. @={:.0f}'.format(time.time() - _time))
    return dfraw

def init_cellspace(config: dict):
    # 1. create cellspase
    # 2. initialize cell embeddings (create graph, train, and dump to file)

    x_min, y_min = lonlat2meters(config['min_lon'], config['min_lat'])
    x_max, y_max = lonlat2meters(config['max_lon'], config['max_lat'])
    x_min -= config['cellspace_buffer']
    y_min -= config['cellspace_buffer']
    x_max += config['cellspace_buffer']
    y_max += config['cellspace_buffer']

    cell_size = int(config['cell_size'])
    cs = CellSpace(cell_size, cell_size, x_min, y_min, x_max, y_max)
    with open(config['dataset_cell_file'], 'wb') as fh:
        pickle.dump(cs, fh, protocol = pickle.HIGHEST_PROTOCOL)

    _, edge_index = cs.all_neighbour_cell_pairs_permutated_optmized()
    edge_index = torch.tensor(edge_index, dtype = torch.long, device = config['device']).T
    train_node2vec(edge_index)
    return