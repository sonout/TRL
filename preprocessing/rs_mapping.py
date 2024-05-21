import os
import sys

sys.path.append("..")

from ast import literal_eval

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipelines.utils import PLAIN_DATASET_NAME, PRE_MAP_DATASET_NAME, ROOT_DIR

from .utils import PREPROCESS_MAP



## This function just saves preprocessed_dataset into right format for FMM
## E.g. final columns: [TRIP_ID;POLYLINE;timestamps;id]
def create_road_mapping_df(
    df: pd.DataFrame,
    city_name: str,
    need_preprocessing: bool = False,
):
    # load original dataset
    #df = pd.read_parquet(
    #    os.path.join(ROOT_DIR, "datasets/trajectory", city_name, PLAIN_DATASET_NAME)
    #)

    if need_preprocessing:
        df = PREPROCESS_MAP[city_name](df)

    input_file = os.path.join(
        ROOT_DIR, "datasets/trajectory", city_name, PRE_MAP_DATASET_NAME
    )
    df_fmm = df.loc[:, ["TRIP_ID", "POLYLINE", "timestamps"]]
    df_fmm["id"] = np.arange(0, df_fmm.shape[0])
    df_fmm["timestamps"] = df_fmm["timestamps"].str.replace("[", "")
    df_fmm["timestamps"] = df_fmm["timestamps"].str.replace("]", "")
    print(f"Saving preprocessed dataset to {input_file}")
    df_fmm.to_csv(input_file, sep=";", index=False)


def merge_preprocessed_and_fmm(df_preprocessed, df_fmm):
    assert len(df_preprocessed) == len(df_fmm) # FMM might not match all trips, but should still have the same length as input data
    df_merged = pd.merge(df_preprocessed, df_fmm, left_index=True, right_on="id")

    df = df_merged[df_merged.cpath.notnull()]
    df = df.reset_index(drop=True)

    df = df[['TRIP_ID', 'TAXI_ID', 'POLYLINE', 'timestamps', 'trajlen', 'coord_seq',
        'merc_seq', 'opath', 'spdist',
        'cpath', 'tpath', 'length', 'duration', 'speed']]
    
    # Reset index and drop old
    df = df.reset_index(drop=True)
    
    print(f"Before FMM: {len(df_preprocessed)} trips. After FMM: {len(df)} trips. Percentage mapped: {len(df)/len(df_preprocessed)}")
    return df


def post_processing_mapped_df(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, "cpath"] = df.cpath.apply(literal_eval)
    df.loc[:, "opath"] = df.opath.apply(literal_eval)
    df.loc[:, "tpath"] = df.loc[:, "tpath"] = df.tpath.apply(lambda x: [[int(z) for z in y.split(",")] for y in x.split("|")])
    # Next are not too important, but might be usefull
    df.loc[:, "speed"] = df.speed.apply(literal_eval)
    df.loc[:, "duration"] = df.duration.apply(literal_eval)
    df.loc[:, "length"] = df.length.apply(literal_eval)
    df.loc[:, "spdist"] = df.spdist.apply(literal_eval)
    

    df = df[df.cpath.str.len() > 2]

    # calculate timestamps for each road segment inside the trajectories
    df = _calculate_timestamps_for_road_segments(df)

    return df


def _calculate_timestamps_for_road_segments(df: pd.DataFrame) -> pd.DataFrame:
    # get information needed for calculation
    cpaths = df.cpath.values
    tpaths = df.tpath.values
    timestamps = df["timestamps"].values
    road_stamps = []
    
    # iterate over each row of the input DataFrame
    for cpath, tpath, timestamp in tqdm(zip(cpaths, tpaths, timestamps)):
        timestamp = np.asarray(timestamp)

        mapped_timestamps = [timestamp[0]]
        lastly_traversed = {}
        
        # Print lengths of cpath, tpath and timestamp
        #print(f"Length of cpath: {len(cpath)}, Length of tpath: {len(tpath)}, Length of timestamp: {len(timestamp)}")

        # iterate over the tpath and timestamps
        for i, (t1, t2) in enumerate(zip(timestamp[0::1], timestamp[1::1])):
            traversed = tpath[i]
            tcount = len(traversed)

            # calculate the time difference and proportion
            diff = t2 - t1
            prop = diff / tcount
            
            # generate an array of timestamps for the traversed edges
            stamps = np.arange(t1, t2 + 1, prop)[-tcount:].astype(int)
            stamps = np.clip(stamps, t1, t2)

            temp_lastly = {}
            
            # check for intersection road and update road timestamps accordingly
            for j, (r, t) in enumerate(zip(traversed, stamps)):
                if r in lastly_traversed.keys() and j == 0:
                    idx = lastly_traversed[r]
                    mapped_timestamps[idx] = t
                    continue

                mapped_timestamps.append(t)

            temp_lastly[traversed[-1]] = len(mapped_timestamps) - 1
            lastly_traversed = temp_lastly

        assert len(mapped_timestamps) == len(cpath) + 1, f"Length not match: {len(mapped_timestamps)} != {len(cpath) + 1}"
        
        # append the calculated road timestamps to a list
        road_stamps.append(mapped_timestamps)

    # add a new column road_timestamps to the input DataFrame
    df["road_timestamps"] = road_stamps

    # return a new DataFrame with columns ["id", "cpath", "road_timestamps"]
    return df

