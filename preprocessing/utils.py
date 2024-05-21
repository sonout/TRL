import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import math
import swifter
from shapely import wkt
from shapely.geometry import LineString


def porto_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    def convert_to_line_string(x):
        m = np.matrix(x).reshape(-1, 2)
        if m.shape[0] >= 3:
            line = LineString(m)
            return line
        return -1

    df = df[df["MISSING_DATA"] == False]
    df["POLYLINE"] = df["POLYLINE"].apply(convert_to_line_string)
    df = df[df["POLYLINE"] != -1]

    df["POLYLINE"] = df["POLYLINE"].swifter.apply(wkt.loads)
    df["coords"] = df["POLYLINE"].swifter.apply(lambda x: list(x.coords))
    df["timestamps"] = df.swifter.apply(
        lambda x: list(
            np.arange(x.TIMESTAMP, x.TIMESTAMP + ((len(x.coords) - 1) * 15) + 1, 15)
        ),
        axis=1,
    )

    return df


PREPROCESS_MAP = {"porto": porto_preprocessing}


def create_edge_emb_mapping(line_graph, edge_df: pd.DataFrame):
    map = {}
    nodes = list(line_graph.nodes)
    for index, id in zip(edge_df.index, edge_df.fid):
        map[id] = nodes.index(index)

    return map



# This function takes in a df of trajectories and the edge_df of the road network
# Then it calculates for each edge in edge_df the mean traffic speed and utilization
def generate_speed_features(df, edge_df) -> pd.DataFrame:
        """
        Generates features containing average speed, utilization and accelaration
        for each edge i.e road segment.

        Returns:
            pd.DataFrame: features in shape num_edges x features
        """
        rdf = pd.DataFrame({"id": edge_df.fid}, index=edge_df.index)
        print(rdf.head())
        # calculate utilization on each edge which is defined as the count an edge is traversed by all trajectories
        seg_seqs = df["cpath"].values
        counter = Counter()
        for seq in seg_seqs:
            counter.update(Counter(seq))

        rdf["util"] = rdf.id.map(counter)

        # rdf["util"] = (rdf["util"] - rdf["util"].min()) / (
        #    rdf["util"].max() - rdf["util"].min()
        # )  # min max normalization

        # generate average speed feature
        # little bit complicater

        # key: edge_id, value: tuple[speed, count]
        cpaths = df["cpath"].values
        opaths = df["opath"].values
        speeds = df["speed"].values
        speed_counter = Counter()
        count_counter = Counter()

        for opath, cpath, speed in tqdm(zip(opaths, cpaths, speeds)):
            last_lidx, last_ridx = 0, 0
            for l, r, s in zip(opath[0::1], opath[1::1], speed):
                # print(l, r, s)
                lidxs, ridxs = np.where(cpath == l)[0], np.where(cpath == r)[0]
                lidx, ridx = (
                    lidxs[lidxs >= last_lidx][0],
                    ridxs[ridxs >= last_ridx][0],
                )
                assert lidx <= ridx
                traversed_edges = cpath[lidx : ridx + 1]
                # print(traversed_edges)
                speed_counter.update(
                    dict(zip(traversed_edges, [s] * len(traversed_edges)))
                )
                count_counter.update(
                    dict(zip(traversed_edges, [1] * len(traversed_edges)))
                )
                last_lidx, last_ridx = lidx, ridx

        rdf["avg_speed"] = rdf.id.map(
            {
                k: (float(speed_counter[k]) / count_counter[k]) * 111000 * 3.6
                for k in speed_counter
            }
        )  # calculate average speed in km/h

        # rdf["avg_speed"] = (rdf["avg_speed"] - rdf["avg_speed"].min()) / (
        #    rdf["avg_speed"].max() - rdf["avg_speed"].min()
        # )
        rdf['avg_util'] = rdf['util'] / rdf['util'].max()
        return rdf



