import os
import sys
import pickle
import math
from tqdm import tqdm 

import osmnx as ox
import geopandas as gpd

from pathlib import Path
#sys.path.append("../..")
sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent))

print(sys.path)

from cellspace import CellSpace
from node2vec import train_node2vec

from pipelines.utils import ROOT_DIR, load_config
from models.utils import meters2lonlat, lonlat2meters



def main():
    data_config = load_config(name='porto', ctype="dataset")

    ### Cell Space
    x_min, y_min = lonlat2meters(data_config['min_lon'], data_config['min_lat'])
    x_max, y_max = lonlat2meters(data_config['max_lon'], data_config['max_lat'])
    x_min -= data_config['cellspace_buffer']
    y_min -= data_config['cellspace_buffer']
    x_max += data_config['cellspace_buffer']
    y_max += data_config['cellspace_buffer']

    cell_size = int(data_config['cell_size'])
    cs = CellSpace(cell_size, cell_size, x_min, y_min, x_max, y_max)
    cell_gdf = cs.get_celldf()

    # Safe Cell Space
    dataset_cell_file = f"{data_config['city']}_cell{int(data_config['cell_size'])}_cellspace.pkl"
    file_path = os.path.join(ROOT_DIR, "models/road_embs/", dataset_cell_file)
    with open(file_path, 'wb') as fh:
        pickle.dump(cs, fh, protocol = pickle.HIGHEST_PROTOCOL)
    print(f"Cell Space saved to {file_path}")

    ### POIS
    tags = {"healthcare": True, "amenity": True, "craft": True, "tourism": True, "office": True, "leisure": True, "shop": True, "building": True}
    poi_df = ox.features.features_from_bbox(data_config['min_lat'], data_config['max_lat'], data_config['min_lon'], data_config['max_lon'], tags)
    poi_df_procesed = preprocess_poi(poi_df)

    ### Feature Extraction
    
    ## 1. POI Category counts

    # Get the distinct values of the 'category' column in poi_df_procesed
    categories = poi_df_procesed['category'].unique()

    # Create a spatial join between cell_gdf and poi_df_procesed
    spatial_join = gpd.sjoin(cell_gdf, poi_df_procesed, how='left', predicate='intersects')

    # Group the spatial join by the 'cell_id' column and count the number of POIs of each category within each cell
    category_counts = spatial_join.groupby('cell_id')['category'].value_counts().unstack().fillna(0).astype(int)

    # Add the category counts as new columns in cell_gdf
    cell_gdf = cell_gdf.merge(category_counts, left_on='cell_id', right_index=True, how='left').fillna(0)

    ## 2. x_i, y_i (cell ids)
    # For features we want also to use the x_i, y_i from cell_tuple
    cell_gdf['x'] = cell_gdf['cell_tuple'].apply(lambda x: x[0])
    cell_gdf['y'] = cell_gdf['cell_tuple'].apply(lambda x: x[1])

    ### Numpy Matrix
    # Final cell feature matrix
    feats_mx = cell_gdf.iloc[:,4:].values

    # Safe Feats Matrix
    feats_mx_file = f"{data_config['city']}_cell{int(data_config['cell_size'])}_feats_mx.pkl"
    file_path = os.path.join(ROOT_DIR, "models/road_embs/", feats_mx_file)
    with open(file_path, 'wb') as fh:
        pickle.dump(feats_mx, fh, protocol = pickle.HIGHEST_PROTOCOL)
    print(f"Cell Features Matrix saved to {file_path}")


def preprocess_poi(poi_df, tags = ["healthcare", "amenity", "craft", "tourism", "office", "leisure", "shop", "building"]  ):
    
    poi_df_cat = poi_df[tags + ["geometry"]].copy()
    #poi_df_cat = poi_df_cat.loc["node", :]
    poi_df_cat.loc[:, "poi"] = poi_df_cat[tags].bfill(axis=1).iloc[:, 0]
    poi_df_cat.loc[:, "poi"] = poi_df_cat["poi"].astype('category')  
    poi_df_cat.loc[:, "category"] = poi_df_cat[tags].notnull().idxmax(axis=1)
    poi_df_cat = poi_df_cat[["poi", "category", "geometry"]].dropna(axis=0)
    # to gdf
    poi_df_cat = gpd.GeoDataFrame(poi_df_cat, geometry='geometry', crs='EPSG:4326')
    return poi_df_cat

if __name__ == '__main__':
    main()