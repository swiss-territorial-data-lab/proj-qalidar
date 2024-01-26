import os
import argparse
import yaml
import pandas as pd
import laspy
import numpy as np
import open3d as o3d
import util_las as las
import pathlib
import geopandas as gpd
import time
import json
from shapely.geometry import Point

def bonus_shapefile_creation(df, out_dir,vox_dimension):
    geometry = [Point(xyz) for xyz in zip(df.X_grid, df.Y_grid, df.Z_grid)]
    gdf = gpd.GeoDataFrame(df, crs='EPSG:2056',geometry=geometry)
    gdf['geometry'] = gdf.geometry.buffer(vox_dimension/2, cap_style=3)   

    gdf.to_file(out_dir)


def main(OUTPUT_DIR, df, cfg, tile_name, vox_dimension):

     # Save the new dataframe as csv
    saving_time = time.strftime("%d%m-%H%M")
    # Create folder which will store all of the visualisation output
    subfolder_path = os.path.join(OUTPUT_DIR,tile_name+f'_{saving_time}')

    pathlib.Path(subfolder_path).mkdir(parents=True, exist_ok=True)

    # --- Save to LAS format ---
    if cfg['visualisation']['format']['LAS']['save']:
        las_cfg = cfg['visualisation']['format']['LAS']

        if las_cfg['fields']['use_default']:
            las_file = las.df_to_las(df)
        else:
            user_data_field = las_cfg['fields']['user_data']
            point_source_id_field = las_cfg['fields']['point_source_id']
            intensity_field = las_cfg['fields']['intensity']
            
            las_file = las.df_to_las(df, user_data_field,point_source_id_field, intensity_field)
        
        las_file.write(os.path.join(subfolder_path, 'change_detection.las'))
    
    # --- Save to shapefile format ---
    if cfg['visualisation']['format']['shapefile']['save']:
        shapefile_cfg = cfg['visualisation']['format']['shapefile']

        change_df = df[df.clusters>1]

        geometry = [Point(xy) for xy in zip(change_df.X_grid, change_df.Y_grid)]
        gdf_change = gpd.GeoDataFrame(change_df[['clusters','cluster_criticity_label']], crs='EPSG:2056',geometry=geometry)
        gdf_change.rename(columns={'cluster_criticity_label':'change_tag'},inplace=True)
        gdf_change['geometry'] = gdf_change.geometry.buffer(vox_dimension/2, cap_style=3)   

        gdf_dissolved = gdf_change.dissolve(by=['clusters'])

        gdf_dissolved.to_file(os.path.join(subfolder_path,'prioritary_changes.shp'))

        if shapefile_cfg['from_all_problematic']:
            all_problematic = df[df.change_criticity=='problematic']
            out_path = os.path.join(subfolder_path,'all_problematic.shp')
            bonus_shapefile_creation(all_problematic, out_path, vox_dimension)
        
        if shapefile_cfg['from_all_grey_zone']:
            all_grey_zone = df[df.change_criticity=='grey_zone']
            out_path = os.path.join(subfolder_path,'all_grey_zone.shp')
            bonus_shapefile_creation(all_grey_zone, out_path, vox_dimension)


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description="This script saves the detections in a format allowing visualisation.")
    parser.add_argument('-cfg', type=str, help='a YAML config file', default="./config_test.yml")
    args = parser.parse_args()


    with open(args.cfg) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    
    WORKING_DIR = cfg['working_dir']
    DF_PATH = cfg['visualisation']['data']['df_path']
    OUTPUT_DIR = cfg['visualisation']['output']['dir']

    tile_name, voxel_dimension, _ = os.path.basename(DF_PATH).split('.')[0].rsplit('_',maxsplit=2)
    voxel_dimension = float(voxel_dimension)/100 

    df = pd.read_csv(DF_PATH)

    df = main(OUTPUT_DIR, df, cfg,tile_name, voxel_dimension)

    print(f'Created visualisation file for tile {tile_name} saved under {OUTPUT_DIR}')

    print(f'\nFinished entire process in: {round(time.time()-start_time, 2)} sec.')