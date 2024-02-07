import os
import sys
import argparse
import yaml
import pandas as pd
import pathlib
import geopandas as gpd
import time
from shapely.geometry import Point

sys.path.append(".") 
import util_las as las

def bonus_shapefile_creation(df, out_dir, vox_dimension):
    '''
    Creates a shapefile with a shape for every row of the given DataFrame
    '''
    geometry = [Point(xyz) for xyz in zip(df.X_grid, df.Y_grid, df.Z_grid)]
    gdf = gpd.GeoDataFrame(df['criticality_number'], crs='EPSG:2056',geometry=geometry)
    gdf.rename(columns={'criticality_number':'#critical'},inplace=True)
    gdf['geometry'] = gdf.geometry.buffer(vox_dimension/2, cap_style=3)   

    gdf.to_file(out_dir)

def main(OUTPUT_DIR, df, cfg, tile_name, vox_dimension):

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
        
        saving_dir_las = os.path.join(OUTPUT_DIR, 'LAS')
        pathlib.Path(saving_dir_las).mkdir(parents=True, exist_ok=True) #Creates the folder if doesn't exist
        las_file.write(os.path.join(saving_dir_las, f'{tile_name}_change_detections.las'))
    
    # --- Save to shapefile format ---
    if cfg['visualisation']['format']['shapefile']['save']:
        shapefile_cfg = cfg['visualisation']['format']['shapefile']

        change_df = df[df.clusters>1]

        geometry = [Point(xy) for xy in zip(change_df.X_grid, change_df.Y_grid)]
        gdf_change = gpd.GeoDataFrame(change_df[['clusters','cluster_criticality_number']], crs='EPSG:2056',geometry=geometry)
        gdf_change.rename(columns={'cluster_criticality_number':'change_tag'},inplace=True)
        gdf_change['geometry'] = gdf_change.geometry.buffer(vox_dimension/2, cap_style=3)   

        gdf_dissolved = gdf_change.dissolve(by=['clusters'])

        saving_dir_shp1 = os.path.join(OUTPUT_DIR, 'priority_change_shp')
        pathlib.Path(saving_dir_shp1).mkdir(parents=True, exist_ok=True) #Creates the folder if doesn't exist
        gdf_dissolved.to_file(os.path.join(saving_dir_shp1,f'{tile_name}_priority_changes.shp'))

        if shapefile_cfg['from_all_problematic']:
            all_problematic = df[df.criticality_tag=='problematic']
            saving_dir_shp2 = os.path.join(OUTPUT_DIR, 'all_problematic_shp')
            pathlib.Path(saving_dir_shp2).mkdir(parents=True, exist_ok=True) #Creates the folder if doesn't exist
            out_path = os.path.join(saving_dir_shp2,f'{tile_name}_all_problematic.shp')
            bonus_shapefile_creation(all_problematic, out_path, vox_dimension)
        
        if shapefile_cfg['from_all_grey_zone']:
            all_grey_zone = df[df.criticality_tag=='grey_zone']
            saving_dir_shp3 = os.path.join(OUTPUT_DIR, 'all_grey_zone_shp')
            pathlib.Path(saving_dir_shp3).mkdir(parents=True, exist_ok=True) #Creates the folder if doesn't exist
            out_path = os.path.join(saving_dir_shp3,f'{tile_name}_all_grey_zone.shp')
            bonus_shapefile_creation(all_grey_zone, out_path, vox_dimension)

    if cfg['visualisation']['format']['csv']:
        saving_dir_csv = os.path.join(OUTPUT_DIR, 'dataframes')
        pathlib.Path(saving_dir_csv).mkdir(parents=True, exist_ok=True) #Creates the folder if doesn't exist
        df.to_csv(os.path.join(saving_dir_csv, f'{tile_name}_change_detections.csv'), index=False)
    


if __name__ == '__main__':
    start_time = time.time()
    print('Starting visualisation process...')

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

    print(f'\nCreated visualisation file for tile {tile_name}, saved under {OUTPUT_DIR}')

    print(f'\nFinished entire process in: {round(time.time()-start_time, 2)} sec.')