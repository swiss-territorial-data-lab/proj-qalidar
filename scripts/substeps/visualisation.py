import argparse
import os
import pathlib
import sys
import time
import yaml
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

sys.path.append("scripts") 
import util_las as las
from constant import criticality_dict

def bonus_shapefile_creation(df, out_dir, vox_size):
    '''
    Creates a shapefile with a shape for every row of the given DataFrame
    '''
    geometry = [Point(xyz) for xyz in zip(df.X_grid, df.Y_grid, df.Z_grid)]
    df = get_description_for_numbers(df, 'criticality_number')
    gdf = gpd.GeoDataFrame(df[['criticality_number', 'desc']], crs='EPSG:2056', geometry=geometry)
    gdf.rename(columns={'criticality_number':'critic_nbr'}, inplace=True)
    gdf['geometry'] = gdf.geometry.buffer(vox_size/2, cap_style=3)   

    gdf.to_file(out_dir)


def get_description_for_numbers(df, criticality_number):

    numbers, descriptions = zip(*criticality_dict)
    criticality_descr_df = pd.DataFrame({'number': numbers, 'desc': descriptions})
    completed_df = df.merge(criticality_descr_df, how='left', left_on=criticality_number, right_on='number')
    
    return completed_df


def main(OUTPUT_DIR, df, cfg, tile_name, vox_size):
    CFG_FORMAT_TYPES = cfg['visualisation']['format']
    SAVE_AS_LAS = CFG_FORMAT_TYPES['LAS']['save']
    SAVE_AS_SHP = CFG_FORMAT_TYPES['shapefile']['save']
    SAVE_AS_CSV = CFG_FORMAT_TYPES['csv']['save']

    df = df.astype({'criticality_number':'int8', 'clusters': 'int32', 'cluster_criticality_number': 'int8'})

    # --- Save to LAS format ---
    if SAVE_AS_LAS:
        las_cfg = CFG_FORMAT_TYPES['LAS']

        if las_cfg['fields']['use_default']:
            las_file = las.df_to_las(df)
        else:
            user_data_field = las_cfg['fields']['user_data']
            point_source_id_field = las_cfg['fields']['point_source_id']
            intensity_field = las_cfg['fields']['intensity']
            
            las_file = las.df_to_las(df, user_data_field, point_source_id_field, intensity_field)
        
        saving_dir_las = os.path.join(OUTPUT_DIR, 'LAS')
        pathlib.Path(saving_dir_las).mkdir(parents=True, exist_ok=True)         # Creates the folder if doesn't exist
        las_file.write(os.path.join(saving_dir_las, f'{tile_name}_change_detections.las'))
    
    # --- Save to shapefile format ---
    if SAVE_AS_SHP:
        shapefile_cfg = CFG_FORMAT_TYPES['shapefile']

        change_df = df[df.clusters>1].copy()        # clusters==0 are non problematic voxels and clusters==1 are problematic, but isolated voxels
        change_df = get_description_for_numbers(change_df, 'cluster_criticality_number')
        
        geometry = [Point(xy) for xy in zip(change_df.X_grid, change_df.Y_grid)]
        gdf_change = gpd.GeoDataFrame(change_df[['clusters','cluster_criticality_number','cluster_distribution','desc']], crs='EPSG:2056', geometry=geometry)
        gdf_change.rename(columns={'cluster_criticality_number':'change_tag','cluster_distribution':'all_tags'},inplace=True)
        gdf_change['geometry'] = gdf_change.geometry.buffer(vox_size/2, cap_style=3)   

        gdf_dissolved = gdf_change.dissolve(by=['clusters'])

        saving_dir_shp1 = os.path.join(OUTPUT_DIR, 'priority_change_shp')
        pathlib.Path(saving_dir_shp1).mkdir(parents=True, exist_ok=True)        # Creates the folder if doesn't exist
        gdf_dissolved.to_file(os.path.join(saving_dir_shp1, f'{tile_name}_priority_changes.shp'))

        if shapefile_cfg['from_all_problematic']:
            all_problematic = df[df.criticality_tag=='problematic'].copy()
            saving_dir_shp2 = os.path.join(OUTPUT_DIR, 'all_problematic_shp')
            pathlib.Path(saving_dir_shp2).mkdir(parents=True, exist_ok=True)    # Creates the folder if doesn't exist
            out_path = os.path.join(saving_dir_shp2, f'{tile_name}_all_problematic.shp')
            bonus_shapefile_creation(all_problematic, out_path, vox_size)
        
        if shapefile_cfg['from_all_grey_zone']:
            all_grey_zone = df[df.criticality_tag=='grey_zone'].copy()
            saving_dir_shp3 = os.path.join(OUTPUT_DIR, 'all_grey_zone_shp')
            pathlib.Path(saving_dir_shp3).mkdir(parents=True, exist_ok=True)    # Creates the folder if doesn't exist
            out_path = os.path.join(saving_dir_shp3,f'{tile_name}_all_grey_zone.shp')
            bonus_shapefile_creation(all_grey_zone, out_path, vox_size)

    if SAVE_AS_CSV:
        saving_dir_csv = os.path.join(OUTPUT_DIR, 'csv')
        pathlib.Path(saving_dir_csv).mkdir(parents=True, exist_ok=True)         # Creates the folder if doesn't exist
        df.to_csv(os.path.join(saving_dir_csv, f'{tile_name}_change_detections.csv'), index=False)
    


if __name__ == '__main__':
    start_time = time.time()
    print('Starting visualisation process...')

    parser = argparse.ArgumentParser(description="This script saves the detections in a format allowing visualisation.")
    parser.add_argument('-cfg', type=str, help='a YAML config file', default="./config.yml")
    args = parser.parse_args()

    with open(args.cfg) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    
    WORKING_DIR = cfg['working_dir']
    DF_PATH = cfg['visualisation']['data']['df_path']
    OUTPUT_DIR = cfg['visualisation']['output']['dir']

    tile_name, voxel_size, _ = os.path.basename(DF_PATH).split('.')[0].rsplit('_',maxsplit=2)
    voxel_size = float(voxel_size)/100 

    df = pd.read_csv(DF_PATH)

    df = main(OUTPUT_DIR, df, cfg,tile_name, voxel_size)

    print(f'\nCreated visualisation file for tile {tile_name}, saved under {OUTPUT_DIR}')

    print(f'\nFinished entire process in: {round(time.time()-start_time, 2)} sec.')