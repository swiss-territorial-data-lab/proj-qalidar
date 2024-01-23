import os

import util_las as las
import pandas as pd
import numpy as np
import yaml
import pathlib
import argparse
import time

def align_columns(df1, df2):
    # Modify the dataframes if one column is missing compared to the other. If it is the case it adds an empty column
    
    # If debugging, uncomment 
    # df1 = df1.copy(deep=True) # Do the modification on a copy of the dataframe
    # df2 = df2.copy(deep=True)

    missing_columns_df1 = set(df2.columns) - set(df1.columns)

    for column in missing_columns_df1:
        df1[column] = pd.Series(dtype=df2[column].dtype)

    missing_columns_df2 = set(df1.columns) - set(df2.columns)

    for column in missing_columns_df2:
        df2[column] = pd.Series(dtype=df1[column].dtype)

    # Make sure that the order of the classification columns is sorted
    sorted_class_columns1 = df1.iloc[:,3:].reindex(sorted(df1.iloc[:,3:].columns), axis=1)
    df1.drop(df1.columns[3:], axis=1, inplace=True)
    df1 = pd.concat([df1, sorted_class_columns1],axis=1)

    sorted_class_columns2 = df2.iloc[:,3:].reindex(sorted(df2.iloc[:,3:].columns), axis=1)
    df2.drop(df2.columns[3:], axis=1, inplace=True)
    df2 = pd.concat([df2, sorted_class_columns2],axis=1)

    return df1, df2

start_time = time.time()

WORKING_DIR = '/mnt/data-01/nmunger/proj-qalidar/data' # So that the default value for argparse work when launching from vscode
os.chdir(WORKING_DIR)

parser = argparse.ArgumentParser(description="This script creates the voxelisation of two point clouds on a common grid and returns it as a .csv files")
parser.add_argument('-cfg', type=str, help='a YAML config file', default="../config.yml")
args = parser.parse_args()


with open(args.cfg) as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

WORKING_DIR = cfg['working_dir']
RUN_ON_FOLDER = cfg['mode']['multiple_files']
VOX_DIMENSION = cfg['vox_dimension']
CLASSES_CORRESPONDENCE_FILE = cfg['classes_correspondence']
PREV_FOLDER_DIR = os.path.join(WORKING_DIR, cfg['data']['folder']['prev_folder'])
NEW_FOLDER_DIR = os.path.join(WORKING_DIR, cfg['data']['folder']['new_folder'])
OUTPUT_DIR = os.path.join(WORKING_DIR, cfg['output_dir'])

os.chdir(WORKING_DIR)

if RUN_ON_FOLDER == True:
    print(f'Starting voxelisation process for tiles located in folder: {PREV_FOLDER_DIR}\n')
    prev_tiles_list = os.listdir(PREV_FOLDER_DIR) 
else: # Run on a single tile
    PREV_TILE_NAME = cfg['data']['single_tile']['prev_tile_name']
    print(f'Starting voxelisation process for tile: {PREV_TILE_NAME}\n')
    prev_tiles_list = [PREV_TILE_NAME]

new_tiles_list = os.listdir(NEW_FOLDER_DIR)

tile_counter = 1 

total_nb_tiles = len(prev_tiles_list)

for prev_tile in prev_tiles_list:

    prev_pc_df = las.las_to_df_xyzclass(os.path.join(PREV_FOLDER_DIR, prev_tile))

    matching_new_tiles = [new_tile for new_tile in new_tiles_list if prev_tile.split('.')[0] in new_tile]

    if len(matching_new_tiles)==0:
        raise SystemExit('Did not find matching new tile in folder. Make sure the tiles share the same name. (Note however that the file format can be .las or .laz)')
    
    new_pc_df = las.las_to_df_xyzclass(os.path.join(NEW_FOLDER_DIR, matching_new_tiles[0]))

    # Remove all points which are noise in the previous generation as they do not bring useful information
    prev_pc_df = prev_pc_df[prev_pc_df['classification']!=7]

    # Match the supplementary class to classes from the previous generation
    new_pc_df = las.reclassify(new_pc_df, os.path.join(WORKING_DIR, CLASSES_CORRESPONDENCE_FILE)) 

    # Set the lowest coordinates of the point clouds in each axis as the origin of the common grid 
    x_origin = min(prev_pc_df.X.min(), new_pc_df.X.min())
    y_origin = min(prev_pc_df.Y.min(), new_pc_df.Y.min())
    z_origin = min(prev_pc_df.Z.min(), new_pc_df.Z.min())
    # Same logic for the highest coordinates
    x_max = max(prev_pc_df.X.max(), new_pc_df.X.max())
    y_max = max(prev_pc_df.Y.max(), new_pc_df.Y.max())
    z_max = max(prev_pc_df.Z.max(), new_pc_df.Z.max())

    grid_origin = x_origin, y_origin, z_origin

    grid_max = x_max, y_max, z_max

    prev_voxelised_df = las.to_voxelised_df(prev_pc_df, grid_origin, grid_max, VOX_DIMENSION, VOX_DIMENSION)
    new_voxelised_df = las.to_voxelised_df(new_pc_df, grid_origin, grid_max, VOX_DIMENSION, VOX_DIMENSION)

    # If one class is missing in either of the dataframe compared to the other, create new empty column
    prev_voxelised_df, new_voxelised_df = align_columns(prev_voxelised_df, new_voxelised_df)

    # Free up space
    del prev_pc_df
    del new_pc_df

    merged_df = prev_voxelised_df.merge(new_voxelised_df, on=['X_grid','Y_grid','Z_grid'], how='outer', suffixes=('_prev','_new'))

    merged_df = merged_df.replace(np.NaN, 0)
    
    merged_df['vox_id'] = merged_df.index # Define a fixed id for each voxel

    # Create the path for the folder to store the .csv file in case it doesn't yet exist
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # In file name, set voxel size in centimeters, so as to avoid decimal (.) presence in the file name
    save_path = os.path.join(OUTPUT_DIR, f'{prev_tile.split(".")[0]}_{int(VOX_DIMENSION*100)}-{int(VOX_DIMENSION*100)}'+'.csv')
    
    merged_df.to_csv(save_path, index=False)

    print(f'{tile_counter}/{total_nb_tiles}: Voxelised file for tile {prev_tile.split(".")[0]} saved under {save_path}')

print(f'\nFinished entire voxelisation process in: {round(time.time()-start_time, 2)} sec.')