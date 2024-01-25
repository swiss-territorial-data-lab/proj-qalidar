import os
import sys
import argparse
import yaml
import pathlib
import pandas as pd
import numpy as np
import time

sys.path.insert(0,'..')
from scripts import util_las as las
from scripts import constant as cst

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

# --------------------------------------------------------------------------------------------

def main(WORKING_DIR, PREV_TILE_PATH, NEW_TILE_PATH, CLASSES_CORRESPONDENCE_PATH, vox_dimension):
    """Performs the process of voxelising two point clouds on a common grid
    Args:
        WORKING_DIR (path): working directory
        PREV_TILE_DIR (path): location for the previous point cloud (in .las or .laz format)
        NEW_TILE_DIR (path): location for the new point cloud (in .las or .laz format)
        CLASSES_CORRESPONDENCE_PATH (path): location for the .csv file containing the match between IDs
        vox_dimension (float): the desired size in meters of voxels
    
    Returns:
        merged_df (pd.DataFrame): the voxelised comparison in a DataFrame
    """

    os.chdir(WORKING_DIR)

    prev_pc_df = las.las_to_df_xyzclass(PREV_TILE_PATH)

    new_pc_df = las.las_to_df_xyzclass(NEW_TILE_PATH)

    # Remove all points which are noise in the previous generation as they do not bring useful information
    prev_pc_df = prev_pc_df[prev_pc_df['classification']!=cst.NOISE]

    # Match the supplementary class to classes from the previous generation
    new_pc_df = las.reclassify(new_pc_df, CLASSES_CORRESPONDENCE_PATH) 

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

    prev_voxelised_df = las.to_voxelised_df(prev_pc_df, grid_origin, grid_max, vox_dimension, vox_dimension)
    new_voxelised_df = las.to_voxelised_df(new_pc_df, grid_origin, grid_max, vox_dimension, vox_dimension)

    # If one class is missing in either of the dataframe compared to the other, create new empty column
    prev_voxelised_df, new_voxelised_df = align_columns(prev_voxelised_df, new_voxelised_df)

    # Free up space
    del prev_pc_df
    del new_pc_df

    merged_df = prev_voxelised_df.merge(new_voxelised_df, on=['X_grid','Y_grid','Z_grid'], how='outer', suffixes=('_prev','_new'))

    merged_df = merged_df.replace(np.NaN, 0)

    merged_df['vox_id'] = merged_df.index # Define a fixed id for each voxel

    return merged_df


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description="This script creates the voxelisation of two point clouds on a common grid and returns it as a .csv files")
    parser.add_argument('-cfg', type=str, help='a YAML config file', default="../config_test.yml")
    args = parser.parse_args()


    with open(args.cfg) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    WORKING_DIR = cfg['working_dir']
    VOX_DIMENSION = cfg['vox_dimension']
    PREV_TILE_PATH = cfg['voxelisation.py']['data']['prev_tile_path']
    NEW_TILE_PATH = cfg['voxelisation.py']['data']['new_tile_path']
    DATA_DIR = cfg['data_dir']
    CLASSES_CORRESPONDENCE_PATH = os.path.join(DATA_DIR, cfg['data']['classes_correspondence'])
    OUTPUT_DIR = cfg['voxelisation.py']['output_dir']

    tile_name = os.path.basename(PREV_TILE_PATH).split('.')[0]

    voxelised_df = main(WORKING_DIR, PREV_TILE_PATH, NEW_TILE_PATH, CLASSES_CORRESPONDENCE_PATH, VOX_DIMENSION)
    # # Create the path for the folder to store the .csv file in case it doesn't yet exist
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # # In file name, set voxel size in centimeters, so as to avoid decimal (.) presence in the file name
    save_path = os.path.join(OUTPUT_DIR, f'{tile_name}_test_{int(VOX_DIMENSION*100)}-{int(VOX_DIMENSION*100)}'+'.csv')
    
    voxelised_df.to_csv(save_path, index=False)

    print(f'Voxelised file for tile {tile_name} saved under {save_path}')

    print(f'\nFinished entire voxelisation process in: {round(time.time()-start_time, 2)} sec.')
    