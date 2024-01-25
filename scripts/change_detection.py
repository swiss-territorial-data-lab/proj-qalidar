import os

import util_las as las
import pandas as pd
import numpy as np
import yaml
import pathlib
import argparse
import time

from cd_submodules import submodule_voxelisation as voxelisation


start_time = time.time()

parser = argparse.ArgumentParser(description="This script creates the voxelisation of two point clouds on a common grid and returns it as a .csv files")
parser.add_argument('-cfg', type=str, help='a YAML config file', default="./config_test.yml")
args = parser.parse_args()


with open(args.cfg) as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)#[os.path.basename(__file__)]

WORKING_DIR = cfg['working_dir']
RUN_ON_FOLDER = cfg['mode']['multiple_files']
VOX_DIMENSION = cfg['vox_dimension']
DATA_DIR = cfg['data_dir']
CLASSES_CORRESPONDENCE_PATH = os.path.join(DATA_DIR, cfg['data']['classes_correspondence'])
PREV_FOLDER_DIR = os.path.join(DATA_DIR, cfg['data']['folder']['prev_folder'])
NEW_FOLDER_DIR = os.path.join(DATA_DIR, cfg['data']['folder']['new_folder'])
OUTPUT_DIR = os.path.join(DATA_DIR, cfg['output_dir'])

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

    matching_new_tiles = [new_tile for new_tile in new_tiles_list if prev_tile.split('.')[0] in new_tile]

    if len(matching_new_tiles)==0:
        raise SystemExit('Did not find matching new tile in folder. Make sure the tiles share the same name. (Note however that the file format can be .las or .laz)')
    prev_tile_path = os.path.join(PREV_FOLDER_DIR, prev_tile)
    new_tile_path = os.path.join(NEW_FOLDER_DIR, matching_new_tiles[0])
    
    voxelised_df = voxelisation.main(WORKING_DIR, prev_tile_path, new_tile_path, CLASSES_CORRESPONDENCE_PATH, VOX_DIMENSION)

    

    print(f'{tile_counter}/{total_nb_tiles}: Voxelised tile {prev_tile.split(".")[0]}.')

print(f'\nFinished entire voxelisation process in: {round(time.time()-start_time, 2)} sec.')