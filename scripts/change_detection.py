import os
import sys

import yaml
import pathlib
import argparse
import json
import time

from util_misc import verify_out_folder
import substeps.voxelisation as voxelisation
import substeps.decision_tree as decision_tree
import substeps.dbscan as dbscan
import substeps.visualisation as visualisation
from constant import BColors

parser = argparse.ArgumentParser(description="This script applies the change detection workflow with the configuration as defined in the yaml file.")
parser.add_argument('-cfg', type=str, help='a YAML config file', default="./config_debug_florian.yml")
args = parser.parse_args()


with open(args.cfg) as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)

WORKING_DIR = cfg['working_dir']
RUN_ON_FOLDER = cfg['mode']['multiple_files']
VOX_DIMENSION = cfg['voxelisation']['vox_dimension']
DATA_DIR = cfg['data_dir']
CLASSES_CORRESPONDENCE_PATH = os.path.join(DATA_DIR, cfg['data']['classes_correspondence'])
PREV_FOLDER_DIR = os.path.join(DATA_DIR, cfg['data']['folder']['prev_folder'])
NEW_FOLDER_DIR = os.path.join(DATA_DIR, cfg['data']['folder']['new_folder'])
OUTPUT_DIR = os.path.join(DATA_DIR, cfg['output_dir'])
PREV_TILE_NAME = cfg['data']['single_tile']['prev_tile_name']
DEBUG = cfg['debug']


os.chdir(WORKING_DIR)

if DEBUG == True:
    saving_time = time.strftime("%d%m-%H%M")
    # Create folder which will store all of the visualisation output
    if RUN_ON_FOLDER == True:
        saving_dir = os.path.join(OUTPUT_DIR, f'debug_vis_{saving_time}')
    else: 
        saving_dir = os.path.join(OUTPUT_DIR, f'{PREV_TILE_NAME.split(".")[0]}_saved_at-{saving_time}')
    pathlib.Path(saving_dir).mkdir(parents=True, exist_ok=True) 

else:
    if verify_out_folder(OUTPUT_DIR):
        saving_dir = OUTPUT_DIR
        pass
    else:
        sys.exit()

start_time = time.time()

if RUN_ON_FOLDER == True:
    print(f'Starting change detection process for tiles located in folder: {PREV_FOLDER_DIR}\n')
    prev_tiles_list = os.listdir(PREV_FOLDER_DIR)
else: # Run on a single tile
    print(f'Starting change detection process for tile: {PREV_TILE_NAME}\n')
    prev_tiles_list = [PREV_TILE_NAME]

new_tiles_list = os.listdir(NEW_FOLDER_DIR)

tile_counter = 1 

total_nb_tiles = len(prev_tiles_list)

non_processed_tiles = []

for prev_tile in prev_tiles_list:

    tile_name = prev_tile.split(".")[0]

    matching_new_tiles = [new_tile for new_tile in new_tiles_list if prev_tile.split('.')[0] in new_tile]

    if len(matching_new_tiles)==0:
        non_processed_tiles.append(prev_tile)
        print(BColors.WARNING + 
              f'Did not find matching new tile in folder for {prev_tile}. Make sure the tiles share the same name. (Note however that the file format can be .las or .laz)\nSkipping to next tile.'
              + BColors.ENDC)
        continue
    prev_tile_path = os.path.join(PREV_FOLDER_DIR, prev_tile)
    new_tile_path = os.path.join(NEW_FOLDER_DIR, matching_new_tiles[0])

    tic = time.time()    
    voxelised_df = voxelisation.main(WORKING_DIR, prev_tile_path, new_tile_path, CLASSES_CORRESPONDENCE_PATH, VOX_DIMENSION)

    print(f'{tile_counter}/{total_nb_tiles}: Voxelised tile {tile_name}. ({round(time.time()-tic, 2)} sec)')

    tic = time.time()    
    criticality_df = decision_tree.main(voxelised_df, cfg, VOX_DIMENSION)
    
    print(f'{tile_counter}/{total_nb_tiles}: Ran decision tree on tile {tile_name}. ({round(time.time()-tic, 2)} sec)')
    
    tic = time.time()    
    clustered_df = dbscan.main(criticality_df, cfg, VOX_DIMENSION)

    print(f'{tile_counter}/{total_nb_tiles}: Ran DBSCAN clustering on tile {tile_name}. ({round(time.time()-tic, 2)} sec)')

    tic = time.time()
    visualisation.main(saving_dir, clustered_df, cfg, tile_name, VOX_DIMENSION)

    print(f'{tile_counter}/{total_nb_tiles}: Saved visualisation files for tile {tile_name}. ({round(time.time()-tic, 2)} sec)')
    

    tile_counter += 1


with open(os.path.join(saving_dir, 'config.json'), "w") as outfile: 
    json.dump(cfg, outfile)

print(f'\nFinished entire change detection process in: {round(time.time()-start_time, 2)} sec.')
print(f'Results saved under {saving_dir}')

if non_processed_tiles: #If the list is non empty
    print(BColors.WARNING + f'List of non processed tiles due to a lack of new tile match: {non_processed_tiles}' + BColors.ENDC)