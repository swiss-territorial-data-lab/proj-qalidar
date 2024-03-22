'''
Script retiling the older generation point cloud tile from swisstopo to match the size of the new generation.
It assumes that the name for the new tiles consist of the complete coordinates in LV95 format, 
separated by an underscore, for example: 2527000_1190500.laz 
The dimension of the tile must be properly set.
'''
import argparse
import os
import sys
import time
import yaml
from tqdm import tqdm

start_time = time.time()
print('Starting...')

parser = argparse.ArgumentParser(description="This script subdivides tiles to a new size.")
parser.add_argument('-cfg', type=str, help='a YAML config file', default="./config.yml")
args = parser.parse_args()


with open(args.cfg) as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)['preprocessing']

LASTOOLS_PATH = cfg['lastools_path']

NEW_DATA_FOLDER = cfg['new_data_folder']
OLD_DATA_FOLDER = cfg['old_data_folder']
DESTINATION_FOLDER = cfg['destination_folder']

NEW_DATA_FORMAT = cfg['new_data_format']
OLD_DATA_FORMAT = cfg['old_data_format']
DESTINATION_FORMAT = cfg['destination_format']
TILE_DIMENSION = cfg['tile_dimension']

os.makedirs(DESTINATION_FOLDER, exist_ok=True)

for filename in tqdm(os.listdir(NEW_DATA_FOLDER), desc='Clip files'):
    if filename.endswith('.copc.las') or filename.endswith('.copc.laz'):
        continue
    
    # Extract the coordinates from the filename
    x_origin, y_origin = filename.replace(f'{NEW_DATA_FORMAT}','').split('_') 
    input_file = os.path.join(OLD_DATA_FOLDER, x_origin[:4] + "_" + y_origin[:4] + OLD_DATA_FORMAT)
    output_file = os.path.join(DESTINATION_FOLDER, x_origin + "_" + y_origin + DESTINATION_FORMAT)

    # Find the tiles with the same coordinates and reshape it with the proper tiling
    lastool_request = f'{os.path.join(LASTOOLS_PATH , "las2las.exe")} \
         -i {input_file} \
         -keep_tile {x_origin} {y_origin} {TILE_DIMENSION} \
         -o {output_file}'
    lastool_request = 'wine ' + lastool_request if 'win' not in sys.platform else lastool_request

    os.system(lastool_request)

print(f'Done! The files were written in {DESTINATION_FOLDER}.')

# Stop chronometer
stop_time = time.time()
print(f"Nothing left to be done: exiting. Elapsed time: {(stop_time-start_time):.2f} seconds")

sys.stderr.flush()