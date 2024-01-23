import os
import yaml
import pandas as pd
from sklearn.cluster import DBSCAN
import util_las as las
import numpy as np
import argparse
import pathlib

WORKING_DIR = '/mnt/data-01/nmunger/proj-qalidar/data' # So that the default value for argparse work when launching from vscode
os.chdir(WORKING_DIR)

parser = argparse.ArgumentParser(description="This script creates the voxelisation of two point clouds on a common grid and returns it as a .csv files")
parser.add_argument('-cfg', type=str, help='a YAML config file', default="../config.yml")
args = parser.parse_args()

with open(args.cfg) as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

WORKING_DIR = cfg['working_dir']
DF_PATH = cfg['data']['criticity_df_path']
OUTPUT_DIR = cfg['output_dir']
os.chdir(WORKING_DIR)

tile_name, voxel_dimension, _ = os.path.basename(DF_PATH).split('.')[0].rsplit('_',maxsplit=2)
voxel_dimension = float(voxel_dimension)/100 

EPSILON = cfg['hyperparam']['max_dist_factor']*voxel_dimension
MIN_SAMPLES = cfg['hyperparam']['min_nb_voxels']
GREY_ZONE = cfg['hyperparam']['consider_grey_zone']

df = pd.read_csv(DF_PATH)

if GREY_ZONE:
    criticity_levels = ['problematic','grey_zone']
else:
    criticity_levels = ['problematic']

problematic_df = df[df.change_criticity.isin(criticity_levels)]
X = problematic_df[['X_grid','Y_grid','Z_grid']]
clustering = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit(X)

df['clusters'] = np.NaN

df.loc[problematic_df.index,'clusters'] = clustering.labels_+2 # Add two, so that isolated become = 1, all other cluster >1

# The rest of the voxels get the label 0
df.loc[~df.change_criticity.isin(criticity_levels), 'clusters'] = 0

las_file = las.df_to_las(df, user_data_col='clusters')

pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# NOTE that the UserData field can only store up to 1 byte, i.e. max 256 values
las_file.write(os.path.join(OUTPUT_DIR, f'change_detection_{tile_name}_{int(voxel_dimension*100)}_{int(EPSILON*100)}_{MIN_SAMPLES}_{"_".join(criticity_levels)}.las'))






