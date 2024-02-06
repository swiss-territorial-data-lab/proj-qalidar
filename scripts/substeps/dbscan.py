import os
import yaml
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import argparse
import pathlib
import time
import json


def main(df, cfg, voxel_dimension):

    EPSILON = cfg['dbscan']['hyperparam']['max_dist_factor']*voxel_dimension
    MIN_SAMPLES = cfg['dbscan']['hyperparam']['min_nb_voxels']
    GREY_ZONE = cfg['dbscan']['hyperparam']['consider_grey_zone']

    
    if GREY_ZONE:
        criticality_levels = ['problematic','grey_zone']
    else:
        criticality_levels = ['problematic']

    problematic_df = df[df.criticality_tag.isin(criticality_levels)]
    X = problematic_df[['X_grid','Y_grid','Z_grid']]
    clustering = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit(X)

    df['clusters'] = np.NaN

    df.loc[problematic_df.index,'clusters'] = clustering.labels_+2 # Add two, so that isolated become = 1, all other cluster >1

    # The rest of the voxels get the label 0
    df.loc[~df.criticality_tag.isin(criticality_levels), 'clusters'] = 0

    # pd.Series.mode returns two values if there is a tie. We only want one value
    cluster_major_criticality_df = df.loc[df.clusters > 1].groupby('clusters').agg(cluster_criticality_number=('criticality_number', lambda x: x.mode()[0]))

    df = df.merge(cluster_major_criticality_df,how='left', on='clusters')

    df['cluster_criticality_number'].fillna(0, inplace=True) 

    return df


if __name__ == '__main__':
    start_time = time.time()
    print('Starting DBSCAN clustering...')

    parser = argparse.ArgumentParser(description="This script clusters the detected changes")
    parser.add_argument('-cfg', type=str, help='a YAML config file', default="./config_test.yml")
    args = parser.parse_args()


    with open(args.cfg) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    
    WORKING_DIR = cfg['working_dir']
    DF_PATH = cfg['dbscan']['data']['criticality_df_path']
    OUTPUT_DIR = cfg['dbscan']['output']['dir']

    tile_name, voxel_dimension, _ = os.path.basename(DF_PATH).split('.')[0].rsplit('_',maxsplit=2)
    voxel_dimension = float(voxel_dimension)/100 

    df = pd.read_csv(DF_PATH)

    df = main(df, cfg, voxel_dimension)

    # Save the new dataframe as csv
    saving_time = time.strftime("%d%m-%H%M")

    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, f'{tile_name}_{int(voxel_dimension*100)}_cluster-{saving_time}.csv'), index = False)
    hyperparam_dict = cfg['dbscan']['hyperparam']

    json.dumps(hyperparam_dict)

    with open(os.path.join(OUTPUT_DIR, f"{tile_name}_{str(int(voxel_dimension*100))}_cluster-{saving_time}.json"), "w") as outfile: 
        json.dump(hyperparam_dict, outfile)

    print(f'\nClustered file for tile {tile_name} saved under {OUTPUT_DIR}')

    print(f'\nFinished entire DBSCAN clustering process in: {round(time.time()-start_time, 2)} sec.')

