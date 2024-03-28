import argparse
import json
import os
import pathlib
import sys
import time
import yaml
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

from collections import Counter

def filter_small_cluster(cluster_attribution, min_cluster_size):
    # Count the occurrences of each value
    value_counts = Counter(cluster_attribution)

    # Values that occur fewer than 'min_cluster_size' times are set to -1
    new_cluster_attribution = np.array([
        value if value_counts[value] >= min_cluster_size 
        else -1 
        for value in cluster_attribution
    ])

    return new_cluster_attribution


def main(df, cfg, voxel_size):

    DBSCAN_HYPERPARAM = cfg['dbscan']['hyperparam']
    EPSILON = DBSCAN_HYPERPARAM['max_dist_factor']*voxel_size
    MIN_SAMPLES = DBSCAN_HYPERPARAM['min_samples']
    MIN_CLUSTER_SIZE = DBSCAN_HYPERPARAM['min_cluster_size']
    GREY_ZONE = DBSCAN_HYPERPARAM['consider_grey_zone']

    
    if GREY_ZONE:
        criticality_levels = ['problematic','grey_zone']
    else:
        criticality_levels = ['problematic']

    problematic_df = df[df.criticality_tag.isin(criticality_levels)]
    coordinates = problematic_df[['X_grid','Y_grid','Z_grid']]
    clustering = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES).fit(coordinates)

    final_clustering = filter_small_cluster(clustering.labels_, MIN_CLUSTER_SIZE)
    df['clusters'] = np.NaN

    # Unclustered voxels get the label 0, small clusters get the label 1, all other cluster get a label > 1
    df.loc[~df.criticality_tag.isin(criticality_levels), 'clusters'] = 0
    df.loc[problematic_df.index,'clusters'] = final_clustering + 2

    try:
        assert(df[df.clusters.isna()].empty), 'Not all voxels were assigned a cluster number.'
    except AssertionError as e:
        print(e)
        sys.exit(1)

    # One column for the majority label, another with the list of all labels in the cluster
    cluster_criticality_df = df.loc[df.clusters > 1].groupby('clusters').agg(
        cluster_criticality_number=('criticality_number', lambda x: x.mode()[0]),
        cluster_size=('criticality_number', lambda x: x.count())
    )

    df = df.merge(cluster_criticality_df, how='left', on='clusters')
    df['cluster_criticality_number'].fillna(0, inplace=True) 

    percentage_list = []
    for cluster in df.loc[df.clusters > 1, 'clusters'].unique():
        tmp_df = df[df.clusters==cluster].groupby('criticality_number').agg(
            criticality_count=('criticality_number', lambda x: x.count())
        )
        tmp_df['percentage']=round(tmp_df.criticality_count / df.loc[df.clusters==cluster, 'cluster_size'].iloc[0] * 100).astype('int')
        tmp_df.sort_values(by='percentage', ascending=False, inplace=True)

        distribution_str = ""
        for criticality_tuple in tmp_df.itertuples():
            distribution_str += f"#{int(criticality_tuple.Index)}: {criticality_tuple.percentage}% & "
        
        percentage_list.append(distribution_str.rstrip(' & '))

    df = df.merge(
        pd.DataFrame({'clusters': df.loc[df.clusters > 1, 'clusters'].unique(), 'cluster_distribution': percentage_list}), 
        how='left',
        on='clusters'
    )

    return df


if __name__ == '__main__':
    start_time = time.time()
    print('Starting DBSCAN clustering...')

    parser = argparse.ArgumentParser(description="This script clusters the detected changes")
    parser.add_argument('-cfg', type=str, help='a YAML config file', default="./config.yml")
    args = parser.parse_args()


    with open(args.cfg) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    
    WORKING_DIR = cfg['working_dir']
    DF_PATH = cfg['dbscan']['data']['criticality_df_path']
    OUTPUT_DIR = cfg['dbscan']['output']['dir']

    tile_name, voxel_size, _ = os.path.basename(DF_PATH).split('.')[0].rsplit('_',maxsplit=2)
    voxel_size = float(voxel_size)/100 

    df = pd.read_csv(DF_PATH)

    df = main(df, cfg, voxel_size)

    # Save the new dataframe as csv
    saving_time = time.strftime("%d%m-%H%M")

    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, f'{tile_name}_{int(voxel_size*100)}_cluster-{saving_time}.csv'), index = False)
    hyperparam_dict = cfg['dbscan']['hyperparam']

    json.dumps(hyperparam_dict)

    with open(os.path.join(OUTPUT_DIR, f"{tile_name}_{str(int(voxel_size*100))}_cluster-{saving_time}.json"), "w") as outfile: 
        json.dump(hyperparam_dict, outfile)

    print(f'Clustered file for tile {tile_name} saved under {OUTPUT_DIR}')

    print(f'Finished entire DBSCAN clustering process in: {round(time.time()-start_time, 2)} sec.')

