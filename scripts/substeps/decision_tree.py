import os
import sys
import argparse
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import open3d as o3d
import laspy
import time
import pathlib
import yaml
from sklearn.neighbors import KDTree

sys.path.append(".") 
from scripts.util_misc import cosine_similarity
import scripts.constant as cst

# Utilitary functions used in the criticity tree
def find_neighbours_occupancy(x, columns_to_compare):
    # Given a voxel to evaluate, return the combined occupancy of all its neighbours
    return np.any(columns_to_compare[x].astype(bool),axis=0)

def compare_to_neighbours(df, tree, kd_tree_query_radius, case='TBD'):
    '''The cases can be: 
    -"TBD", we compare the new vox. occupancy with the new neighbours 
    -"apparition", we compare the new vox. occupancy with the previous neighbours occupancies
    -"disparition", we compare the prev. vox. occupancy with the new neighbours occupancies'''
    df = df.copy() 

    if case=='TBD':
        to_evaluate, to_compare = 'new', 'new'
    elif case == 'apparition':
        to_evaluate, to_compare = 'new', 'prev'
    elif case == 'disparition':
        to_evaluate, to_compare = 'prev', 'new'
    else:
        raise ValueError

    voxels_to_evaluate_df = df.loc[df.change_criticity == case]

    # Query all ids of neighbours to the location to evaluate. This also returns the id of the voxel itself which must be removed
    all_neighbours_ids = tree.query_radius(voxels_to_evaluate_df.loc[:, ['X_grid','Y_grid','Z_grid']].to_numpy(),  kd_tree_query_radius)

    # Remove the id of the voxel itself in each neighbour sets
    list_neighbours = []
    for i in range(len(all_neighbours_ids)):
        valid_neighbours_ids = all_neighbours_ids[i][all_neighbours_ids[i] != voxels_to_evaluate_df.index[i]]
        list_neighbours.append(valid_neighbours_ids)
    
    # Select 'new' or 'prev' columns depending on the case
    columns_to_compare = df.loc[:,df.columns.str.endswith(to_compare)].to_numpy()
    
    neighbours_occupancy = np.asarray([find_neighbours_occupancy(sub_array, columns_to_compare) for sub_array in list_neighbours])

    voxels_to_evaluate_bool = voxels_to_evaluate_df.loc[:, df.columns.str.endswith(to_evaluate)].to_numpy().astype(bool)
    
    # For each voxel to evaluate, check if the class present in it are also present in the neighbours
    presence_in_neighbours = np.all(np.equal(voxels_to_evaluate_bool, (neighbours_occupancy & voxels_to_evaluate_bool)), axis=1)
    
    if case == 'disparition':
        df.loc[df.change_criticity==case, 'change_criticity'] = np.where(presence_in_neighbours==True, 'non_prob-4', 'problematic-9')
    elif case == 'apparition':
        df.loc[df.change_criticity==case, 'change_criticity'] = np.where(presence_in_neighbours==True,'non_prob-5','problematic-10')
    elif case == 'TBD':
        df.loc[df.change_criticity==case, 'change_criticity'] = np.where(presence_in_neighbours==True, 'grey_zone-8', 'problematic-11')
    
    return df

def non_prob_apparition(df, class_name):
    if class_name == 'building':
        class_id = cst.BUILDING
    elif class_name == 'vegetation': 
        class_id = cst.VEGETATION
    else:
        print('Unvalid class name.')

    # Find for each planar grid cell the altitude of the highest point of the class we're evaluating
    highest_voxel_df = df[df[f'{class_id}_new']>0].groupby(['X_grid','Y_grid'], observed=False)['Z_grid'].max()\
                            .to_frame(f'top_{class_name}_voxel').reset_index()
    
    highest_voxel_df = highest_voxel_df.merge(df,\
                            left_on=['X_grid','Y_grid',f'top_{class_name}_voxel'], right_on=['X_grid','Y_grid','Z_grid'],how='left') \
                            [['X_grid','Y_grid',f'top_{class_name}_voxel','change_criticity']]\
                            .rename(columns={'change_criticity':'highest_change'})
    
    # For all voxel which have a problematic apparition of class building, match with the altitude of highest building point
    # in their planar grid cell
    temporary_df = df[(df.change_criticity=='problematic-10') & (df.majority_class==f'{class_id}_new')]\
                .merge(highest_voxel_df, how='left', on=['X_grid','Y_grid'])
    
    # Get the voxel's IDs for which the highest building voxel in their column is not problematic
    non_prob_apparition_idx = temporary_df.loc[temporary_df['highest_change'].str.contains('non_prob'), 'vox_id'].to_numpy()

    df.loc[df['vox_id'].isin(non_prob_apparition_idx), 'change_criticity'] = 'non_prob-6'

    return df 

# -------------------------------------------------------------------------------------------------------------


def main(df, cfg, vox_dimension):
    """Performs the assignement of each voxel to a certain criticity level
    Args:
        df (pd.DataFrame): voxelised comparison created from submodule_voxelisation
        COS_THRESHOLD (float): threshold [0,1] for decision C
        SECOND_COS_THRESHOLD (float): threshold [0,1] for decision D
        THIRD_COS_THRESHOLD (float): threshold [0,1] for decision E
        THRESHOLD_CLASS_1_PRESENCE (float): threshold for decision F
        KD_TREE_QUERY_RADIUS (float): threshold for decision H and I
    
    Returns:
        df (pd.DataFrame): the updated DataFrame with the criticity information added
    """

    COS_THRESHOLD = cfg['criticity_tree']['threshold']['first_cos_threshold']
    SECOND_COS_THRESHOLD = cfg['criticity_tree']['threshold']['second_cos_threshold']
    THIRD_COS_THRESHOLD = cfg['criticity_tree']['threshold']['third_cos_threshold']
    THRESHOLD_CLASS_1_PRESENCE = cfg['criticity_tree']['threshold']['threshold_class_1_presence']
    KD_TREE_QUERY_RADIUS = cfg['criticity_tree']['threshold']['kd_tree_search_factor']*vox_dimension

    df['change_criticity'] = 'TBD' # Set all change criticities to TBD = To be determined

    # ---------------------------------------------------------------------------------------
    # Decision A: Is there only one class in both generation, and is it the same?
    # ---------------------------------------------------------------------------------------
    voxels_to_evaluate = df[df['change_criticity']=='TBD'].copy()
    voxels_to_evaluate_prev = voxels_to_evaluate.iloc[:, voxels_to_evaluate.columns.str.endswith('_prev')].to_numpy().astype(bool)
    voxels_to_evaluate_new = voxels_to_evaluate.iloc[:, voxels_to_evaluate.columns.str.endswith('_new')].to_numpy().astype(bool)

    mask = (voxels_to_evaluate_prev.sum(axis=1)==1) & (np.all(voxels_to_evaluate_prev==voxels_to_evaluate_new, axis=1))

    # Set criticity to 'non_prob_1' for rows for which the mask is True
    df.loc[mask, 'change_criticity'] = 'non_prob-1'

    # ---------------------------------------------------------------------------------------
    # Decision B: 'Is there noise in the new voxel?'
    # ---------------------------------------------------------------------------------------
    if '7_new' in df.columns: # Only execute if there is some noise in the new pointcloud   
        df.loc[df['7_new']>0,'change_criticity'] = 'problematic-13'

    # ---------------------------------------------------------------------------------------
    # ### Decision C: Does the number of class and distribution stay the same? 
    # ---------------------------------------------------------------------------------------
    df['cosine_similarity'] = np.where(df['change_criticity']=='TBD', 0, 1.0) # Set cosine similarity to 1 for all already determined voxels

    voxels_to_evaluate =  df[df['change_criticity']=='TBD'].copy()

    cosine_similarity_array = cosine_similarity(voxels_to_evaluate)
    df.loc[df['change_criticity']=='TBD', 'cosine_similarity'] = cosine_similarity_array

    # Mask where True if the boolean presence of the classes are exactly the same in both generation
    same_class_present = np.all(df.iloc[:, df.columns.str.endswith('_prev')].to_numpy().astype(bool) == df.iloc[:, df.columns.str.endswith('_new')].to_numpy().astype(bool), axis=1)

    df.loc[(df['cosine_similarity']>COS_THRESHOLD) & (df['change_criticity']=='TBD') & (same_class_present), 'change_criticity'] = 'non_prob-2'

    # ---------------------------------------------------------------------------------------
    # Decision D: Do the previous classes keep the same distribution?
    # ---------------------------------------------------------------------------------------
    # Computing the cosine similarity only between classes which are present in the previous generation 
    # Note: if only one class is present in the previous generation, the cosine similarity is either 1 or -1 (unvalid division) 
    # which doesn't provide much info, possibly compare euclidean distance between the normalised density
    voxels_to_evaluate = df[df['change_criticity']=='TBD']

    df['second_cosine_similarity'] = np.nan

    voxels_to_evaluate_prev = voxels_to_evaluate.iloc[:, voxels_to_evaluate.columns.str.endswith('_prev')].to_numpy()
    voxels_to_evaluate_new = voxels_to_evaluate.iloc[:, voxels_to_evaluate.columns.str.endswith('_new')].to_numpy()
    dot_product = np.sum(voxels_to_evaluate_prev * voxels_to_evaluate_new, axis=1)
    # For new vector, only take to_numpy() for which class is present in the previous vector
    product_of_norm = np.linalg.norm(voxels_to_evaluate_prev, axis=1)*np.linalg.norm(voxels_to_evaluate_prev.astype(bool) * voxels_to_evaluate_new, axis=1) 

    # For cases where one vector is completely empty, avoid division by zero and replace by -1
    cosine_similarity_array = np.divide(dot_product, product_of_norm, out = np.full_like(dot_product, -1), where = product_of_norm!=0)

    df.loc[voxels_to_evaluate.index, 'second_cosine_similarity'] = cosine_similarity_array

    # Added condition of 'df.cosine_similarity!=-1' as this represent cases of complete disparition in the voxel which we want to keep for decision G
    df.loc[(df.second_cosine_similarity<SECOND_COS_THRESHOLD) & (df.cosine_similarity!=-1), 'change_criticity']='problematic-12'

    # ---------------------------------------------------------------------------------------
    # Decision E: is the change due to class 1?
    # ---------------------------------------------------------------------------------------
    # We want to compare whether the voxels are similar if we don't consider the unclassified points. If they stay the same, 
    # it means the difference comes from unclassified point.  
    voxels_to_evaluate = df[df['change_criticity']=='TBD'].drop(columns=['1_prev','1_new'])

    # For the specific cases of apparition or disparition only due to class 1, find rows which are empty for prev. and new gen. when not
    # considering the class 1
    mask_disparition_apparition = (voxels_to_evaluate.iloc[:, voxels_to_evaluate.columns.str.contains('_prev|_new')].to_numpy()).sum(axis=1)==0
    df.loc[voxels_to_evaluate[mask_disparition_apparition].index, 'change_criticity'] = 'class_1_specific'

    voxels_to_evaluate = df[df['change_criticity']=='TBD'].drop(columns=['1_prev','1_new'])
    cosine_similarity_array = cosine_similarity(voxels_to_evaluate)

    df.loc[voxels_to_evaluate.index, 'third_cosine_similarity'] = cosine_similarity_array

    # We want to find the voxels which have changed **because** of class 1. We assume those are the ones for which the cosine similarity was low when considering all 
    # the class but is actually high if we don't consider the class 1. <br> (Note that the condition on the first cosine threshold is necessary since in condition C we ask 
    # wheter the distribution stays the same **and** that the class don't change. This keeps a lot of voxels which have a very high cosine similarity but which do not have exactly the same class.)

    df.loc[(df.change_criticity=='TBD') \
            & (df['third_cosine_similarity']>THIRD_COS_THRESHOLD) \
            & (df['cosine_similarity']<COS_THRESHOLD), 'change_criticity'] = 'class_1_specific'

    # ---------------------------------------------------------------------------------------
    # Decision F: Does the class 1 have a low presence in the new voxel?
    # ---------------------------------------------------------------------------------------
    nb_points_prev = np.sum(df.iloc[:,df.columns.str.endswith('_prev')].to_numpy())
    nb_points_new = np.sum(df.iloc[:,df.columns.str.endswith('_new')].to_numpy())
    normalising_factor = nb_points_prev/nb_points_new
    class_1_new_normalised = df.loc[df.change_criticity == 'class_1_specific', '1_new']*normalising_factor


    df.loc[class_1_new_normalised.index, 'change_criticity'] = np.where(class_1_new_normalised<THRESHOLD_CLASS_1_PRESENCE,'non_prob-3', 'grey_zone-7')

    # ---------------------------------------------------------------------------------------
    # Decision G: Is the change from (empty -> class x) | (class x -> empty)
    # ---------------------------------------------------------------------------------------
    df.loc[(df['change_criticity']=='TBD') & (df['cosine_similarity']==-1) & (df.iloc[:,df.columns.str.endswith('_prev')].sum(axis=1).astype(bool)), 'change_criticity'] = 'disparition'

    df.loc[(df['change_criticity']=='TBD')& (df['cosine_similarity']==-1) & (df.iloc[:,df.columns.str.endswith('_new')].sum(axis=1).astype(bool)), 'change_criticity'] = 'apparition'

    # ---------------------------------------------------------------------------------------
    # Decision H: do the neighbouring voxels contain also the new class (case of apparition/disparition)?
    # ---------------------------------------------------------------------------------------
    # Generate KDTree for efficient neighbours research
    tree = KDTree(df[['X_grid','Y_grid','Z_grid']].to_numpy())

    df = compare_to_neighbours(df, tree, KD_TREE_QUERY_RADIUS, case='apparition')

    df = compare_to_neighbours(df, tree, KD_TREE_QUERY_RADIUS, case='disparition')

    # ---------------------------------------------------------------------------------------
    # Decision I: do the neighbouring voxels also contain the new class (case of change of distribution)?
    # ---------------------------------------------------------------------------------------
    df = compare_to_neighbours(df, tree, KD_TREE_QUERY_RADIUS, case='TBD')

    # ---------------------------------------------------------------------------------------
    # ### Decision J: Additional check up for class 3 (vegetation) and 6 (building). If apparition, check if one voxels located exactly above contains class 3/6 and is non problematic
    # ---------------------------------------------------------------------------------------
    df['majority_class'] = df.iloc[:,df.columns.str.contains('_prev|_new')].idxmax(axis=1)

    df = non_prob_apparition(df, 'building')
    
    df = non_prob_apparition(df, 'vegetation')

    # ---------------------------------------------------------------------------------------
    # END OF DECISIONAL TREE 
    # ---------------------------------------------------------------------------------------
    # Change df so that the label and change_criticity are in a column of their own

    df['change_criticity_label'] = 0

    df['change_criticity_label'] = df.change_criticity.apply(lambda x: x.split(sep='-')[1]).astype(float)
    df['change_criticity'] = df.change_criticity.apply(lambda x: x.split(sep='-')[0])

    return df



if __name__ == '__main__':
    start_time = time.time()
    print('Starting criticity tree process...')

    parser = argparse.ArgumentParser(description="This script assigns to each voxel a level of criticity")
    parser.add_argument('-cfg', type=str, help='a YAML config file', default="./config_test.yml")
    args = parser.parse_args()


    with open(args.cfg) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    WORKING_DIR = cfg['working_dir']
    VOX_DF_PATH = cfg['criticity_tree']['data']['vox_df_path'] 
    OUTPUT_DIR = cfg['criticity_tree']['output_dir']

    os.chdir(WORKING_DIR)

    # Create the path for the folder to store the .csv file in case it doesn't yet exist
    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    tile_name, vox_dimension = os.path.basename(VOX_DF_PATH).split('.')[0].rsplit('_', maxsplit=1)
    vox_dimension = float(vox_dimension)/100

    df = pd.read_csv(os.path.join(VOX_DF_PATH))

    df = main(df, cfg, vox_dimension)

    # Save the new dataframe as csv

    saving_time = time.strftime("%d%m-%H%M")

    csv_file_name = f'{tile_name}_{str(int(vox_dimension*100))}_criticity-{saving_time}.csv'
    df.to_csv(os.path.join(OUTPUT_DIR, csv_file_name), index=False)

    # Save hyperparameters in JSON file with the same time as the .csv
    hyperparam_dict = cfg['criticity_tree']['threshold']

    with open(os.path.join(OUTPUT_DIR, f"{tile_name}_{str(int(vox_dimension*100))}_criticity-{saving_time}.json"), "w") as outfile: 
        json.dump(hyperparam_dict, outfile)

    print(f'\nCriticity assignement done, files for tile {tile_name} saved under {OUTPUT_DIR}')

    print(f'\nFinished entire voxelisation process in: {round(time.time()-start_time, 2)} sec.')