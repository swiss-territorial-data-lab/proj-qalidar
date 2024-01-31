import os
import pathlib
import numpy as np

def cosine_similarity(voxels_to_evaluate):
    """Calculate the cosine similarity along the lines of a dataframe between columns ending with '_prev' and the ones ending with '_new'.

    Args:
        voxels_to_evaluate (DataFrame): Voxels with columns describing the number of points in each class.

    Returns:
        cosine_similarity: 1-D array
    """

    voxels_to_evaluate_prev = voxels_to_evaluate.iloc[:, voxels_to_evaluate.columns.str.endswith('_prev')].to_numpy()
    voxels_to_evaluate_new = voxels_to_evaluate.iloc[:, voxels_to_evaluate.columns.str.endswith('_new')].to_numpy()
    dot_product = np.sum(voxels_to_evaluate_prev * voxels_to_evaluate_new, axis=1)
    product_of_norm = np.linalg.norm(voxels_to_evaluate_prev, axis=1)*np.linalg.norm(voxels_to_evaluate_new, axis=1)

    # For cases where one vector is completely empty, avoid division by zero and replace by -1
    cosine_similarity = np.divide(dot_product, product_of_norm, out = np.full_like(dot_product, -1), where = product_of_norm!=0)

    return cosine_similarity

def verify_out_folder(path):
    '''Checks if the folder to which the path leads is empty or not and handle possible overwriting. 
        If the folder doesn't exist it will be created
    Args:
        path: path to the folder to be evaluated
    Returns:
        bool: Whether the folder is safe to use or not.'''
    folder_exists = os.path.isdir(path)

    if folder_exists == False:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        return True
    
    elif len(os.listdir(path)) != 0:
        answer=input(f"The folder you've provided is not empty ({path}). \nIf you continue, files might be overwritten. Do you wish to continue? [y/n]\n")
        
        if answer.lower() in ['y','yes']:
            return True
        elif answer.lower() in ['n','no']:
            print('Exiting')
            return False
        else:
            print('Wrong input. Exiting.')
            return False
    
    else:
        return True