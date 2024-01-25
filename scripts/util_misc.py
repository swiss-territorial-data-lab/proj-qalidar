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