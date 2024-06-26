'''
Utilitary function for .las and .laz file manipulation
'''

import laspy
import numpy as np
import pandas as pd
import math


# Given a .las or .laz file, return the desired characteristics 
# (x,y,z coord. and classification) in a numpy array 
def las_to_np_xyzclass(las_file_path):
    las_file = laspy.read(las_file_path)
    las_x = np.array(las_file.points.x)
    las_y = np.array(las_file.points.y)
    las_z = np.array(las_file.points.z)
    las_class = np.array(las_file.points.classification).astype(int)
  
    las_xyzclass = np.vstack((las_x,las_y,las_z,las_class)).T

    return las_xyzclass

# Given a .las or .laz file, return the desired characteristics 
# (x,y,z coord. and intensity) in a dataframe 
def las_to_df_xyzintensity(las_file_path):
    las_file = laspy.read(las_file_path)
    las_x = np.array(las_file.points.x)
    las_y = np.array(las_file.points.y)
    las_z = np.array(las_file.points.z)
    las_intensity = np.array(las_file.points.intensity).astype(int) #By default, uint64, can cause trouble with data manipulation
  
    df_xyzintensity = pd.DataFrame({"X": las_x, "Y": las_y, "Z": las_z, "intensity": las_intensity})
    
    return df_xyzintensity

# Given a .las or .laz file, return the desired characteristics 
# (x,y,z coord. and class) in a dataframe 
def las_to_df_xyzclass(las_file_path):
    las_file = laspy.read(las_file_path)
    las_x = np.array(las_file.points.x)
    las_y = np.array(las_file.points.y)
    las_z = np.array(las_file.points.z)
    las_class = np.array(las_file.points.classification).astype(int)     
  
    df_xyzclass = pd.DataFrame({"X": las_x, "Y": las_y, "Z": las_z, "classification": las_class})
    
    return df_xyzclass

#Returns a Dataframe with coordinates of point (X,Y,Z), intensity and classification
def las_to_df_xyzintensityclass(las_file_path):
    las_file = laspy.read(las_file_path)
    las_x = np.array(las_file.points.x)
    las_y = np.array(las_file.points.y)
    las_z = np.array(las_file.points.z)
    las_intensity = np.array(las_file.points.intensity).astype(int) #By default, uint64, can cause trouble with data manipulation
    las_class = np.array(las_file.points.classification).astype(int)
    
    df_xyzintensityclass = pd.DataFrame({"X": las_x, "Y": las_y, "Z": las_z, "intensity": las_intensity, "classification":las_class})
    
    return df_xyzintensityclass


def df_columns_sanity_check(df, column_name):
    """Verifies that the column of the dataframe can be transformed to a field of the LAS file"""

    if column_name not in df:
        print(f"The column name ({column_name}) wasn't found in the DataFrame. This custom field will be ignored.")
   
    elif df[column_name].dtype not in [int, float, 'int8', 'int32', 'int64', 'float64']:
        print(f"This column ({column_name}) is not stored in int or float format. This custom field will be ignored.")

    elif np.any(df[column_name]<0):
        print(f"This column ({column_name}) has the valid type, but has values lower than zeros. This custom field will be ignored")

    else:
        return True

    return False


def df_to_las(df, user_data_col = 'criticality_number', point_source_id_col = 'clusters', intensity_col = 'cluster_criticality_number'):
    '''
    Create a las file given a dataframe. Create the field user_data with the content of the
        column given in input. Change index_to_point_source_id to True to save the index of the dataframe to
        point_source_id. Note that this field is stored in unsigned short, so it only works for dataframe shorter 
        than 65,535 rows
    '''

    header = laspy.LasHeader(point_format=0)
    out_las = laspy.LasData(header)
    out_las.x = df.X_grid
    out_las.y = df.Y_grid
    out_las.z = df.Z_grid
    
    if (user_data_col is not None) and (df_columns_sanity_check(df, user_data_col)):
        out_las.user_data = df[user_data_col]

    if (point_source_id_col is not None) and (df_columns_sanity_check(df, point_source_id_col)):
        out_las.point_source_id = df[point_source_id_col]

    if (intensity_col is not None) and (df_columns_sanity_check(df, intensity_col)):
        out_las.intensity = df[intensity_col]

    return out_las   

def reclassify(df, path_correspondance_csv, drop_ground_level_noise = True):
    ''' Returns the dataframe with the mapped classification 
        The .csv should have a column "id" which are the original classes and 
        "matched_id" which are the mapped classes
        Classes which are mapped to -1 are removed from the dataframe if drop_ground_level_noise is True. '''
    
    class_eq_df = pd.read_csv(path_correspondance_csv, sep=';')

    reclassified_df = df.merge(class_eq_df[['id','matched_id']],how='left', left_on='classification', right_on='id')\
                        .drop(columns=['classification', 'id']).rename(columns={'matched_id':'classification'})

    if drop_ground_level_noise:
        reclassified_df = reclassified_df[reclassified_df.classification != -1]

    return reclassified_df

# Return the voxelised version of a dataframe describing a pointcloud
def to_voxelised_df(df_pc, grid_origin, grid_max, vox_xy, vox_z):
    '''
    - df_pc : pandas dataframe with the x,y,z coordinates and the class for each point of a point cloud
    - grid_origin : tuple, (x,y,z) coordinates of the origin of the voxelisation grid
    - grid_max : tuple, (x,y,z) coordinates of the maximum value of the voxelisation grid
    - vox_xy : dimension, in meters, of the width and depth to use for voxel creation
    - vox_z : dimension, in meters, of the height to use for voxel creation
    '''
    x_origin, y_origin, z_origin = grid_origin
    x_max, y_max, z_max = grid_max

    bins = np.arange(int(x_origin), math.ceil(x_max)+vox_xy, vox_xy)
    df_pc['X_grid'] = pd.cut(df_pc['X'], bins=bins, right=False)
    df_pc['X_grid'] = df_pc['X_grid'].apply(lambda bin: bin.mid) # Set the middle of the bin as the coordinate

    bins = np.arange(int(y_origin), math.ceil(y_max)+vox_xy, vox_xy)
    df_pc['Y_grid'] = pd.cut(df_pc['Y'],bins=bins, right=False)
    df_pc['Y_grid'] = df_pc['Y_grid'].apply(lambda bin: bin.mid) # Set the middle of the bin as the coordinate

    bins = np.arange(int(z_origin), math.ceil(z_max)+vox_z, vox_z)
    df_pc['Z_grid'] = pd.cut(df_pc['Z'], bins=bins, right=False)
    df_pc['Z_grid'] = df_pc['Z_grid'].apply(lambda bin: bin.mid) # Set the middle of the bin as the coordinate

    # Create column with the number of points for each class in each cell
    grouped_by_class_df = df_pc.groupby(
        ['X_grid','Y_grid','Z_grid','classification'], 
        observed=True
    )['classification'].count().to_frame('nb_points').reset_index()

    # Create unique voxel IDs to facilitate the pivot
    grouped_by_class_df['vox_id'] = grouped_by_class_df.groupby(['X_grid','Y_grid','Z_grid'], observed=True).ngroup()
    
    # Pivot on voxel IDs
    pivoted_df = pd.pivot_table(grouped_by_class_df,values='nb_points', index=['vox_id'], columns='classification').reset_index() 
    
    # Removes the column title 'classification' which is confusing otherwise
    pivoted_df.columns.name = None 

    # Reattribute the voxel location
    voxel_df = grouped_by_class_df[['vox_id','X_grid','Y_grid','Z_grid']].drop_duplicates().merge(pivoted_df, on='vox_id',how='left') 

    voxel_df.drop(columns=['vox_id'],inplace=True)

    return voxel_df



