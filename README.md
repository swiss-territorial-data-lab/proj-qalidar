## Project Quality Assessment of LiDAR data

Organisation of the repo:

- **code**:
    - **retile_las.py**: Script to have tiles of matching dimensions between the two generations
    - **reclassify_pointcloud.py**: Script to have a point cloud reclassified given the matching class provided in *classes_equivalences.csv*
    - **util_las.py**: Utilitary functions to process .las data
    - **voxelisation.ipynb**: Notebook to create the .csv file countaining the voxelised comparison of the previous and new point cloud
    - **criticity_tree.ipynb**: Evaluation by a decisional tree to determine the criticity level of all voxels
    - **criticity_changes_analysis.ipynb**: Plots creation for interpretation of the results from *criticity_tree.ipynb*
    - **criticity_changes_to_file.ipynb**: Notebook to transform the .csv from criticity_tree to a file format allowing to display the voxels (.las, shapefile or .ply mesh)
- **classes_equivalence.csv**: Definition of how the classes of the new point cloud should be matched to the previous generation

To run the change detection method, one must have two point cloud tiles with the same origin and dimensions. The procedure is as follows: 
1. Set path for both tiles in *voxelisation.ipynb*, the desired voxels dimension can also be changed. Run the Notebook. This creates a .csv file where each row corresponds to a given voxel with the occupancy in said voxel for the previous and new point cloud provided. The .csv file is stored in the folder /out_dataframe/voxelised_comparison which is created automatically.
2. To get the criticity level of each voxel, run *criticity_tree.ipynb*. The input file name at the begining of the notebook must be set accordingly to the outgoing file from step 1. This creates a .csv file similar to the one created in step 1, but with new columns, the most important one being *change_criticity_label* which provides for each voxel the label coming out of the criticity tree. The .csv file is stored in the folder /out_dataframe/criticity_changes_df which is created automatically.
3. To get plots summarizing the results from  *criticity_tree.ipynb*, run *criticity_changes_analysis.ipynb*. If desired the plots can be saved to html. They are then stored in the /plots folder. Use the *criticity_changes_to_file.ipynb* notebook if you want to visualize the voxels for further analysis. The output is stored in the folder /out_vis. In both case, the input file name must be set at the beginning of the notebook accordingly to the outgoing file from step 2.