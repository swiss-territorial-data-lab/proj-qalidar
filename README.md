## Project Quality Assessment of LiDAR data

Organisation of the repo:

- **code**:
    - **retile_las.py**: script have tiles of matching dimension between the two generations
    - **reclassify_pointcloud.py**: script to have a point cloud reclassified given the matching class provided in *classes_equivalences.csv*
    - **util_las.py**: utilitary functions to process .las data
    - **voxelisation.ipynb**: Notebook to create the .csv file countaining the voxelised comparison of the pervious and new point cloud
    - **criticity_tree.ipynb**: evaluation by a decisional tree to determine the criticity level of all voxels
    -**criticity_changes_analysis.ipynb**: Plots creation for interpretation of the results from *criticity_tree.ipynb*
    -**criticity_changes_to_file.ipynb**: Notebook to transform the .csv from criticity_tree to a file format allowing to display the voxels (.las, shapefile or .ply mesh)
- **classes_equivalence.csv**: definition of how the classes of the new point cloud should be matched to the previous generation