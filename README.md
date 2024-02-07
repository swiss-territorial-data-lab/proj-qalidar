# Project Quality Assessment of LiDAR data

This projet provides a script allowing to obtain change detections between a reference and a new point cloud, based on a voxel comparison method.

## Hardware requirements

No specific requirements. However the dimension of the point cloud tile or the density of said point cloud can be too large for the setup's RAM.
We conducted our succefuly our tests on a machine with 16 GB of RAM. And point cloud tile of dimension 2km x 2km with an approximate density of 15-20 pts/m<sup>2</sup>.


## Software Requirements

* Python 3.10: The dependencies may be installed with either `pip` or `conda`, by making using the provided `requirements.txt` file. 
* (Optional) [LAStools](https://lastools.github.io/): some of the scripts rely on this tool set in order to preprocess the data. The change detection process itself relies only on python libraries. Those scripts are indicated in the folder structure description.  


## Folder structure
```bash
proj-qalidar
├── scripts
│   ├── change_detection.py     
│   ├── substeps
│   │   ├── voxelisation.py
│   │   ├── decision_tree.py
│   │   ├── dbscan.py
│   │   └── visualisation.py
│   ├── constant.py
│   ├── utils_las.py
│   ├── utils_misc.py
│   ├── retile_las.py                   #requires LAStools
│   ├── reclassify_pointcloud.py        #requires LAStools
│   └── plots_creation
│       ├── sankey.ipynb
│       └── detections_analysis.ipynb
├── config.yml
├── requirements.txt 
└── data 
    └── classes_equivalence.csv         # One to one mapping of the new classes to the reference classes

```

## Procedure
### Valid data:
In order to run the change detections process, at least two distinct point clouds are required, one acting as the reference, the other being the one to evaluate. The expected format is LAS or LAZ.  <br>
The workflow is based on the assumption that the two point clouds cover the same area and have the same coordinate system (i.e. no point cloud registration is performed) <br>
It is necessary for the two tiles to share the same name, although the file formating can differ. <br>
<p align="center">


|   | Tile 1        | Tile 2              |
|---|---------------|---------------------|
| ✅ | 2533_1155.las | 2533_1155.las       |
| ✅ | tile_1.laz    | tile_1.las          |
| ❌ | 2533_1155.las | 2533000_1155000.las |
| ❌ | tile_prev.las | tile_new.las |
</p>
The reference and evaluated tiles are to be stored in two separate folders, whose path must be provided in the yaml file with *prev_folder* and *new_folder* respectively. 

The script *retile_las.py* was used in order to create tiles of dimension 500 x 500 meters from tiles of dimension 1000 x 1000 meters. It may be useful as a basis for users who need to crop a set of tiles to fit the requirements mentioned above.

### Change detection

Once the data is set as described in the previous section, the change detection process can be launched with:
```bash
python scripts/change_detection.py -cfg config.yml
# cfg defaults to config.yml if no argument is provided
```
With the default settings of the configuration, it will run the change detection on all tiles provided and produce a mapping in shapefile.
This script relies on four subscripts placed in the folder *substeps*. All of the directories, threshold or other parameters must be defined in the yaml config file. It is separated in two main part, with the **main parameters** and the **sub parameters**. 
If desired, the subscripts can be run individually on a single tile, for example.
```bash
python scripts/substeps/voxelisation.py -cfg config.yml
```
Note that in and out path must be properly set in the yaml file uner the corresponding section.

The change detection process goes through these steps:

1. **Voxelisation**: Puts the two point clouds on a common grid and create voxels information in the form of a DataFrame.
2. **Decision tree**: All of the voxels are assigned to a specific criticality level.
3. **DBSCAN**:The problematic voxels are filtered out if they are isolated, following a clustering made with the algorithm DBSCAN
4. **Visualisation** : The detections are converted in a file format allowing for analysis. In 3D, a las file, in 2D a shapefile
- **code**:
    - **retile_las.py**: Script to have tiles of matching dimensions between the two generations
    - **reclassify_pointcloud.py**: Script to have a point cloud reclassified given the matching class provided in *classes_equivalences.csv*
    - **util_las.py**: Utilitary functions to process .las data
    - **voxelisation.ipynb**: Notebook to create the .csv file containing the voxelised comparison of the previous and new point cloud
    - **criticity_tree.ipynb**: Evaluation by a decisional tree to determine the criticity level of all voxels
    - **criticity_changes_analysis.ipynb**: Plots creation for interpretation of the results from *criticity_tree.ipynb*
    - **criticity_changes_to_file.ipynb**: Notebook to transform the .csv from criticity_tree to a file format allowing to display the voxels (.las, shapefile or .ply mesh)
- **classes_equivalence.csv**: Definition of how the classes of the new point cloud should be matched to the previous generation

To run the change detection method, one must have two point cloud tiles with the same origin and dimensions. The procedure is as follows: 
1. Set path for both tiles in *voxelisation.ipynb*, the desired voxels dimension can also be changed.
    - Run the Notebook. 
        - This creates a .csv file where each row corresponds to a given voxel with the occupancy in said voxel for the previous and new point cloud provided.
        - The .csv file is stored in the folder /out_dataframe/voxelised_comparison which is created automatically.
2. To get the criticity level of each voxel, run *criticity_tree.ipynb*. 
    - The input file name at the beginning of the notebook must be set accordingly to the outgoing file from step 1. 
        - This creates a .csv file similar to the one created in step 1, but with new columns, the most important one being *change_criticity_label* which provides for each voxel the label coming out of the criticity tree. 
        - The .csv file is stored in the folder /out_dataframe/criticity_changes_df which is created automatically.
3. To get plots summarizing the results from  *criticity_tree.ipynb*, run *criticity_changes_analysis.ipynb*. 
    - If desired the plots can be saved to html. They are then stored in the /plots folder.
4.  Use the *criticity_changes_to_file.ipynb* notebook if you want to visualize the voxels for further analysis. 
    - The output is stored in the folder /out_vis. In both case, the input file name must be set at the beginning of the notebook accordingly to the outgoing file from step 2.


