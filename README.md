# Project Quality Assessment of LiDAR data

**POSSIBLE TODOS TO POLISH THE CODE AND README**
- Make the textual descript for the description in shapefile shorter (this is set in constant.py).
- Change the script 'change_detection.py' so that it use multiprocessing and run multiple tiles at once instead of one by one in the for loop (I have never done this so don't know exactly what's the best way of implementing it)
- Implement a new field in the clustered detection with the proportion of each criticality number (for ex. : #9:25%, #10:25%, #12:50%)
- Maybe do a script that downloads one tile from swisstopo, one from Neuchatel and place them in proper folder, so as to have an example of data to run the change detection methodology
- Maybe find a 'clearer' way of defining the kdtree and DBSCAN search radius for neighbourhood in the yaml fil (in the current implementation you have to put a value like 1.42 or else, we could do a dictionnary with 6 (neighbours) => 1*vox_dimension, 18 => 2**(1/2)*vox_dimension, 26 => 3**(1/2)*vox_dimension.. however this would imply that you cannot search for neighbours further than that -which you can in reality-, so I don't really know what is best...).
- Possibly use https://pypi.org/project/connected-components-3d/ instead of DBSCAN for filtering the isolated voxels

-------
This projet provides a script allowing to obtain change detections between a reference and a new point cloud, based on a voxel comparison method.

## Hardware requirements

No specific requirements. However the dimension of the point cloud tile or the density of said point cloud can be too large for the setup's RAM.
We conducted successfully our tests on a machine with 16 GB of RAM and point cloud tile of dimension 2km x 2km with an approximate density of 15-20 pts/m<sup>2</sup>.


## Software Requirements

* Python 3.10: The dependencies may be installed with either `pip` or `conda`, by making use of the provided `requirements.txt` file. 
```bash
conda create -n <environment_name> python=3.10
conda activate <environment_name> 
pip install -r requirements.txt
```
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
It is necessary for the two tiles to share the same name, although the file formatting can differ. <br>
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
This script relies on four subscripts placed in the folder *substeps*. All of the directories, threshold or other parameters must be defined in the yaml config file. It is separated in two main part, with the **main parameters** and the **sub parameters**. <br>
If desired, the substeps scripts can be run individually on a single tile, for example:
```bash
python scripts/substeps/voxelisation.py -cfg config.yml
```
Note that in and out path must be properly set in the yaml file under the corresponding section.

The change detection process goes through these steps:

1. **Voxelisation**: Puts the two point clouds on a common grid and create voxels information in the form of a DataFrame.
2. **Decision tree**: All of the voxels are assigned to a specific criticality level.
3. **DBSCAN**:The problematic voxels are filtered out if they are isolated, following a clustering made with the algorithm DBSCAN
4. **Visualisation** : The detections are converted in a file format allowing for analysis. In 3D, a las file, in 2D a shapefile



