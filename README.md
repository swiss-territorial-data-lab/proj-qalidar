# Cross-generation change detection between two classified LiDAR point clouds for a semi-automated quality control

**POSSIBLE TODOS TO POLISH THE CODE AND README**
- I left the class_equialence for VD and swisstopo, which was used for the creation of the last zone at the border between VD and NE. This is more for development purpose and I would actually remove it for the final deployment.
- Make the textual descript for the description in shapefile shorter (this is set in constant.py).
- Change the script 'change_detection.py' so that it use multiprocessing and run multiple tiles at once instead of one by one in the for loop (I have never done this so don't know exactly what's the best way of implementing it)
- Implement a new field in the clustered detection with the proportion of each criticality number (for ex. : #9:25%, #10:25%, #12:50%)
- Maybe do a script that downloads one tile from swisstopo, one from Neuchatel and place them in proper folder, so as to have an example of data to run the change detection methodology
- Maybe find a 'clearer' way of defining the kdtree and DBSCAN search radius for neighbourhood in the yaml fil (in the current implementation you have to put a value like 1.42 or else, we could do a dictionnary with 6 (neighbours) => 1*vox_dimension, 18 => 2**(1/2)*vox_dimension, 26 => 3**(1/2)*vox_dimension.. however this would imply that you cannot search for neighbours further than that -which you can in reality-, so I don't really know what is best...).
- Possibly use https://pypi.org/project/connected-components-3d/ instead of DBSCAN for filtering the isolated voxels

-------

**Table of content**

- [Introduction](#introduction)
- [Requirements](#requirements)
    - [Hardware](#hardware-requirements)
    - [Software](#software-requirements)
- [Data](#data)
    - [Point clouds](#point-clouds)
    - [Class equivalence](#class-equivalence)
- [Workflow](#workflow)
- [Additional information](#additional-information)

## Introduction

This project provides a set of scripts to detect changes between a reference point cloud and a new point cloud. The goal is to highlight area of change in the new point cloud to make the control process faster for an operator. <br>
It performs voxelization and then compare the class distribution in the voxels. The changes are classified by type and criticality level. The global workflow is summarized on Figure 1.

<div align="center" style="font-style: italic">
  <img
  src="img/overall_workflow.svg"
  alt="Workflow of project"
  width = "70%">
  <figcaption>Figure 1: Overview of the workflow for change detection and assignment of a criticality level to the detected changes.</figcaption>
</div>

The full documentation of the project is available on the STDL's [technical website](https://tech.stdl.ch/PROJ-QALIDAR/).

## Requirements

### Hardware requirements

No specific hardware is needed. However, the RAM must be bie enough for the dimension of the point cloud and its density. <br>
We conducted successfully our tests on a machine with 16 GB of RAM and point cloud tiles of dimension 2km x 2km with an approximate density of 15-20 pts/m<sup>2</sup> for the reference generation and 100 pts/<sup>2</sup> for the new generation.


### Software requirements

* Python 3.10: The dependencies may be installed with either `pip` or `conda`, by making use of the provided `requirements.txt` file. 
```bash
conda create -n <environment_name> python=3.10
conda activate <environment_name> 
pip install -r requirements.txt
```
* (Optional) [LAStools](https://lastools.github.io/): some pre-processing scripts rely on LAStools to reclassify the point clouds and modify the tiling. The change detection process itself relies only on python libraries.


## Data

### Point clouds

In order to run the change detections, at least two point clouds are required, one acting as the reference and the other as the point cloud to control. The expected format is LAS or LAZ.  <br>
The workflow is based on the assumption that the two point clouds cover the same area and have the same coordinate system (i.e. no point cloud registration is performed) <br>
It is necessary for the two tiles to share the same name, although the file formatting can differ. <br>

|   | Tile 1        | Tile 2              |
|---|---------------|---------------------|
| ✅ | 2533_1155.las | 2533_1155.las       |
| ✅ | tile_1.laz    | tile_1.las          |
| ❌ | 2533_1155.las | 2533000_1155000.las |
| ❌ | tile_prev.las | tile_new.las |

The reference and evaluated tiles are to be stored in two separate folders, whose paths must be provided in the yaml file with *prev_folder* and *new_folder* respectively. 

The script *retile_las.py* was used in order to create tiles of dimension 500 x 500 meters from tiles of dimension 1000 x 1000 meters. It may be useful as a basis for users who need to crop a set of tiles to fit the requirements mentioned above.

### Class equivalence
The correspondence between the old and new classes is needed. It must be provided in the CSV *classes_equivalence.csv*. 

Every class which is present in the newer point cloud must be provided in the *id* column. The overarching class from the reference generation must be indicated in the *matched_id* column. Note that the column *class_name* is purely for understandability purpose and does not need to be filled, or can even be removed. Observe that classes that are preserved should also be defined in the CSV file. The file provided in this repository is designed for usage with the classes from swisstopo as reference set and the classes of Canton Neuchâtel for the new classes. 

## Workflow

The change detection can be launched with:

```bash
python scripts/change_detection.py -cfg config.yml
```

With the default configuration, the change detection runs on all tiles provided in the input folder and produce a shapefile. The configuration can be adjusted through the file `config.yml`. <br>

The process goes through the following **substeps**:

1. **Voxelisation**: Creates a common grid of voxels for the two point clouds and resume the class distribution in each voxel in the form of a dataframe.
2. **Decision tree**: All voxels are assigned a criticality level.
3. **DBSCAN**:The problematic voxels are filtered out if they are isolated, following a clustering made with the algorithm DBSCAN
4. **Visualisation** : The detections are converted in a file format allowing for analysis. In 3D, a las file, in 2D a shapefile

If desired, each substep can be run individually on a single tile. For example:

```bash
python scripts/substeps/voxelisation.py -cfg config.yml
```

## Additional information

The full decision tree to sort the voxel is given here below in Figure 2. It sort the pixels by criticality level (non-problematic, grey zone, problematic) and by type of change.

<div align="center" style="font-style: italic">
  <img
  src="img/decisional_tree.svg"
  alt="Workflow of project"
  width = "100%">
  <figcaption>Figure 2: Decision tree.</figcaption>
</div>

The number for the types of change correspond to the following definition:

- Grey zone:
    - **7**: Appearance of a voxel or change in the class proportions due to *unclassified* points in the new generation;
    - **8**: Change in the class distribution due to extra classes present in the voxel compared to the reference generation. The neighboring voxels share the same class occupancy.

- Problematic:
    - **9**: Disappearance, i.e. a voxel which contains points in **v.1** but not in **v.2**. The neighboring voxels do not show the same change;
    - **10**: Appearance, i.e. a voxel which contains no points in **v.1** but is filled in **v.2**. The neighboring voxels do not show the same change;
    - **11**: Change in the class distribution due to extra classes present in the voxel compared to the reference generation. The neighboring voxels do not share the same class occupancy;
    - **12**: Changes in the distribution for classes previously and newly present in the voxel;
    - **13**: Presence of points classified as noise in **v.2**.