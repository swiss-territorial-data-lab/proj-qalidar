'''
Script retiling the older generation point cloud tile from swisstopo to match the size of the new generation.
It assumes that the name for the new tiles consist of the complete coordinates in LV95 format, 
separated by an underscore, for example: 2527000_1190500.laz 
The dimension of the tile must be properly set.
'''
import os

lastools_path = '/home/nmunger/Desktop/LAStools/bin/'

new_data_folder = '/mnt/s3/proj-qalidar/02_Data/LiDAR_data/2022_Neuchatel/lidar2022_classified/'
old_data_folder = '/mnt/s3/proj-qalidar/02_Data/LiDAR_data/G1_NE_2018-2019_NF02/'
destination_folder = '/mnt/s3/proj-qalidar/02_Data/LiDAR_data/2018_NE_retiled/'

new_data_format = '.laz'
old_data_format = '.laz'
destination_format = '.las'
tile_dimension = 500

for filename in os.listdir(new_data_folder):
    
    x_origin, y_origin = filename.replace(f'{new_data_format}','').split('_') # Keep only the coordinates

    # Find the tiles with the same coordinates and reshape it with the proper tiling
    os.system(f'wine {lastools_path}las2las.exe \
         -i {os.path.join(old_data_folder, x_origin[:4] + "_" + y_origin[:4] + old_data_format)} \
         -keep_tile {x_origin} {y_origin} {tile_dimension} \
         -o {os.path.join(destination_folder, x_origin + "_" + y_origin + destination_format)}')