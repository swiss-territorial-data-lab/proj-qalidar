import os
import argparse
import pandas as pd
import time

'''
Script that returns a .las file with the mapped classification as desired 
from the 'classes_equivalences.csv' file
This is useful when wanting to compare pointclouds in CloudCompare for example
'''
start_time = time.time()

lastools_path = '/home/nmunger/Desktop/LAStools/bin/' 
classes_equivalence_path ='/home/nmunger/Documents/classes_equivalence.csv'

default_pc = '/home/nmunger/Desktop/2547500_1212000.laz'
default_dest_folder = '/home/nmunger/Desktop/'

parser = argparse.ArgumentParser()

parser.add_argument('--pc', help='path to original point cloud', default='/home/nmunger/Desktop/2547000_1212000.laz')
parser.add_argument('--dest', help='destination folder for reclassified point cloud', default='/home/nmunger/Desktop')
parser.add_argument('--format', help='las or laz', default = 'laz')
parser.add_argument('--thin', help='When True, thins the point cloud', default = True)
args = parser.parse_args()

df = pd.read_csv(classes_equivalence_path,sep=';')
df_ids_to_change = df[df['id']!=df['matched_id']]

lastool_command = f'wine {lastools_path}las2las.exe -i {args.pc} '

for id in range(len(df_ids_to_change)):
    if df_ids_to_change["matched_id"].iloc[id] == -1:
        # If the matched_id is -1, means that we want to drop this class, for example 'groud level noise'
        lastool_command = lastool_command + f'-drop_class {df_ids_to_change["id"].iloc[id]} '
    else:
        # Else change the class
        lastool_command = lastool_command + f'-change_classification_from_to {df_ids_to_change["id"].iloc[id]} {df_ids_to_change["matched_id"].iloc[id]} '

if args.thin == True:
    lastool_command = lastool_command + f'-keep_every_nth 5 '

file_name, _ = os.path.basename(args.pc).split('.')
lastool_command = lastool_command + f'-o {os.path.join(args.dest,file_name)}_reclassified.{args.format}'

print('LAStools command: \n \n', lastool_command)

os.system(lastool_command)

print('\n Finished reclassifying the point cloud in ', round(time.time()-start_time, 2), 'seconds')

