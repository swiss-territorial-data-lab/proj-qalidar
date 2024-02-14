UNCLASSIFIED = 1
GROUND = 2
VEGETATION = 3
BUILDING = 6
NOISE = 7
WATER = 9
BRIDGE = 17

## Criticality tree descript:
criticality_dict=([(9,'In a previously containing voxel, there is no more geometries (disappearance of geometry)'),
       (10, 'In a previously empty voxel, new points appears (appearance of geometry)'),
       (11, 'Isolated minor class change (other reference classes behaves similarly)'),
       (12, 'Major class distribution change'),
       (13, 'Noise to check')])

# Taken from https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
