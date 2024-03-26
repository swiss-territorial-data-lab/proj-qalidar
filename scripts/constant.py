
## Criticality tree descript:
criticality_dict=([
    (7, 'Increase in the unclassified points'),
    (8, 'Presence of extra classes in the area'),
    (9,'Disappearance of geometry'),
    (10, 'Appearance of geometry'),
    (11, 'Isolated minor class change'),
    (12, 'Major change in the class distribution'),
    (13, 'Noise')
])

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
