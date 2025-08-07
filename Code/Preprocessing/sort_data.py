# -----------------------------------------------------------------------------
# File          : sort_data.py
# Description   : sort data into clean and noisy data
# Author        : Daniel G. Li
# -----------------------------------------------------------------------------

from util import *
from pathlib import Path
import numpy as np
from os import symlink
from tqdm.contrib.concurrent import thread_map


def sort_file(f):
    if (Path(f'{CLEAN_PATH}/{f.name}').exists() or
        Path(f'{NOISY_PATH}/{f.name}').exists()):
        return 1
    
    data = np.load(f)
    slope, intercept = get_line(data['so2'])
    data.close()
    if intercept < (-4 * slope + 3.8):
        symlink(f, f'{CLEAN_PATH}/{f.name}')
    else:
        symlink(f, f'{NOISY_PATH}/{f.name}')
    return 0


if __name__ == '__main__':
    files = [f for f in Path(CROPPED_PATH).iterdir() if f.suffix == '.npz']
    print('sorting cropped data into clean and noisy data')
    _ = thread_map(sort_file, files)
    print('done')
