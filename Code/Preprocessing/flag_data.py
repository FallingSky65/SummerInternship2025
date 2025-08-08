# -----------------------------------------------------------------------------
# File          : flag_data.py
# Description   : adds an additional flag to clean data based on a threshold
# Author        : Daniel G. Li
# -----------------------------------------------------------------------------

# imports
from util import *
import numpy as np
from pathlib import Path
import cv2
from tqdm.contrib.concurrent import thread_map


def getflag(so2):
    blurred = cv2.blur(so2, (3, 3))
    _, blurthreshflag = cv2.threshold(blurred, 0.5, 1, cv2.THRESH_BINARY)

    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(
        blurthreshflag.astype(np.uint8), connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 20:
            blurthreshflag[labels_im == i] = 0

    return blurthreshflag


def addflag(f):
    data = np.load(f)
    arrays = {key: data[key].copy() for key in data}
    so2 = data['so2']
    so2_added_flag = getflag(so2)
    data.close()
    np.savez(f'{ADDED_FLAG_PATH}/{f.name}', **arrays,
             so2_added_flag=so2_added_flag)
    return 0


if __name__ == '__main__':
    files = [f for f in sorted(Path(CLEAN_PATH).iterdir()) if f.suffix == '.npz']
    print('adding flag...')
    _ = thread_map(addflag, files)
    print('done')
