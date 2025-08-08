# -----------------------------------------------------------------------------
# File          : calibrate.py
# Description   : calibrates denoised data to match clean data distribution
# Author        : Daniel G. Li
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
from tqdm.contrib.concurrent import thread_map
import gc
from dotenv import load_dotenv
from os import getenv

load_dotenv()

# paths
PROJ_ROOT = getenv('PROJ_ROOT')
CLEAN_PATH = f'{DATA_PATH}_clean'
DENOISED_PATH = f'{DATA_PATH}_denoised'
CALIBRATED_PATH = f'{DATA_PATH}_calibrated'

# change this to choose model
MODEL_PATH = f'{PROJ_ROOT}/Models/segment/checkpoint_1600.pth'


def load_so2_flat(f):
    with np.load(f) as data:
        so2 = data['so2']
    so2 = np.clip(so2, 0, 20)
    so2 = np.nan_to_num(so2, 0, 20, 0)
    return so2.flatten()        


def gen_match_denoised(clean_std, clean_mean):
    def match_denoised(f):
        with np.load(f) as data:
            arrays = {key: data[key].copy() for key in data}
            denoised = data['denoised']
            flat = denoised.flatten()
            normed = (denoised - flat.mean()) / flat.std()
            calibrated = normed * clean_std + clean_mean
            np.savez(f'{CALIBRATED_PATH}/{f.name}', **arrays,
                     calibrated=calibrated)
        return 0
    return match_denoised


if __name__ == '__main__':
    clean_files = [f for f in Path(CLEAN_PATH).iterdir() if f.suffix == '.npz']
    print('calculating clean std and mean...')
    clean_imgs = thread_map(load_so2_flat, clean_files)
    clean_flat = np.concatenate([img for img in clean_imgs]).astype(np.float64)
    clean_std = clean_flat.std()
    clean_mean = clean_flat.mean()
    del clean_imgs
    del clean_flat
    gc.collect()
    print('done.')
    
    denoised_files = [f for f in Path(DENOISED_PATH).iterdir() if f.suffix == '.npz']
    '''
    print('calculating denoised std and mean...')
    denoised_imgs = thread_map(load_so2_flat, denoised_files)
    denoised_flat = np.concatenate([img for img in denoised_imgs]).astype(np.float64)
    denoised_std = denoised_flat.std()
    denoised_mean = denoised_flat.mean()
    del denoised_imgs
    del denoised_flat
    gc.collect()
    print('done.')
    '''
    
    print('matching denoised std and mean...')
    _ = thread_map(gen_match_denoised(clean_std, clean_mean), denoised_files)
    print('done.')
