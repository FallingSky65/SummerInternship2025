# -----------------------------------------------------------------------------
# File          : train_unet.py
# Description   : trains unet
# Author        : Daniel G. Li
# -----------------------------------------------------------------------------

# imports
from train_utils import Params, spawn_train, PROJ_ROOT, ADDED_FLAG_PATH
from unet import UNet
from dice_loss import DiceLoss
import numpy as np
from matplotlib.colors import LogNorm


norm = LogNorm(vmin=0.5, vmax=10.0, clip=True)


def load_segment(npzfile):
    with np.load(npzfile) as npz:
        x = npz['so2'].astype(np.float32)
        y = npz['so2_added_flag'].astype(np.float32)
        x, y = norm(x), np.clip(y, 0, 1)
        x = x[np.newaxis, np.newaxis, :, :]
        y = y[np.newaxis, np.newaxis, :, :]
        xy = np.concatenate([x, y], axis=1)
        return xy


if __name__ == '__main__':
    p = Params(
        Model=UNet,
        Criterion=DiceLoss,
        Checkpoint=None,
        DATA_PATH=ADDED_FLAG_PATH,
        LOAD_FUNC=load_segment,
        LEARNING_RATE=1e-6,
        BATCH_SIZE=64,
        BATCH_CHUNKS=1,
        EPOCH_START=0,
        EPOCH_END=256,
        MASTER_PORT='42267',
        SAVE_PATH=f'{PROJ_ROOT}/Models/segment'
    )
    spawn_train(p)
