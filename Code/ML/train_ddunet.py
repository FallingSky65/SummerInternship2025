# -----------------------------------------------------------------------------
# File          : train_ddunet.py
# Description   : trains ddunet
# Author        : Daniel G. Li
# -----------------------------------------------------------------------------

# imports
from train_utils import Params, spawn_train, PROJ_ROOT, ADDED_NOISE_PATH
from ddunet import DDUNet
from torch.nn import MSELoss
import numpy as np
from matplotlib.colors import LogNorm


norm = LogNorm(vmin=0.5, vmax=10.0, clip=True)


def load_denoise(npzfile):
    with np.load(npzfile) as npz:
        x = npz['so2_added_noise'].astype(np.float32)
        y = npz['so2'].astype(np.float32)
        x, y = norm(x), norm(y)
        x = x[np.newaxis, np.newaxis, :, :]
        y = y[np.newaxis, np.newaxis, :, :]
        xy = np.concatenate([x, y], axis=1)
        return xy


if __name__ == '__main__':
    p = Params(
        Model=DDUNet,
        Criterion=MSELoss,
        Checkpoint=None,
        DATA_PATH=ADDED_NOISE_PATH,
        LOAD_FUNC=load_denoise,
        LEARNING_RATE=1e-6,
        BATCH_SIZE=8,
        BATCH_CHUNKS=8,
        EPOCH_START=0,
        EPOCH_END=256,
        MASTER_PORT='42067',
        SAVE_PATH=f'{PROJ_ROOT}/Models/denoise'
    )
    spawn_train(p)
