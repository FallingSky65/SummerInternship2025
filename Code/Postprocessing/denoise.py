# -----------------------------------------------------------------------------
# File          : denoise.py
# Description   : applies the denoising model to noisy data
# Author        : Daniel G. Li
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
from tqdm import tqdm
from matplotlib.colors import LogNorm
import torch
from dotenv import load_dotenv

load_dotenv()

# paths
PROJ_ROOT = getenv('PROJ_ROOT')
NOISY_PATH = f'{DATA_PATH}_noisy'
DENOISED_PATH = f'{DATA_PATH}_denoised'

# change this to choose model
MODEL_PATH = f'{PROJ_ROOT}/Models/denoise/checkpoint_0032.pth'

import sys
sys.path.append(f'{PROJ_ROOT}/Code/ML')
from ddunet import DDUNet


def invnorm(x, vmin=0.5, vmax=10):
    log_vmin = np.log10(vmin)
    log_vmax = np.log10(vmax)
    log_x = x * (log_vmax - log_vmin) + log_vmin
    return 10 ** log_x


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH)
    model = DDUNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    norm = LogNorm(vmin=0.5, vmax=10, clip=True)

    print('adding noise to clean data')
    for f in tqdm(Path(NOISY_PATH).iterdir()):
        if f.is_file() and f.suffix == '.npz':
            if Path(f'{DENOISED_PATH}/{f.name}').exists():
                continue

            data = np.load(f)
            arrays = {key: data[key].copy() for key in data}
            so2 = data['so2']
            data.close()

            xnorm = torch.from_numpy(norm(so2)).unsqueeze(0).unsqueeze(0).to(device)
            denoisenorm = model(xnorm)
            denoised = invnorm(denoisenorm.cpu().detach()[0, 0]).numpy()

            np.savez(f'{DENOISED_PATH}/{f.name}', **arrays,
                     denoised=denoised)
    print('done')
