# -----------------------------------------------------------------------------
# File          : util.py
# Description   : variables and functions for processing data
# Author        : Daniel G. Li
# -----------------------------------------------------------------------------

# imports
from dotenv import load_dotenv
from os import getenv
import numpy as np
from scipy.stats import linregress

load_dotenv()

# paths
PROJ_ROOT = getenv('PROJ_ROOT')
DATA_PATH = f'{PROJ_ROOT}/Data/TROPOMI/S5P_L2__SO2____HiR'
CROPPED_PATH = f'{DATA_PATH}_cropped'
CLEAN_PATH = f'{DATA_PATH}_clean'
NOISY_PATH = f'{DATA_PATH}_noisy'
ADDED_NOISE_PATH = f'{DATA_PATH}_added_noise'
ADDED_FLAG_PATH = f'{DATA_PATH}_added_flag'


# get power spectrum
def get_power_spectrum(noise):
    f = np.fft.fft2(noise)
    f_shifted = np.fft.fftshift(f)
    power = np.abs(f_shifted)**2
    return power


# perform linear regression on loglog of power spectrum (radius, power)
def get_line(noise):
    rows, cols = noise.shape
    y_i, x_i = np.indices((rows, cols))

    radius_grid = np.sqrt((x_i - cols/2)**2 + (y_i - rows/2)**2)
    X, y = radius_grid.flatten(), get_power_spectrum(noise).flatten()
    # X[X <= 0], y[y <= 0] = 1e-9, 1e-9
    X[X == 0] = 1
    X, y = np.log10(X), np.log10(y)
    slope, intercept, _, _, _ = linregress(X, y)

    return slope, intercept
