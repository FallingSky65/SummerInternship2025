# -----------------------------------------------------------------------------
# File          : add_noise.py
# Description   : adds artificial noise to clean data to mimic noisy data
# Author        : Daniel G. Li
# -----------------------------------------------------------------------------

# imports
from pathlib import Path
import numpy as np
from scipy.stats import linregress
from tqdm.contrib.concurrent import thread_map
from random import seed, random
from util import *


# generate red noise
def gen_red_noise(size=(320,320)):
    white_noise = np.random.normal(loc=0, scale=1, size=size)
    f_transform = np.fft.fft2(white_noise)
    f_transform_shifted = np.fft.fftshift(f_transform)

    rows, cols = size
    cy, cx = rows // 2, cols // 2

    x, y = np.arange(-cx, cx), np.arange(-cy, cy)
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X**2 + Y**2)
    radius[radius == 0] = 1

    f_filtered = f_transform_shifted / (radius ** 2)
    f_ishifted = np.fft.ifftshift(f_filtered)
    red_noise = np.fft.ifft2(f_ishifted)

    return np.real(red_noise)


# generate red noise matching a line
def gen_noise(boost=0.3):
    red_noise = gen_red_noise()

    rows, cols = red_noise.shape
    y_i, x_i = np.indices((rows, cols))
    radii = np.sqrt((x_i - cols/2)**2 + (y_i - rows/2)**2)

    m_a, b_a = get_line(red_noise)
    m_b, b_b = -0.13, 4.95 + boost

    f = np.fft.fft2(red_noise)
    f_shifted = np.fft.fftshift(f)
    phase = np.angle(f_shifted)

    p_a = get_power_spectrum(red_noise)
    p_a[p_a <= 0] = 1e-9
    p_a = np.log10(p_a)

    radii[radii == 0] = 1e-9
    radii = np.log10(radii)

    p_a -= m_a * radii + b_a
    p_a += m_b * radii + b_b

    p_a += 1 / ((2 * radii)**3 + 1)

    p_a = np.power(10, p_a)

    f_modified = np.sqrt(p_a) * np.exp(1j * phase)

    return np.real(np.fft.ifft2(np.fft.ifftshift(f_modified)))


def add_noise(f):
    if Path(f'{ADDED_NOISE_PATH}/{f.name}').exists():
        return 1

    data = np.load(f)
    arrays = {key: data[key].copy() for key in data}
    so2 = data['so2']
    so2_added_noise = so2 + gen_noise(boost=random()*1.2-1.0)
    data.close()
    np.savez(f'{ADDED_NOISE_PATH}/{f.name}', **arrays,
             so2_added_noise=so2_added_noise)
    return 0


if __name__ == '__main__':
    seed(67)
    files = [f for f in Path(CLEAN_PATH).iterdir() if f.suffix == '.npz']
    print('adding noise to clean data')
    thread_map(add_noise, files)
    print('done')
