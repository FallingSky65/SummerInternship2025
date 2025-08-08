# -----------------------------------------------------------------------------
# File          : evaluate.py
# Description   : evaluates model effectiveness
# Author        : Daniel G. Li
# -----------------------------------------------------------------------------

# imports
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
from os import getenv

load_dotenv()

# paths
PROJ_ROOT = getenv('PROJ_ROOT')
DATA_PATH = f'{PROJ_ROOT}/Data/TROPOMI/S5P_L2__SO2____HiR'
ADDED_NOISE_PATH = f'{DATA_PATH}_added_noise'
ADDED_FLAG_PATH = f'{DATA_PATH}_added_flag'

UNET_PATH = f'{PROJ_ROOT}/Models/segment/checkpoint_1600.pth'


import sys
sys.path.append(f'{PROJ_ROOT}/Code/ML')
from dataset import Dataset, split_dset
from unet import UNet
from train_unet import load_segment


def calc_IoU(preds, targets, threshold=0.5, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    I = (preds * targets).sum(dim=(1, 2, 3))
    U = (preds + targets).sum(dim=(1, 2, 3)) - I
    IoU = (I + smooth) / (U + smooth)
    return IoU.sum().item()


def calc_dice(preds, targets, threshold=0.5, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    
    TP = (preds * targets).sum(dim=(1, 2, 3))
    FP = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    FN = ((1 - preds) * targets).sum(dim=(1, 2, 3))

    dice = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
    return dice.sum().item()


def calc_conf_mat(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).int()
    targets = targets.int()
    
    TP = ((preds == 1) & (targets == 1)).sum().item()
    TN = ((preds == 0) & (targets == 0)).sum().item()
    FP = ((preds == 1) & (targets == 0)).sum().item()
    FN = ((preds == 0) & (targets == 1)).sum().item()

    return np.array([[TP, FN], [FP, TN]], dtype=np.uint64)


def eval_unet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    print(f'loading model from {UNET_PATH}')
    model = UNet()
    checkpoint = torch.load(UNET_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    _, _, data = split_dset(ADDED_FLAG_PATH, load_segment)
    dset = Dataset(data)
    dataloader = DataLoader(dset, batch_size=64)    

    model.eval()
    total_iou, total_dice = 0, 0
    conf_mat = np.zeros((2, 2), dtype=np.uint64)
    n_imgs = 0

    print('evaluating model')
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            outs = model(x)

            total_iou += calc_IoU(outs, y)
            total_dice += calc_dice(outs, y)
            conf_mat += calc_conf_mat(outs, y)
            n_imgs += x.shape[0]

    avg_iou = total_iou / n_imgs
    avg_dice = total_dice / n_imgs
    return avg_iou, avg_dice, conf_mat


if __name__ == '__main__':
    avg_iou, avg_dice, conf_mat = eval_unet()
    print(f'average iou: {avg_iou}')
    print(f'average dice: {avg_dice}')
    print('confusion matrix:')
    print(f'\tTP: {conf_mat[0, 0]}')
    print(f'\tFN: {conf_mat[0, 1]}')
    print(f'\tFP: {conf_mat[1, 0]}')
    print(f'\tTN: {conf_mat[1, 1]}')
