# -----------------------------------------------------------------------------
# File          : dataset.py
# Description   : dataset class and helper methods
# Author        : Daniel G. Li
# -----------------------------------------------------------------------------

# imports
from random import shuffle, seed
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from matplotlib.colors import LogNorm
from tqdm.contrib.concurrent import thread_map


class Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, 0].unsqueeze(0), self.data[idx, 1].unsqueeze(0)


def split_dset(data_path, loadfunc):
    seed(67)
   
    files = [str(f) for f in sorted(Path(data_path).iterdir())
        if f.is_file() and f.suffix == '.npz']

    print('loading data...', flush=True)
    datas = thread_map(loadfunc, files)
    print('done loading data', flush=True)
    
    # train:valid:test = 8:1:1
    split1 = (8*len(datas))//10
    split2 = (9*len(datas))//10
    data_test = np.concatenate(datas[split2:], axis=0)
    datas = datas[:split2]
    shuffle(datas)    

    data_train = np.concatenate(datas[:split1], axis=0)
    data_valid = np.concatenate(datas[split1:], axis=0)

    data_train = torch.from_numpy(data_train)
    data_valid = torch.from_numpy(data_valid)
    data_test = torch.from_numpy(data_test)
    
    return data_train, data_valid, data_test
