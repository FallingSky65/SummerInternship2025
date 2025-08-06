# -----------------------------------------------------------------------------
# File          : train_utils.py
# Description   : helper methods and example for training
# Author        : Daniel G. Li
# -----------------------------------------------------------------------------

# imports
from dataset import Dataset, split_dset
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# paths
PROJ_ROOT = os.getenv('PROJ_ROOT')
DATA_PATH = f'{PROJ_ROOT}/Data/TROPOMI/S5P_L2__SO2____HiR'
CROPPED_PATH = f'{DATA_PATH}_cropped'
CLEAN_PATH = f'{DATA_PATH}_clean'
NOISY_PATH = f'{DATA_PATH}_noisy'
ADDED_NOISE_PATH = f'{DATA_PATH}_added_noise'
ADDED_FLAG_PATH = f'{DATA_PATH}_added_flag'

# parameters class
class Params:
    def __init__(self, Model, Criterion, Checkpoint, DATA_PATH, LOAD_FUNC,
            LEARNING_RATE, BATCH_SIZE, BATCH_CHUNKS, EPOCH_START, EPOCH_END,
            MASTER_PORT, SAVE_PATH):
        self.Model = Model
        self.Criterion = Criterion
        self.Checkpoint = Checkpoint
        self.DATA_PATH = DATA_PATH
        self.LOAD_FUNC = LOAD_FUNC
        self.LEARNING_RATE = LEARNING_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.BATCH_CHUNKS = BATCH_CHUNKS
        self.EPOCH_START = EPOCH_START # exclusive
        self.EPOCH_END = EPOCH_END # inclusive, epochs are 1-indexed
        self.MASTER_PORT = MASTER_PORT
        self.SAVE_PATH = SAVE_PATH


def setup(rank, world_size, p):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = p.MASTER_PORT
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def get_dataloaders(rank, world_size, data_train, data_valid, data_test, p):
    train_dset = Dataset(data_train.clone())
    valid_dset = Dataset(data_valid.clone())
    test_dset = Dataset(data_test.clone())

    if world_size > 1:
        train_sampler = DistributedSampler(train_dset, num_replicas=world_size,
                                       rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_dset, batch_size=p.BATCH_SIZE, sampler=train_sampler,
            num_workers=2, pin_memory=True)
        valid_loader = DataLoader(valid_dset, batch_size=p.BATCH_SIZE,
                shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dset, batch_size=p.BATCH_SIZE,
                shuffle=False, num_workers=2, pin_memory=True)
    else:
        train_sampler = None
        train_loader = DataLoader(train_dset, batch_size=p.BATCH_SIZE,
                pin_memory=True)
        valid_loader = DataLoader(valid_dset, batch_size=p.BATCH_SIZE,
                shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dset, batch_size=p.BATCH_SIZE,
                shuffle=False, pin_memory=True)

    return train_loader, train_sampler, valid_loader, test_loader


def evaluate(model, dataloader, device, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
    return test_loss / len(dataloader)


def train(rank, world_size, data_train, data_valid, data_test, p):
    if world_size > 1:
        print(f'running ddp training on rank {rank}.', flush=True)
        setup(rank, world_size, p)
    else:
        print('beginning training.')
    device = torch.device(f'cuda:{rank}')

    if p.Checkpoint is not None:
        checkpoint = torch.load(p.Checkpoint)
    model = p.Model()
    if p.Checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    criterion = p.Criterion().to(device)
    optimizer = optim.Adam(model.parameters(), lr=p.LEARNING_RATE)
    if p.Checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    train_loader, train_sampler, valid_loader, test_loader = get_dataloaders(
            rank, world_size, data_train, data_valid, data_test, p)

    for epoch in range(p.EPOCH_START, p.EPOCH_END):
        if rank == 0:
            print(f'[epoch {epoch+1}] training...', flush=True)
        model.train()
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        total_loss = 0.0

        optimizer.zero_grad()
        for step, (x, y) in tqdm(enumerate(train_loader)):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            loss = loss / p.BATCH_CHUNKS
            loss.backward()

            if (step + 1) % p.BATCH_CHUNKS == 0:
                optimizer.step()
                optimizer.zero_grad()

        if (step + 1) % p.BATCH_CHUNKS != 0:
            optimizer.step()
            optimizer.zero_grad()
        train_loss = total_loss / len(train_loader)

        if rank == 0:
            print(f'[epoch {epoch+1}] training loss: {train_loss:.6f}', flush=True)
            valid_loss = evaluate(model.module, valid_loader, device, criterion)
            test_loss = evaluate(model.module, test_loader, device, criterion)
            print(f'[epoch {epoch+1}] valid_loss: {valid_loss:.6f}', flush=True)
            print(f'[epoch {epoch+1}] test_loss: {test_loss:.6f}', flush=True)

            checkpoint = {
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch+1,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'test_loss': test_loss
            }
            torch.save(checkpoint, f'{p.SAVE_PATH}/checkpoint_{epoch+1:0>4}.pth')
            print('[epoch {epoch+1}] checkpoint saved', flush=True)

    if world_size > 1:
        dist.destroy_process_group()
    torch.cuda.empty_cache()


def spawn_train(p):
    torch.cuda.empty_cache()
    world_size = torch.cuda.device_count()
    
    data_train, data_valid, data_test = split_dset(p.DATA_PATH, p.LOAD_FUNC)
    print('data shapes:')
    print(f'\ttrain: {data_train.shape}')
    print(f'\tvalid: {data_valid.shape}')
    print(f'\ttest: {data_test.shape}')

    if world_size > 1:
        print(f'{world_size} gpus detected', flush=True)
        mp.spawn(train, args=(world_size, data_train, data_valid, data_test, p),
                nprocs=world_size, join=True)
    elif world_size == 1:
        print('one gpu detected', flush=True)
        train(0, 1, data_train, data_valid, data_test, p)
    else:
        print('no gpus detected', flush=True)
