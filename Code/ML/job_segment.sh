#!/bin/bash

module load anaconda
conda activate torch

mv slurm_logs/segment/* prev_slurm_logs

srun \
    --job-name=train_unet \
    --output=slurm_logs/segment/output_%j.txt \
    --error=slurm_logs/segment/error_%j.txt \
    --time=24:00:00 \
    --mem=160G \
    --cpus-per-task=16 \
    --gpus-per-node=2 \
    python train_unet.py
