#!/bin/bash

module load anaconda
conda activate torch

mv slurm_logs/denoise/* prev_slurm_logs

srun \
    --job-name=train_ddunet \
    --output=slurm_logs/denoise/output_%j.txt \
    --error=slurm_logs/denoise/error_%j.txt \
    --time=24:00:00 \
    --mem=160G \
    --cpus-per-task=16 \
    --gpus-per-node=2 \
    python train_ddunet.py
