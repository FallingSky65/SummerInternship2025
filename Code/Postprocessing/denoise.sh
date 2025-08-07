#!/bin/bash

mv slurm_logs/denoise/* prev_slurm_logs

module load anaconda
conda activate torch

srun \
    --job-name=denoise_TROPOMI_data \
    --output=slurm_logs/denoise/output_%j.txt \
    --error=slurm_logs/denoise/error_%j.txt \
    --time=06:00:00 \
    --mem=64G \
    --cpus-per-task=16 \
    --gpus-per-node=1 \
    python denoise.py
