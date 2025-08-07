#!/bin/bash

mv slurm_logs/crop/* prev_slurm_logs

module load anaconda
conda activate myenv

srun \
    --job-name=crop_TROPOMI_data \
    --output=slurm_logs/crop/output_%j.txt \
    --error=slurm_logs/crop/error_%j.txt \
    --time=03:00:00 \
    --mem=48G \
    --cpus-per-task=16 \
    python crop_data.py
