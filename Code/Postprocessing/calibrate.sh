#!/bin/bash

mv slurm_logs/normalize/* prev_slurm_logs

module load anaconda
conda activate myenv

srun \
    --job-name=normalize_TROPOMI_data \
    --output=slurm_logs/normalize/output_%j.txt \
    --error=slurm_logs/normalize/error_%j.txt \
    --time=08:00:00 \
    --mem=128G \
    --cpus-per-task=16 \
    python normalize.py
