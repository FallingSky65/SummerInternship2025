#!/bin/bash

mv slurm_logs/sort/* prev_slurm_logs

module load anaconda
conda activate myenv

srun \
    --job-name=sort_TROPOMI_data \
    --output=slurm_logs/sort/output_%j.txt \
    --error=slurm_logs/sort/error_%j.txt \
    --time=03:00:00 \
    --mem=48G \
    --cpus-per-task=16 \
    python sort_data.py
