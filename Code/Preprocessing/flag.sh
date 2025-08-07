#!/bin/bash

mv slurm_logs/flag/* prev_slurm_logs

module load anaconda
conda activate myenv

srun \
    --job-name=flag_TROPOMI_data \
    --output=slurm_logs/flag/output_%j.txt \
    --error=slurm_logs/flag/error_%j.txt \
    --time=03:00:00 \
    --mem=48G \
    --cpus-per-task=16 \
    python flag_data.py
