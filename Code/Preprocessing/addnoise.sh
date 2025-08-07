#!/bin/bash

mv slurm_logs/addnoise/* prev_slurm_logs

module load anaconda
conda activate myenv

srun \
    --job-name=addnoise_TROPOMI_data \
    --output=slurm_logs/addnoise/output_%j.txt \
    --error=slurm_logs/addnoise/error_%j.txt \
    --time=03:00:00 \
    --mem=48G \
    --cpus-per-task=10 \
    python add_noise.py
