#!/bin/bash
#SBATCH --job-name=mamba-subset
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/tinnt/mamba/cross-modal/runs/slurm_logs/prepare-%j.out
#SBATCH --error=/home/tinnt/mamba/cross-modal/runs/slurm_logs/prepare-%j.err
set -euo pipefail
cd /home/tinnt/mamba/cross-modal
export LD_LIBRARY_PATH="/opt/hpc/miniforge3/envs/ai-pytorch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="/tmp/mamba-pycache-${SLURM_JOB_ID}"
exec srun /opt/hpc/miniforge3/envs/ai-pytorch/bin/python -m models.keypoints.experiment prepare --limit 5000 "$@"
