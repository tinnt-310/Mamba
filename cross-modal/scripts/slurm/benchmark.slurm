#!/bin/bash
#SBATCH --job-name=mamba-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/tinnt/mamba/cross-modal/runs/slurm_logs/benchmark-%j.out
#SBATCH --error=/home/tinnt/mamba/cross-modal/runs/slurm_logs/benchmark-%j.err
set -euo pipefail
cd /home/tinnt/mamba/cross-modal
export LD_LIBRARY_PATH="/opt/hpc/miniforge3/envs/ai-pytorch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="/tmp/mamba-pycache-${SLURM_JOB_ID}"
exec srun /opt/hpc/miniforge3/envs/ai-pytorch/bin/python -m models.keypoints.experiment benchmark --device cuda --images 1 "$@"
