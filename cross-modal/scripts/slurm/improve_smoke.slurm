#!/bin/bash
#SBATCH --job-name=mb-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/tinnt/mamba/cross-modal/runs/improve_20261025/smoke-%j.out
#SBATCH --error=/home/tinnt/mamba/cross-modal/runs/improve_20261025/smoke-%j.err
set -euo pipefail
cd /home/tinnt/mamba/cross-modal
source scripts/slurm/cuda_scan_env.sh
export MAMBA_JOB_SECONDS=1200
srun "$mamba_python" -m models.keypoints.trial_runner --campaign runs/improve_20261025 --task runs/improve_20261025/smoke_global.json
srun "$mamba_python" -m models.keypoints.trial_runner --campaign runs/improve_20261025 --task runs/improve_20261025/smoke_local.json
