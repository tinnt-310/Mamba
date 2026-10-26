#!/bin/bash
#SBATCH --job-name=mb-validate
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/tinnt/mamba/cross-modal/runs/improve_20261025/validate-%j.out
#SBATCH --error=/home/tinnt/mamba/cross-modal/runs/improve_20261025/validate-%j.err
set -euo pipefail
mamba_campaign_root=/home/tinnt/mamba/cross-modal/runs/improve_20261025
cd "$mamba_campaign_root/source"
export LD_LIBRARY_PATH=/opt/hpc/miniforge3/envs/ai-pytorch/lib
export PYTHONPATH="$mamba_campaign_root/source"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun /opt/hpc/miniforge3/envs/ai-pytorch/bin/python -m pytest -q \
  --junitxml="$mamba_campaign_root/correctness.xml"
srun /opt/hpc/miniforge3/envs/ai-pytorch/bin/python scripts/seal_preflight.py "$mamba_campaign_root"
