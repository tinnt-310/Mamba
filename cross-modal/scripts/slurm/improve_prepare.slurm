#!/bin/bash
#SBATCH --job-name=mb-prepare
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/home/tinnt/mamba/cross-modal/runs/improve_20261025/prepare-%j.out
#SBATCH --error=/home/tinnt/mamba/cross-modal/runs/improve_20261025/prepare-%j.err
set -euo pipefail
cd /home/tinnt/mamba/cross-modal
export LD_LIBRARY_PATH=/opt/hpc/miniforge3/envs/ai-pytorch/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec srun /opt/hpc/miniforge3/envs/ai-pytorch/bin/python -m models.keypoints.multisource \
  --output runs/improve_20261025/manifest50k.json
