#!/bin/bash
#SBATCH --job-name=mb-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/tinnt/mamba/cross-modal/runs/improve_20261025/baseline-%j.out
#SBATCH --error=/home/tinnt/mamba/cross-modal/runs/improve_20261025/baseline-%j.err
set -euo pipefail
cd /home/tinnt/mamba/cross-modal
source scripts/slurm/cuda_scan_env.sh
export PYTHONPATH=/home/tinnt/mamba/cross-modal/runs/improve_20261025/baseline_source
srun "$mamba_python" scripts/profile_baseline.py \
  --checkpoint runs/ablation5000_new50k_cuda_20261005/A1/global/best.pt \
  --manifest /work/tinnt/MAMBA_NEW50K_SUBSET5000_20261005/subset5000.json \
  --output runs/improve_20261025/baseline/A1_NEW
srun "$mamba_python" scripts/profile_baseline.py \
  --checkpoint runs/ablation5000_new50k_cuda_20261005/A4/joint/best.pt \
  --manifest /work/tinnt/MAMBA_NEW50K_SUBSET5000_20261005/subset5000.json \
  --output runs/improve_20261025/baseline/A4_NEW
