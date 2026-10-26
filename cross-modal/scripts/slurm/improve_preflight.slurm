#!/bin/bash
#SBATCH --job-name=mb-preflight
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/tinnt/mamba/cross-modal/runs/improve_20261025/preflight-%j.out
#SBATCH --error=/home/tinnt/mamba/cross-modal/runs/improve_20261025/preflight-%j.err
set -euo pipefail
cd /home/tinnt/mamba/cross-modal
source scripts/slurm/cuda_scan_env.sh
exec srun "$mamba_python" scripts/improve_preflight.py /home/tinnt/mamba/cross-modal/runs/improve_20261025
