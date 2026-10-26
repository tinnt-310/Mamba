#!/bin/bash
#SBATCH --job-name=mb-controller
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/tinnt/mamba/cross-modal/runs/improve_20261025/controller-%j.out
#SBATCH --error=/home/tinnt/mamba/cross-modal/runs/improve_20261025/controller-%j.err
set -euo pipefail
mamba_campaign_root=/home/tinnt/mamba/cross-modal/runs/improve_20261025
cd "$mamba_campaign_root/source"
export PYTHONPATH="$mamba_campaign_root/source"
exec srun /opt/hpc/miniforge3/envs/ai-pytorch/bin/python -m models.keypoints.campaign controller \
  --root "$mamba_campaign_root" --baseline-job 5705
