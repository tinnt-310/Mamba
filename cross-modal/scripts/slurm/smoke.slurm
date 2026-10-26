#!/bin/bash
#SBATCH --job-name=mamba-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/tinnt/mamba/cross-modal/runs/slurm_logs/smoke-%j.out
#SBATCH --error=/home/tinnt/mamba/cross-modal/runs/slurm_logs/smoke-%j.err
set -euo pipefail
cd /home/tinnt/mamba/cross-modal
export LD_LIBRARY_PATH="/opt/hpc/miniforge3/envs/ai-pytorch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="/tmp/mamba-pycache-${SLURM_JOB_ID}"
mamba_python=/opt/hpc/miniforge3/envs/ai-pytorch/bin/python
mamba_smoke_dir="runs/smoke5000_${SLURM_JOB_ID}"
srun "$mamba_python" -m models.keypoints.experiment ablate --device cuda \
  --global-epochs 1 --local-epochs 1 --joint-epochs 1 --max-samples 1 \
  --output "$mamba_smoke_dir" "$@"
srun "$mamba_python" -m models.keypoints.experiment infer --device cuda \
  --checkpoint "$mamba_smoke_dir/A4/joint/best.pt" --split val --max-samples 1 \
  --output "$mamba_smoke_dir/infer"
mamba_artifact=$(find "$mamba_smoke_dir/infer/artifacts" -maxdepth 1 -name '*.npz' -print -quit)
srun "$mamba_python" -m models.keypoints.experiment visualize \
  --artifact "$mamba_artifact" --output "$mamba_smoke_dir/preview.png"
