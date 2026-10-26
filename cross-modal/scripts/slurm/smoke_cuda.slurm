#!/bin/bash
#SBATCH --job-name=mamba-cuda-smk
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --exclude=hpc01
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/tinnt/mamba/cross-modal/runs/slurm_logs/smoke-cuda-%j.out
#SBATCH --error=/home/tinnt/mamba/cross-modal/runs/slurm_logs/smoke-cuda-%j.err
set -euo pipefail
cd /home/tinnt/mamba/cross-modal
source scripts/slurm/cuda_scan_env.sh
srun "$mamba_python" scripts/slurm/verify_cuda_scan.py
mamba_smoke_dir="runs/smoke5000_cuda_${SLURM_JOB_ID}"
srun "$mamba_python" -m models.keypoints.experiment benchmark --device cuda \
  --backend mamba --images 1 --output "runs/slurm_logs/benchmark-cuda-${SLURM_JOB_ID}.json"
srun "$mamba_python" -m models.keypoints.experiment ablate --device cuda --backend mamba \
  --global-epochs 1 --local-epochs 1 --joint-epochs 1 --max-samples 1 \
  --output "$mamba_smoke_dir" "$@"
srun "$mamba_python" -m models.keypoints.experiment infer --device cuda \
  --checkpoint "$mamba_smoke_dir/A4/joint/best.pt" --split val --max-samples 1 \
  --output "$mamba_smoke_dir/infer"
mamba_artifact=$(find "$mamba_smoke_dir/infer/artifacts" -maxdepth 1 -name '*.npz' -print -quit)
srun "$mamba_python" -m models.keypoints.experiment visualize \
  --artifact "$mamba_artifact" --output "$mamba_smoke_dir/preview.png"
