#!/bin/bash
#SBATCH --job-name=mamba-cuda-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --exclude=hpc01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/tinnt/mamba/cross-modal/runs/cuda_scan_audit/slurm-%j.out
#SBATCH --error=/home/tinnt/mamba/cross-modal/runs/cuda_scan_audit/slurm-%j.err
set -euo pipefail
cd /home/tinnt/mamba/cross-modal
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export SLURM_EXPORT_ENV=ALL
# The user's login profile points to a proxy listening only on the login host.
# It is not reachable at localhost on a separately allocated compute node.
for audit_proxy_name in HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; do
  if [[ "${!audit_proxy_name:-}" == "http://127.0.0.1:18081" ]]; then
    unset "$audit_proxy_name"
  fi
done
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="/tmp/cuda-scan-audit-${SLURM_JOB_ID}"
export LD_LIBRARY_PATH="/opt/hpc/miniforge3/envs/ai-pytorch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec srun /opt/hpc/miniforge3/envs/ai-pytorch/bin/python scripts/cuda_scan_audit/run_audit.py
