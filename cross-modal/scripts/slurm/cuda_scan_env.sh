#!/bin/bash
# Use the exact native CUDA extension that passed audit 2585.
export MAMBA_CUDA_ENV=/home/tinnt/mamba/cross-modal/runs/cuda_scan_audit/2585/environment
export PATH="$MAMBA_CUDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$MAMBA_CUDA_ENV/lib"
export CUDA_HOME="$MAMBA_CUDA_ENV/lib/python3.13/site-packages/nvidia/cu13"
export PYTHONPATH=/home/tinnt/mamba/cross-modal
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SLURM_EXPORT_ENV=ALL
export PYTHONPYCACHEPREFIX="/tmp/mamba-cuda-pycache-${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="/tmp/mamba-cuda-triton-${SLURM_JOB_ID}"
unset MAMBA_STOP_AT MAMBA_BUDGET_SECONDS
mamba_python="$MAMBA_CUDA_ENV/bin/python"
