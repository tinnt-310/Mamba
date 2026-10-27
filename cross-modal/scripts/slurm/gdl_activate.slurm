#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --export=NONE
set -euo pipefail
cd /home/tinnt/mamba/cross-modal
export LD_LIBRARY_PATH=/opt/hpc/miniforge3/envs/ai-pytorch/lib
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SLURM_EXPORT_ENV=ALL
export PYTHONPYCACHEPREFIX="/tmp/mamba-gdl-activation-${SLURM_JOB_ID}"
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE SRUN_CPUS_PER_TASK
exec srun --cpu-bind=cores /opt/hpc/miniforge3/envs/ai-pytorch/bin/python scripts/gdl_activate_automation.py "$@"
