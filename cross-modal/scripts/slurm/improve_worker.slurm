#!/bin/bash
#SBATCH --job-name=mb-trial
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --export=NONE
set -euo pipefail
mamba_campaign_root="$1"
mamba_task="$2"
mamba_seconds="$3"
cd "$mamba_campaign_root/source"
source scripts/slurm/cuda_scan_env.sh
export PYTHONPATH="$mamba_campaign_root/source"
export MAMBA_JOB_SECONDS="$mamba_seconds"
export OPENBLAS_NUM_THREADS=1
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE SRUN_CPUS_PER_TASK
exec srun --cpus-per-task=8 --cpu-bind=cores "$mamba_python" -m models.keypoints.trial_runner --campaign "$mamba_campaign_root" --task "$mamba_task"
