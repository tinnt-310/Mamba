#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --export=NONE
set -euo pipefail
mamba_campaign_root="$1"
mamba_role="$2"
mamba_task="$3"
mamba_seconds="$4"
mamba_source_root="$5"
cd "$mamba_source_root"
source scripts/slurm/cuda_scan_env.sh
export PYTHONPATH="$mamba_source_root"
export MAMBA_JOB_SECONDS="$mamba_seconds"
export OPENBLAS_NUM_THREADS=1
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE SRUN_CPUS_PER_TASK
if [[ "$mamba_role" == controller || "$mamba_role" == initialize || "$mamba_role" == intervals ]]; then
  export LD_LIBRARY_PATH=/opt/hpc/miniforge3/envs/ai-pytorch/lib
  exec srun --cpu-bind=cores /opt/hpc/miniforge3/envs/ai-pytorch/bin/python -m models.keypoints.gdl_search "$mamba_role" --root "$mamba_campaign_root"
fi
if [[ "$mamba_role" == tests ]]; then
  export LD_LIBRARY_PATH=/opt/hpc/miniforge3/envs/ai-pytorch/lib
  exec srun --cpu-bind=cores /opt/hpc/miniforge3/envs/ai-pytorch/bin/python -m pytest -q --junitxml="$mamba_campaign_root/architecture_gdl/correctness.xml"
fi
if [[ "$mamba_role" == worker && "${6:-gpu}" == cpu ]]; then
  mamba_python=/opt/hpc/miniforge3/envs/ai-pytorch/bin/python
  export LD_LIBRARY_PATH=/opt/hpc/miniforge3/envs/ai-pytorch/lib
fi
exec srun --cpu-bind=cores "$mamba_python" -m models.keypoints.gdl_search worker --root "$mamba_campaign_root" --task "$mamba_task"
