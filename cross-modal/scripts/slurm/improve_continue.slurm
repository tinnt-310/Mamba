#!/bin/bash
#SBATCH --job-name=mb-next-control
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --export=NONE
set -euo pipefail
mamba_campaign_root="$1"
mamba_role="$2"
mamba_source_root="${5:-$mamba_campaign_root/continuation/source}"
cd "$mamba_source_root"
source scripts/slurm/cuda_scan_env.sh
export PYTHONPATH="$mamba_source_root"
export OPENBLAS_NUM_THREADS=1
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE SRUN_CPUS_PER_TASK
if [[ "$mamba_role" == architecture-initialize || "$mamba_role" == architecture-controller ]]; then
  export LD_LIBRARY_PATH=/opt/hpc/miniforge3/envs/ai-pytorch/lib
  exec srun --cpu-bind=cores /opt/hpc/miniforge3/envs/ai-pytorch/bin/python -m models.keypoints.architecture_search "${mamba_role#architecture-}" --root "$mamba_campaign_root"
fi
if [[ "$mamba_role" == architecture-worker ]]; then
  export MAMBA_JOB_SECONDS="$4"
  exec srun --cpu-bind=cores "$mamba_python" -m models.keypoints.architecture_search worker --root "$mamba_campaign_root" --task "$3"
fi
if [[ "$mamba_role" == tests ]]; then
  export LD_LIBRARY_PATH=/opt/hpc/miniforge3/envs/ai-pytorch/lib
  exec srun --cpu-bind=cores /opt/hpc/miniforge3/envs/ai-pytorch/bin/python -m pytest -q --junitxml="$mamba_campaign_root/continuation/correctness.xml"
fi
if [[ "$mamba_role" == worker ]]; then
  export MAMBA_JOB_SECONDS="$4"
  exec srun --cpu-bind=cores "$mamba_python" -m models.keypoints.continuation worker --root "$mamba_campaign_root" --task "$3"
fi
exec srun --cpu-bind=cores "$mamba_python" -m models.keypoints.continuation controller --root "$mamba_campaign_root"
