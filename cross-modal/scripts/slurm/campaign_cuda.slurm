#!/bin/bash
#SBATCH --job-name=mamba-cuda-A0A4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --exclude=hpc01
#SBATCH --export=NONE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/tinnt/mamba/cross-modal/runs/slurm_logs/campaign-cuda-%j.out
#SBATCH --error=/home/tinnt/mamba/cross-modal/runs/slurm_logs/campaign-cuda-%j.err
set -euo pipefail
cd /home/tinnt/mamba/cross-modal
source scripts/slurm/cuda_scan_env.sh
srun "$mamba_python" scripts/slurm/verify_cuda_scan.py
mamba_campaign_out=runs/ablation5000_cuda_20261005
mamba_args=("$@")
mamba_data_args=()
for ((mamba_i=0;mamba_i<${#mamba_args[@]};mamba_i++)); do
  case "${mamba_args[mamba_i]}" in
    --output)
      mamba_campaign_out="${mamba_args[mamba_i+1]}"; mamba_i=$((mamba_i+1));;
    --output=*) mamba_campaign_out="${mamba_args[mamba_i]#--output=}";;
    --data-root|--manifest|--decoder)
      mamba_data_args+=("${mamba_args[mamba_i]}" "${mamba_args[mamba_i+1]}"); mamba_i=$((mamba_i+1));;
    --data-root=*|--manifest=*|--decoder=*) mamba_data_args+=("${mamba_args[mamba_i]}");;
  esac
done
# Use elapsed allocation duration, avoiding observed controller/node clock skew.
mamba_job_info=$(scontrol show job --oneliner "$SLURM_JOB_ID")
if [[ "$mamba_job_info" =~ TimeLimit=([^[:space:]]+) ]]; then
  mamba_limit="${BASH_REMATCH[1]}"
else
  echo 'Unable to read allocation time limit; refusing unmanaged long campaign' >&2
  exit 1
fi
mamba_days=0
if [[ "$mamba_limit" == *-* ]]; then
  mamba_days="${mamba_limit%%-*}"; mamba_limit="${mamba_limit#*-}"
fi
IFS=: read -r mamba_hours mamba_minutes mamba_seconds <<< "$mamba_limit"
mamba_remaining=$((10#$mamba_days*86400+10#$mamba_hours*3600+10#$mamba_minutes*60+10#$mamba_seconds))
mamba_margin=900
if ((mamba_remaining<1800)); then
  mamba_margin=$((mamba_remaining/10))
  if ((mamba_margin<20)); then mamba_margin=20; fi
fi
unset MAMBA_STOP_AT
export MAMBA_BUDGET_SECONDS=$((mamba_remaining-mamba_margin))
printf '%s\t%s\t%s\t%s\n' "$SLURM_JOB_ID" "${MAMBA_PARENT_JOB_ID:-none}" "$(date -u +%FT%TZ)" "$mamba_campaign_out" >> runs/slurm_logs/campaign-chain.tsv
printf 'Campaign allocation %s, parent %s, output %s, soft budget %ss\n' "$SLURM_JOB_ID" "${MAMBA_PARENT_JOB_ID:-none}" "$mamba_campaign_out" "$MAMBA_BUDGET_SECONDS"
# Exit75 alone means controlled time-budget expiration; other failures stop.
set +e
srun "$mamba_python" -m models.keypoints.experiment ablate \
  --device cuda --backend mamba --global-epochs 200 --local-epochs 50 --joint-epochs 50 --output "$mamba_campaign_out" "$@"
mamba_status=$?
set -e
if ((mamba_status==75)); then
  mamba_successor=$(sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" \
    --export="ALL,MAMBA_PARENT_JOB_ID=${SLURM_JOB_ID}" scripts/slurm/campaign_cuda.slurm "$@")
  printf 'Budget reached; continuation job %s will resume %s\n' "$mamba_successor" "$mamba_campaign_out"
  printf '%s\t%s\t%s\n' "$SLURM_JOB_ID" "$mamba_successor" "$mamba_campaign_out" >> runs/slurm_logs/campaign-continuations.tsv
  exit 0
fi
if ((mamba_status!=0)); then exit "$mamba_status"; fi
# Complete the inference and saved-artifact visualization after all locked test runs.
unset MAMBA_STOP_AT MAMBA_BUDGET_SECONDS
srun "$mamba_python" -m models.keypoints.experiment infer --device cuda \
  --checkpoint "$mamba_campaign_out/A4/joint/best.pt" --split val --max-samples 1 \
  --output "$mamba_campaign_out/infer" "${mamba_data_args[@]}"
mamba_artifact=$(find "$mamba_campaign_out/infer/artifacts" -maxdepth 1 -name '*.npz' -print -quit)
srun "$mamba_python" -m models.keypoints.experiment visualize \
  --artifact "$mamba_artifact" --output "$mamba_campaign_out/preview.png"
printf '{"status":"complete","slurm_job_id":"%s","report":"report.json","preview":"preview.png"}\n' "$SLURM_JOB_ID" > "$mamba_campaign_out/end_to_end_complete.json"
printf 'End-to-end campaign complete: %s/report.json\n' "$mamba_campaign_out"
