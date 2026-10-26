#!/bin/bash
# File transfers only. Installation, extraction, builds and tests run under Slurm.
set -euo pipefail
audit_project=/home/tinnt/mamba/cross-modal
audit_stage="$audit_project/runs/cuda_scan_audit/offline-inputs-20261005"
audit_base=/opt/hpc/miniforge3/envs/ai-pytorch
mkdir -p "$audit_stage/conda-cache" "$audit_stage/wheels" "$audit_stage/metadata"
audit_fetch() {
  local audit_url="$1" audit_target="$2"
  if [[ -s "$audit_target" ]]; then return; fi
  curl --fail --location --retry 2 --connect-timeout 20 --max-time 900 --silent --show-error \
    "$audit_url" --output "$audit_target.part"
  mv "$audit_target.part" "$audit_target"
}
export -f audit_fetch
jq -r 'select(.url != null) | [.url,.fn,.sha256,.md5] | @tsv' \
  "$audit_base"/conda-meta/*.json > "$audit_stage/metadata/conda-packages.tsv"
while IFS=$'\t' read -r audit_url audit_filename audit_sha audit_md5; do
  audit_folder="${audit_filename%.conda}"
  audit_folder="${audit_folder%.tar.bz2}"
  if [[ -s "/opt/hpc/miniforge3/pkgs/$audit_filename" ]]; then
    if [[ ! -s "$audit_stage/conda-cache/$audit_filename" ]]; then
      cp --reflink=auto -- "/opt/hpc/miniforge3/pkgs/$audit_filename" "$audit_stage/conda-cache/$audit_filename.part"
      mv "$audit_stage/conda-cache/$audit_filename.part" "$audit_stage/conda-cache/$audit_filename"
    fi
  else
    printf '%s\0%s\0' "$audit_url" "$audit_stage/conda-cache/$audit_filename"
  fi
done < "$audit_stage/metadata/conda-packages.tsv" |
  xargs -0 -r -n 2 -P 6 bash -c 'audit_fetch "$1" "$2"' _
cut -f 1 "$audit_stage/metadata/conda-packages.tsv" > "$audit_stage/conda-cache/urls.txt"
if [[ "${1:-}" == '--conda-only' ]]; then exit 0; fi
for audit_spec in 'nvidia-cuda-nvcc/13.0.88' 'nvidia-cuda-crt/13.0.88' \
                  'nvidia-cuda-cccl/13.0.85' 'nvidia-nvvm/13.0.88' \
                  'ninja/1.11.1.4' 'wheel/0.45.1'; do
  audit_meta="$audit_stage/metadata/${audit_spec//\//-}.json"
  audit_fetch "https://pypi.org/pypi/$audit_spec/json" "$audit_meta"
  jq -r '.urls[] | select(.packagetype == "bdist_wheel") |
    select((.filename | endswith("none-any.whl")) or
           ((.filename | contains("manylinux")) and (.filename | endswith("x86_64.whl")))) |
    [.url,.filename] | @tsv' "$audit_meta" |
  while IFS=$'\t' read -r audit_url audit_filename; do
    audit_fetch "$audit_url" "$audit_stage/wheels/$audit_filename"
  done
done
audit_fetch 'https://codeload.github.com/state-spaces/mamba/tar.gz/refs/tags/v2.3.2.post1' \
  "$audit_stage/mamba-v2.3.2.post1.tar.gz"
audit_wheel='mamba_ssm-2.3.2.post1+cu13torch2.11cxx11abiTRUE-cp313-cp313-linux_x86_64.whl'
audit_wheel_url="https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/$audit_wheel"
audit_http_code=$(curl --location --retry 2 --connect-timeout 20 --max-time 180 --silent --show-error \
  --write-out '%{http_code}' "$audit_wheel_url" --output "$audit_stage/wheels/$audit_wheel.part")
case "$audit_http_code" in
  200) mv "$audit_stage/wheels/$audit_wheel.part" "$audit_stage/wheels/$audit_wheel" ;;
  404) rm -f "$audit_stage/wheels/$audit_wheel.part" ;;
  *) printf 'Unexpected wheel HTTP status: %s\n' "$audit_http_code" >&2; exit 1 ;;
esac
jq -n --arg url "$audit_wheel_url" --arg status "$audit_http_code" \
  '{url:$url,http_status:$status}' > "$audit_stage/metadata/mamba-wheel-probe.json"
printf 'Downloads staged at %s\n' "$audit_stage"
