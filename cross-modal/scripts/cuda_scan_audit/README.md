# Independent CUDA selective-scan audit

From `/home/tinnt/mamba/cross-modal`, stage files and submit:

```bash
bash scripts/cuda_scan_audit/stage_downloads.sh
sbatch scripts/slurm/cuda_scan_audit.slurm
```

The staging script only transfers official source archives, wheels and cached
Conda archives. It does not install packages, extract archives, import torch or
run numerical work. This is needed because compute nodes cannot resolve public
Internet domains, and the user's login proxy is bound to login-host localhost.
The Slurm script removes that specific localhost proxy from its own environment.

All checksum verification, environment cloning, package installation, extension
compilation, correctness tests, validation and benchmarks run in the dedicated
Slurm allocation. The job requests one RTX A5000, eight CPUs, 64 GiB RAM and six
hours, excluding hpc01. Neither the source environment nor job 2082 is modified.

Each allocation creates `runs/cuda_scan_audit/<job_id>/` containing an immutable
source/checkpoint snapshot, a private environment, detailed logs, raw timing
samples, JSON reports and `RESULTS.md`. Inspect the phase with:

```bash
jq '{status,phase,CUDA_USABLE,failures,gates}' runs/cuda_scan_audit/<job_id>/report.json
```

`CUDA_USABLE` requires native CUDA forward/backward evidence, FP32 output and
gradient checks, A1 validation equivalence on all 500 validation images, and at
least 2x faster mean full inference and full training in every one of three
rounds for both A1 and controlled A4. Controlled A4 uses A1 global weights and
untrained deterministic local weights. Its results do not establish accuracy
or latency of a trained A4 model.

The audit runs the two backends in separate processes on the same allocated GPU.
It changes backend selection only in RAM and never migrates checkpoints, changes
the campaign protocol, or submits a training campaign. A failure or timeout
leaves `CUDA_USABLE=false`; partial benchmark files cannot produce a pass.

## Setup attempts retained for provenance

- 2570/2578: clone stopped at inherited localhost proxy configuration.
- 2579: direct Internet DNS unavailable on compute node.
- 2581: offline clone could not find caches because the Conda sequence variable
  used an incorrect separator; corrected to comma and recorded in the cache log.
- 2585: offline clone and CUDA 13.0 toolkit installation succeeded; preserves
  Python 3.13.13, PyTorch 2.11.0+cu130 and C++11 ABI. Its report is the source of
  truth for subsequent build and test outcomes.

These setup failures do not demonstrate numerical incompatibility of the CUDA
backend. Consult the final report rather than interpreting a Slurm exit code
alone as a correctness or performance result.
