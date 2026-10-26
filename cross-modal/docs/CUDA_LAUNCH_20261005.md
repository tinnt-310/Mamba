# CUDA selective-scan campaign — 2026-10-05

This is a new A0–A4 campaign trained from scratch using the existing 5,000-image
manifest (4,000 train / 500 validation / 500 test), seed 20260922, FP32, and the
standard 200 global / 50 local / 50 joint epoch limits with early stopping.

- Audit prerequisite: `runs/cuda_scan_audit/2585/report.json`, all gates PASS.
- Runtime: `runs/cuda_scan_audit/2585/environment/bin/python`.
- Both `global_ssm.backend` and `local.backend`: `mamba`.
- GPU: one NVIDIA RTX A5000, partition `gpu`, excluding hpc01.
- Benchmark + one-sample A0–A4 smoke job: **2660**.
- Full campaign job: **2661**, dependency `afterok:2660`.
- Output: `runs/ablation5000_cuda_20261005`.
- Existing torch campaign job 2082 keeps its own output and runtime.

All Python imports, CUDA checks, benchmarks, training, inference and visualization
run via `srun` on Slurm compute nodes, as required by `/home/tinnt/slurm.md`.
Each allocation first checks the audited extension checksum and executes native
CUDA forward/backward for state dimensions 4 and 16. Evidence is saved in
`runs/slurm_logs/cuda-backend-JOBID.json`. The smoke also benchmarks the full
model, trains all A0–A4 stages, reloads checkpoints, performs inference, and
renders a preview. The full campaign cannot start if the smoke fails.

Monitor:

```bash
squeue -j 2660,2661
tail -f runs/slurm_logs/campaign-cuda-2661.out
```

The campaign uses a seven-day allocation limit and resumes automatically after
controlled time-budget expiration. Successor jobs use `campaign_cuda.slurm` and
the same backend, environment, arguments and output. Ordinary errors stop the
chain. Completion requires `runs/ablation5000_cuda_20261005/end_to_end_complete.json`.

Resume after an ordinary interruption (only when no allocation is using this output):

```bash
sbatch scripts/slurm/campaign_cuda.slurm --output runs/ablation5000_cuda_20261005
```

Create a separate output for a different seed, backend, or epoch budget; the
campaign protocol is immutable. Do not point CUDA jobs at `runs/ablation5000`,
whose protocol uses the torch backend.

Verified startup: smoke 2660 completed with exit 0:0 in 55 seconds; full
campaign 2661 started on hpc02. All five smoke model configurations use mamba
for both global and local scan, with finite benchmark gradients and a saved
inference preview. The full campaign protocol confirms max_samples=null.
