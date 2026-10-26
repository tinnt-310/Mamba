# RGB-D four-scale + adaptive local Mamba (5,000 frames)

This checkout now supplies the keypoint package that its original README referenced but did not contain. The global decoder implements the handed-off projection → resize → mean → two Conv/GN/GELU blocks → nine-channel head. Existing geometry encoders, RGB-D compensation, selective scan and inverse scan are reused.

## Dataset and coordinate conventions

Source: `/work/tinnt/BlenderProc/output/keypoints_bbox_50000_job_1634` (metadata declares 50,000 frames). `runs/subset5000.json` fixes 4,000 training, 500 validation, 500 test IDs with seed 20260922, sampled inside the corresponding source split. Source split fingerprints and dataset root are checked at reload. The remaining45,000 frames are never used for training/evaluation. No source images are copied or modified.

RGB comes from `rgb/<ID>.png`; depth is metric optical-axis Z from `depth_sensor/<ID>.npy`; intrinsics come from `poses/<ID>.json`; annotations join by keypoint semantic name. Visibility1/2 is valid when finite and in the original pixel footprint; visibility0/missing labels receive weight0. RGB receives ImageNet channel normalization. Depth uses nearest resampling and retains meters/invalid depth for the existing geometry validity mask.

Integer image coordinates locate pixel centers. Resize points with `(p+.5)*scale-.5`; resize K with the same affine, including its half-pixel principal-point translation. Boxes are integer half-open edges; patch membership uses `[x1-.5,x2-.5) × [y1-.5,y2-.5)`.

## Architecture and losses

Default input180×320; selected backbone stages(1,2,3,4); fused shapes32×45×80,32×23×40,32×12×20,32×6×10. The global heatmaps are9×90×160. `forward(rgb,depth,K)` returns this global tensor. `forward_stages(...,return_patches=False)` returns global plus local stages from one encoder/fusion pass; `refine_cached` serves the frozen-global phase.

Regions detach global heatmaps, reject nonfinite responses, use a min-max threshold.30 and four-connected component containing the row-major argmax, then margin.25 per side. Constant heatmaps use the full grid. Outward edge scaling maps boxes to actual F1 dimensions. Corners plus center form at most five deduplicated patches with overlap.50. Patches are sliced at native size; equal sizes are grouped with microbatch at most8, without resize/padding/ROIAlign.

One local selective SSM(state16) shares weights across four scan directions, patches and keypoints, followed by channel LayerNorm,1×1 projection/residual and1×1 heatmap head. Optional embedding identifies keypoints. Global compensation stays state4/expand1/direction1. The default backend is `torch`, with no automatic accelerated-backend substitution.

Patch predictions are placed at their F1 boxes and averaged using explicit integer coverage; support is separate. Masked argmax cannot select uncovered zero pixels when valid responses are negative. Subpixel offsets use a three-point log parabola independently per axis only for supported positive responses, negative curvature and finite offsets ≤.5; otherwise that axis falls back to argmax.

Targets use annotations, never predictions. Global sigma2 corresponds to local sigma1. Pixel weights are `1+(positive_weight-1)*target`; patches excluding GT receive zero targets. Local MSE averages pixels, then patches of a keypoint, then valid keypoints. Joint loss is global plus local with lambda1. Missing labels produce finite graph-connected zero loss.

## Training and experiment protocol

Defaults: float32, batch1, accumulation8, AdamW/weight decay1e-4 with `_no_weight_decay` exemptions, gradient clipping5, positive weight100. The last partial accumulation window is normalized by its actual sample count. Validation runs at first/last epoch, every5 epochs and the potential early-stop epoch. Checkpoints select validation mean original-image pixel error, breaking ties byP95.

| Phase | Learned modules | Max epochs | LR | Early-stop patience |
|---|---|---:|---:|---:|
|global|RGB/geometry encoders, fusion, global decoder/head|200|3e-4|20 epochs|
|local|local SSM/head/embedding|50|3e-4|10 epochs|
|joint|all modules|50|1e-4|10 epochs|

Local phase freezes global parameters and keeps their modules in eval after every train-mode switch. F1/global maps are cached on CPU with an8GiB LRU budget. Joint phase creates a new optimizer and releases the frozen-feature cache. Global phase uses global validation predictions even if an untrained local module is present.

A0 uses the final two selected stages(default3,4), global only. A1 uses all four, global only. Both train from scratch. A2 uses one whole-region patch without embedding; A3 uses five patches without embedding; A4 adds embedding. A2–A4 start from the same best A1 and identical common local weights; embedding initialization has an independent seed. Region tuning uses validation only: default.30/.25; if ROI recall <95%, search thresholds.15/.30/.50 and margins.25/.50/1.0. Choose minimum area among passing settings, or maximum recall then area if none passes. Lock one setting for A2–A4. Test split is evaluated only after all settings/runs are locked. Both decoders are reported for every run.

Version2 checkpoints include full config, names, split IDs, phase, optimizer, epoch, selected metric, training settings and RNG. Reload is strict. A1→local warm-start permits only missing local parameters and rejects all other missing/unexpected keys. Ordered semantic names and split IDs must match the manifest. Completed phase markers let a campaign skip finished phases; interrupted phases resume the last completed epoch and recreate the optimizer from its saved state. An interrupted partial epoch is repeated. The campaign locks config, seed, decoder, epoch budgets and full manifest before training, then locks the SHA256 of the chosen A1 and validation ROI settings for A2–A4. Changing that protocol requires a new output directory. When standalone resume changes output directory, the companion best.pt is verified and copied so historical checkpoint selection is retained.

Historical keypoint source/checkpoints were absent from this imported folder; actual old checkpoint name compatibility cannot be proven. Configs without a local section are interpreted as global-only, and legacy `model_state_dict`/`state_dict` fields are accepted with strict parameter validation.

## Slurm commands

Follow `/home/tinnt/slurm.md`: no Python imports, numerical tests, image processing, training or benchmarks on login nodes. All supplied scripts launch through `srun` inside `sbatch`. CPU work requests `compute`; GPU work requests `gpu` and one GPU. Scripts use the existing Linux environment `/opt/hpc/miniforge3/envs/ai-pytorch/bin/python`; the copied Windows `.venv` is unused. A per-job `PYTHONPYCACHEPREFIX` prevents stale shared bytecode. Logs live under `runs/slurm_logs`.

Run these lightweight submissions from the project folder:

```bash
mkdir -p runs/slurm_logs
sbatch scripts/slurm/test.slurm
sbatch scripts/slurm/prepare.slurm
sbatch scripts/slurm/benchmark.slurm --output runs/benchmark_A4.json
sbatch scripts/slurm/smoke.slurm
sbatch scripts/slurm/campaign.slurm --output runs/ablation5000
```

Wait for prepare before dependent jobs; use `--dependency=afterok:JOBID` when submitting. Run benchmark and smoke successfully before the full campaign. The smoke script trains one epoch per phase on one selected train/val/test frame; it checks operation and produces an A0–A4 quantitative engineering report, without evidence of generalization. The full campaign uses all4,000/500/500 and defaults to a seven-day job limit; resubmit the same command/output to continue from completed epochs/phases.

Individual phase and output examples:

```bash
sbatch scripts/slurm/run.slurm train --phase global --local off --output runs/A1/global
sbatch scripts/slurm/run.slurm train --phase local --local on --warm-start runs/A1/global/best.pt --output runs/A4/local
sbatch scripts/slurm/run.slurm train --phase joint --local on --warm-start runs/A4/local/best.pt --output runs/A4/joint
sbatch scripts/slurm/run.slurm evaluate --checkpoint runs/A4/joint/best.pt --split test --output runs/A4/test
sbatch scripts/slurm/run.slurm infer --checkpoint runs/A4/joint/best.pt --split val --max-samples 1 --output runs/A4/infer
sbatch scripts/slurm/run.slurm visualize --artifact runs/A4/infer/artifacts/FRAME.npz --output runs/A4/preview.png
```

Use `--resume` for same-phase continuation, `--warm-start` for phase transitions. CLI flags include `--scales 2|4`, `--local on|off`, `--patches one|five`, `--embedding on|off`, `--decoder argmax|subpixel` and evaluation `--split`. Whitespace is required between an option and its value (for example `--max-samples 1`). `--max-samples` is explicitly a smoke/testing subset inside the persisted5,000 IDs.

## Artifacts and evidence

Each prediction saves numeric NPZ and readable JSON: global/local heatmaps, support/coverage, region and patch boxes and patch responses, argmax/subpixel coordinates in grid/input/original image spaces, peaks, per-axis usage flags, label names, config and source RGB path. Visualization reads those saved artifacts. Metrics include mean/median/P95 and semantic keypoint errors, ROI recall, region/patch size, patch count, sequence length, subpixel/fallback fractions, area/peak/error correlations, runtime/image and VRAM. No prediction is corrected using GT.

Evidence logs and reports in `runs/` are authoritative for completed jobs. Benchmark2031 measured one default A4 image at10.428s forward+backward and2,366,661,632bytes allocated VRAM, with correct four fused shapes and finite gradients. Extrapolated4,000-image epoch time was11.59hours before validation/disk I/O; region configuration and phase affect runtime. Long A0–A4 results must be reviewed separately from engineering smoke results. Local refinement improvement is concluded only on holdout metrics versus A1 with the same decoder.

## Automatic end-to-end continuation

`campaign.slurm` uses its actual Slurm TimeLimit to establish a monotonic soft compute budget, reserving15minutes for a graceful stop in the normal7-day allocation. The elapsed budget avoids the node/controller clock skew observed on this cluster. Numerical code exits75 only on the controlled budget stop; the wrapper submits one successor with `afterany:CURRENT_JOB`, preserving arguments and output. Other failures stop the chain. Cancellation stops the active allocation rather than creating a successor.

The full wrapper then runs A4 inference and saved-artifact visualization, and writes `end_to_end_complete.json` only after those steps succeed. `campaign-chain.tsv` lists allocation IDs; `campaign-continuations.tsv` records successor links. Training progress is logged every100images and saved in each phase's `active_epoch.json`. `sbatch scripts/slurm/validate.slurm` checks all5,000 samples on a compute node before launching the campaign.

Proven continuation:2076→2078, completed smoke report/inference/preview with a real allocation boundary. All5,000 selected frames were validated by2071. Final complete-suite2077 passed165tests with one optional CUDA test skipped on the CPU allocation.
