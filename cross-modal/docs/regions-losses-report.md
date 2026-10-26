# Adaptive regions, keypoint decoding, and losses

Task 1 implements the interfaces in `docs/superpowers/plans/2026-10-01-rgbd-local-mamba.md` against the finalized algorithm specification. No Git repository was initialized and no commits were made.

## Delivered files

- `models/keypoints/regions.py`: immutable integer half-open `Box`, `RegionRecord`, detached float32 extraction with one CPU transfer per batch, row-major peak tie selection, min-max thresholding, four-connected component flood fill, ceil margins, outward box mapping, and native-size corners-plus-center patches with stable deduplication.
- `models/keypoints/heatmaps.py`: torch/NumPy point-center transforms, annotation-derived Gaussian targets, and `DecodedKeypoints` with masked argmax, independently validated log-parabola offsets, peak response, and per-axis subpixel flags.
- `models/keypoints/local_losses.py`: exact pixel weights `1 + (positive_weight - 1) * target`; global annotation-weighted MSE; local pixel/patch/keypoint balancing; half-open pixel-center footprint checks; invalid-label handling; finite graph-connected zero for all-invalid batches.
- `tests/test_keypoint_regions.py`, `tests/test_keypoint_heatmaps.py`, `tests/test_local_losses.py`: 45 focused numerical cases.

`gaussian_heatmaps` accepts unbatched `[K,2]` and batched `[B,K,2]` points. Decoder outputs preserve leading map dimensions; for `[B,K,H,W]`, coordinates and argmax are `[B,K,2]` in `(x,y)` order, peak responses `[B,K]`, and subpixel flags `[B,K,2]`. Positive local annotation weights act as validity flags so every valid keypoint has one vote. Global annotation weights use a normalized weighted mean. Local loss consumes patch records with prediction `[1,1,h,w]` and already-transformed F1 points.

## Verification evidence

All Python imports and numerical tests ran on Slurm compute nodes through `scripts/slurm/test.slurm`, using `/opt/hpc/miniforge3/envs/ai-pytorch/bin/python`. Login-node activity was limited to file reads, edits, submission, and log inspection.

| Slurm job | Scope | Observed result |
| --- | --- | --- |
| 2017 | Three new focused test files before implementation | Expected RED: three missing-module collection errors; 0.87 seconds. |
| 2018 | Initial implementation, three focused test files | 41 passed; 0.84 seconds; stderr empty. |
| 2023 | Added finite-response regressions | Expected RED: all-invalid large finite prediction returned NaN; 1 failed, 42 passed; 0.85 seconds. |
| 2025 | Loss regression confirmation | Same expected RED; 1 failed, 12 passed; 0.81 seconds. |
| 2028 | Mixed valid/invalid label overflow test | Expected RED: invalid keypoint squared before masking produced NaN; 1 failed; 0.80 seconds. |
| 2030 | Corrected modules, three focused files, shared bytecode cache | Stale imports executed preceding module code: 3 failed, 42 passed. Source/traceback line mismatches identified the cache issue. |
| 2035 | Corrected modules, fresh Python cache prefix | **45 passed; 2.15 seconds; stderr empty.** |
| 2038 | Complete project suite, fresh Python cache prefix | 1 failed, 128 passed, 1 skipped; 11.65 seconds. Only failure: model owner's in-progress `test_overlap_merge_keeps_large_finite_mean_finite`; all Task 1 tests passed. |

Job logs are under `runs/slurm_logs/test-JOBID.out` and `.err`. Job 2035 used `sbatch --export=ALL,PYTHONPYCACHEPREFIX=/tmp/mamba-regions-fresh-20261001-1512 scripts/slurm/test.slurm tests/test_keypoint_regions.py tests/test_keypoint_heatmaps.py tests/test_local_losses.py` to isolate shared-filesystem stale bytecode. The same fresh-cache pattern was used for full-suite job 2038.

An earlier integration run submitted by another worker (job 2024) observed 5 failed, 117 passed, 1 skipped: the extreme-range region and all-invalid loss regressions, plus `test_invalid_config_is_rejected_before_allocating_model[overrides6]`, `test_opt_in_profile_reports_region_scan_and_merge_timings`, and `test_full_overlap_deduplicates_to_one_native_region_patch`. The model owner addressed those three model/config failures; Task 1 fixed its two numerical failures and added the mixed-validity regression before final focused verification.

The remaining job 2038 merge overflow was reported to the model owner and root worker. The model owner confirmed a coverage-before-summation fix and submitted focused job 2040; its verification and next full-suite run are tracked by that worker.

## Numerical details and limits

Region inputs still detach and convert to float32; normalization arithmetic then uses float64 so finite opposite-sign extrema cannot overflow during subtraction. All-invalid losses anchor zero to a single finite element per prediction instead of summing values before multiplying by zero. Global loss masks invalid keypoints before squaring to prevent irrelevant large errors contaminating valid labels.

No patch resizing, padding, ROIAlign, image transforms through GT predictions, new selective scan, or dataset access is introduced by this task. Empty prediction lists have no model graph available; local loss returns a finite differentiable zero leaf in that case. The required all-invalid batch path remains attached to every supplied patch prediction.
