# Implementation status — 2026-10-01

The implementation is in `/home/tinnt/mamba/cross-modal`; the source dataset is `/work/tinnt/BlenderProc/output/keypoints_bbox_50000_job_1634`. The persisted manifest is `runs/subset5000.json`, selecting4,000 train /500 validation /500 test from the source splits, seed20260922, with root and split fingerprints checked on reuse.

## Verified evidence

All numerical work was executed on Slurm compute nodes. Initial suite job2013:20 passed,1 skipped. Integrated pre-review suite job2043:129 passed,1 skipped in11.63s. Focused model/local job2040:33 passed; regions/decoder/loss job2035:45 passed; data/artifacts job2037:23 passed. The one skipped test covers the optional CUDA backend (CUDA is unavailable in the CPU allocation); default torch backend was exercised on both CPU and GPU.

GPU benchmark2031 with default A4 (180×320, four scales,45 native patches) measured10.428s for forward/backward,2,366,661,632bytes peak allocated VRAM, finite gradients, fused shapes32×45×80/23×40/12×20/6×10 and global/local outputs9×90×160/9×45×80. Local component timings were80ms regions,1.027s grouped scan/crop/head and2.86ms merge. Estimate for a4,000-image joint epoch:11.59h before validation/disk I/O.

Real-data smoke job2042 completed A0–A4 with one train/validation/test image per phase, one epoch per phase, checkpoint reload, inference and visualization. Report: `runs/smoke5000_2042/report.json`; saved preview: `runs/smoke5000_2042/preview.png`. The chosen validation ROI threshold/margin was.30/.25; recall on that single validation frame was100%, with the entire45×80 F1 covered. This is an engineering smoke and does not assess the true500-frame holdout.

|Smoke run|Mean argmax error(original px)|Mean subpixel error|Subpixel median|Subpixel P95|Test inference s/image|Training pass s/image|Peak VRAM(bytes)|
|---|---:|---:|---:|---:|---:|---:|---:|
|A0|610.577|609.687|666.314|787.981|.081|.395|337914368|
|A1|562.373|562.262|647.506|796.617|1.166|6.358|349358592|
|A2|212.025|214.728|203.726|367.586|1.696|9.555|1245392896|
|A3|212.025|214.917|203.914|367.774|1.860|9.728|2377672192|
|A4|212.025|214.957|203.931|367.841|1.845|9.702|2377676800|

These one-frame values are retained for reproducible pipeline checking and do not establish improvement. Full5000-frame results and the claim that local refinement improves on A1 require the complete research campaign and final locked test evaluation.

## Review fixes

Independent report: `docs/final-review.md`. Four important recovery/provenance issues were identified and accepted: campaign protocol/A1 parent lock, ordered keypoint-name validation, current-invocation-only artifact statistics, and carrying the best checkpoint when resume output changes. Global-phase validation selection was also corrected and its regression observed failing on job2044. Root recovery/artifact regressions were observed failing on jobs2045/2046 before fixes. Subsequent verification results are appended below as jobs finish.

## Limits

The imported folder had no keypoint baseline source, no historical keypoint checkpoint and no Git repository despite its README references. The documented global baseline was rebuilt; actual historical parameter-name compatibility cannot be verified. Current checkpoint roundtrip is strict and tested; configs missing local are global-only.

The full A0–A4 campaign takes substantially longer than the smoke. Job time limits bound each submission; safe continuation resumes completed epochs/phases under an immutable protocol. The incomplete epoch after an interruption is repeated. All source images remain read-only; all outputs/logs/checkpoints are under this project.

## End-to-end continuation — 2026-10-01

The user explicitly requested continuing end-to-end. Final suite2077: **165 passed,1 skipped in15.73s**; skipped CUDA backend test in CPU allocation. Dataset audit2071 decoded and validated all5,000 selected RGB/depth/pose/annotation tuples in123.85s:4,000 train/500 validation/500 test; every semantic keypoint has5,000 valid labels; no frame has wholly invalid depth; native size720×1280.

Strict companion-best validation now checks a full independent state reload, config, phase, recorded best metric, epoch, names, splits and training metadata while preserving resumed RNG. Regression job2068 demonstrated rejection gaps; targeted2070 passed35tests and complete2077 passed165tests.

Real continuation proof: GPU job2076 ran a one-minute allocation, retained completed phases, exited75 on the soft compute budget and automatically submitted2078. Both batch jobs completed0:0;2078 resumed and produced the A0–A4 report, inference artifacts, preview and `runs/continuation_smoke5000/end_to_end_complete.json`. Numeric step2076.0 exit75 is the expected controlled-stop signal, not a training failure. The preceding2072 trial revealed separate `python -m` exception identities and controller/node wall-clock skew; both causes were fixed using a shared runtime module and monotonic elapsed budgets.

The full campaign is submitted separately at `runs/ablation5000` without `--max-samples`:2from-scratch global runs(max200epochs each),3local/joint variants(max50epochs each phase) with validation/early stopping, shared A1/local initialization, locked ROI, final500-frame test perrun, inference and saved-artifact visualization. Each GPU allocation is at most7days; a controlled budget stop automatically submits one successor using the same immutable protocol and output. Ordinary errors/cancellation do not trigger a new job. An incomplete epoch is repeated from the last completed checkpoint. Completion is indicated by `runs/ablation5000/end_to_end_complete.json`, not by scheduler completion of an intermediate allocation.

Live progress is saved every100training images to `RUN/PHASE/active_epoch.json`. Finished epochs append `history.jsonl` and save `last.pt`/`best.pt`. All allocations and links are in `runs/slurm_logs/campaign-chain.tsv` and `campaign-continuations.tsv`. Initial full-campaign job ID and verified startup are recorded below. A source snapshot is stored at `runs/provenance/full5000_source_20261001.tar.gz`.

**Full campaign startup verified:** Slurm2082 is RUNNING on gpu/hpc01 with one GPU,4CPU,64GiB and7-day allocation. Scheduler start:2026-10-01 16:09:18 Asia/Ho_Chi_Minh. `campaign_protocol.json` records `max_samples:null`, exact4,000/500/500 IDs, seed20260922 and epoch ceilings200global/50local/50joint. A0/global/config.json was written and stderr was empty on startup. The full research run is ongoing; final metrics are not yet available.

Verified training progress:2082/A0/global epoch1 has processed600/4,000 training images; running mean loss0.033387; elapsed260.79s, stderr empty. These are training-health values, not holdout quality. Archive header audit2090 checked53,693entries in `/home/tinnt/cross-modal.rar`, with no matching original keypoint source/docs/debug100checkpoint paths and no archive errors; historical compatibility remains unverifiable from the supplied files.
