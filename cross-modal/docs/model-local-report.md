# Model and native local refinement delivery

Task 2 of the approved 2026-10-01 implementation plan is implemented in
`models/keypoints/config.py`, `model.py`, and `local_refinement.py`, with
coverage in `tests/test_keypoints.py` and `tests/test_local_refinement.py`.

The default model uses selected encoder stages `(1,2,3,4)` directly, five
geometry stage widths `(32,48,96,128,192)`, four compensation blocks, and a
checked projection/resize/mean decoder with two Conv/GN/GELU blocks. The
180x320 input produces fused grids 45x80, 23x40, 12x20 and 6x10, and nine
90x160 global maps. No extra geometry projector is inserted before fusion.

`forward` returns only global heatmaps. `forward_stages` shares one encoder
and fusion pass; `refine_cached` returns the same `StageOutput` interface.
Region generation detaches global heatmaps but retains F1 autograd. Local
phase freezes global parameters and keeps all global modules in eval mode
after subsequent `train()` calls; joint phase restores trainable parameters.

Local refinement slices native F1 patches, groups only equal shapes, and
processes microbatches of at most eight patches. One `SimpleSelectiveSSM`
shares weights across four directions, patches and keypoints. The local
path is scan/SSM/inverse merge/channel LayerNorm/1x1 projection/residual,
optional keypoint embedding, and one 1x1 prediction head. Outputs retain
integer coverage and an independent bool support mask. Gradient-bearing
patch tensors are returned only when requested.

Overlap merging determines coverage before scattering coverage-normalized
contributions. This computes the same arithmetic mean while avoiding an
overflowing numerator for large finite predictions. No resampling or
padding is used to batch patches. Opt-in `local_refinement.profile=True`
records region, scan and merge durations; scan includes grouping, slicing,
embedding and the head. CUDA synchronization for those timing boundaries
occurs only when profiling is enabled.

## Compute-node verification

All imports, tensor operations and tests were executed with
`sbatch scripts/slurm/test.slurm`; none ran on the login node.

| Slurm job | Command scope | Observed result |
|---|---|---|
| 2015 | Model/local tests before implementation | RED: missing config and local modules, two collection failures |
| 2019 | Initial model/local integration and profiling contract | 29 passed, one expected missing-profile failure |
| 2024 | Parallel full-suite run with boundary regression additions | RED: level-zero validation, full-overlap acceptance and missing profiling; two independent Task 1 regressions also reported |
| 2027 | Model/local after config/profile edits | 30 passed, two failures from old config bytecode despite newer traceback source |
| 2034 | Model/local with fresh bytecode prefix | 32 passed in 8.40 seconds |
| 2039 | Large finite overlap averaging regression | RED: finite 2e38 inputs produced infinite merged values |
| 2040 | Model/local after overlap arithmetic fix | 33 passed in 8.01 seconds |
| 2043 | Bare full-project pytest after final changes | 129 passed, one skipped in 11.63 seconds |

Logs are in `runs/slurm_logs/test-<job>.out` and `.err`. Fresh
`PYTHONPYCACHEPREFIX` directories were used for the final runs to avoid
shared cached bytecode after rapid remote filesystem edits.

The focused tests cover default four-scale shapes, odd input grids,
native 1x1/1xN/Nx1/rectangular/23x40 patches, actual scan ordering, one shared
SSM, exact negative overlap means and gradients, finite extreme overlap,
variable shape groups and microbatch limits, embedding on/off, shared
encoder passes, cached refinement, global-only legacy config behavior,
decoder scale-count checks, phase-B BatchNorm isolation, and local-loss
gradients to F1/fusion/encoders without gradients through the global head.

A separate read-only review by the data/artifact agent found no important
defects in the model/config/local contracts. That review also identified a
phase-A evaluation selection issue in the experiment module; it was sent
to the experiment owner because that module lies outside this task.

## Limits of this evidence

The imported folder has no historical keypoint model source or historical
checkpoint. This recreates the documented decoder; it does not establish
compatibility with unknown old parameter names. Current strict checkpoint
roundtrip checks are part of the full project suite owned by the experiment
task.

This report establishes numerical and interface behavior on Slurm CPU
compute nodes. The skipped full-suite check is the optional CUDA backend
test when CUDA is unavailable. Default-size local runtime/VRAM, real-data
smoke evidence, training campaigns and holdout improvement are reported by
the experiment task separately. No research improvement is inferred from
these unit tests, and no Git commits were created.
