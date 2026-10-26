# Dataset and artifact implementation evidence

The task-specific dataset and artifact modules are implemented in
`models/keypoints/data.py` and `models/keypoints/outputs.py`. The task's final
covering verification ran on Slurm compute node hpc08 in job **2037**:
**23 passed in 3.09 seconds**, with no warnings or errors.

## Data and subset contract

The source dataset is
`/work/tinnt/BlenderProc/output/keypoints_bbox_50000_job_1634`.
Its camera metadata declares 1280x720 RGB, pinhole intrinsics with integer pixel
centers, and optical-axis Z depth in meters. Sample `000000` contains RGB at
`rgb/000000.png`, sensor depth at `depth_sensor/000000.npy`, per-frame intrinsics
in `poses/000000.json`, and nine named `keypoints_2d` annotation records.
Generation metadata reports 50,000 frames; the implementation does not rely on
the generation quality report to validate a loaded sample.

`create_subset_manifest(root,path,limit=5000,seed=20260922)` samples independently
within the original source splits and saves exact, sorted selected IDs. The
default allocation is 4,000 train, 500 validation and 500 test frames. No source
images are copied. The manifest includes the absolute source root, seed,
keypoint names and SHA-256 fingerprints of the source split ID sequences.

`load_manifest(path,root=...)` rejects caps above 5,000, unsafe filename stems,
duplicates, intersecting splits, changed 80/10/10 subset counts, source split
changes, incorrect source membership and a requested root differing from the
persisted source root. All numerical subset creation/validation is done in
Slurm jobs. The real-dataset regression verifies exactly 5,000 unique selected
IDs and their source split membership.

`RGBDKeypointDataset` returns the trainer fields `frame_id`, `rgb`, `depth`, `K`,
`points`, `weights`, `original_points`, `original_size` and `target`, plus
`visibility` and `rgb_path`. RGB is bilinearly resized and normalized with
ImageNet channel statistics. Sensor depth is float32 in meters and nearest
resized, without filling or visualization-depth substitution. RGB, sensor
depth and pose paths must match the annotation's frame ID and remain within
the dataset root. Original RGB and sensor depth dimensions must match. K must
be finite with positive focal lengths and pinhole bottom row.

Points and K use the same pixel-center affine transform. Labels are ordered by
semantic names rather than record positions. Visibility 1 and 2 are labeled;
visibility 0, absent labels, nonfinite coordinates and points outside the
original pixel footprint have zero target/loss weight. The original footprint
is half-open `[-0.5,W-0.5) x [-0.5,H-0.5)`.

## Artifact and analysis contract

`save_prediction_artifact(path,sample,stage,config,keypoint_names)` accepts one
unbatched sample and a batch-one stage. It writes an NPZ of numeric arrays,
readable with `allow_pickle=False`, and a JSON sidecar. The files retain raw
global/local heatmaps, support/coverage, region/patch boxes, optional individual
patch predictions, labels/weights, semantic names, source RGB path and config.

Both argmax and subpixel coordinates are saved in heatmap-grid, input-image
and original-image spaces. Peak responses and independent-axis subpixel-use
flags are retained. Local decoding always uses saved support; an unsupported
large response cannot win even when supported predictions are negative.

`compute_metrics` reports valid-point mean, median, P95 and per-keypoint pixel
errors, accepting optional semantic names. Empty label sets return count zero
and null metrics. Nonfinite predictions at valid labels raise an error rather
than silently disappearing from the results.

`aggregate_region_statistics` reads saved JSON/NPZ paths or metadata dicts and
reports ROI recall from the union of patch pixel footprints, region/patch
sizes and counts, mean/max sequence lengths, independent-axis subpixel and
fallback rates, and correlations of region area/peak response with global
localization error. It also aggregates optional saved image runtimes and peak
VRAM values. `visualize_artifact` uses the saved source image, boxes and decoded
coordinates directly; it does not alter predictions using ground truth.

## Verification record

- Job 2016: expected initial missing-module failures before implementation.
- Job 2022: initial 20 dataset/artifact tests passed, including real 5,000-frame
  manifest validation and saved-artifact visualization.
- Job 2033: three provenance regressions failed before their fixes.
- Job 2037: final 23 dataset/artifact tests passed in 3.09 seconds.

An intermediate full-suite snapshot in job 2024 ran during concurrent module
development and reported 117 passed, 1 skipped and 5 failures in other agents'
new regression tests. Those were reported to their owners: large-response
region normalization, finite unlabeled loss zero, level-zero config rejection,
optional local profiling, and full-overlap patch handling. Final complete-suite
verification is coordinated by the root agent after all parallel tasks finish.
No numerical programs or image decoding ran on the login node in this task.
