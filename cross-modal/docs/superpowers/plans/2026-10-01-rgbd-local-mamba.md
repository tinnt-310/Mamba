# RGB-D four-scale and adaptive local Mamba implementation plan

> **For agentic workers:** Use the supplied finalized spec, focused parallel tasks with disjoint files, and verification on Slurm compute nodes. Checkboxes track delivery.

**Goal:** Implement the supplied 2D keypoint pipeline in this folder and restrict the experiment to exactly 5,000 samples from the 50,000-frame dataset.
**Architecture:** SE-ResNet RGB and metric geometry encoders feed four existing compensation blocks and a mean multiscale global decoder. Adaptive native-size F1 patches feed one shared four-direction SSM, a 1x1 heatmap head, overlap averaging and masked subpixel decoding.
**Tech stack:** PyTorch, timm, NumPy, Pillow, Slurm; existing selective scan only.
**Spec:** ../specs/2026-10-01-rgbd-local-mamba.md (user-provided approved handoff).

## Global constraints

- Work in `/home/tinnt/mamba/cross-modal`; this imported folder has no Git repository and no keypoint baseline source. Recreate the documented baseline alongside the new modules; do not claim historical checkpoint compatibility without a real old checkpoint.
- Dataset root `/work/tinnt/BlenderProc/output/keypoints_bbox_50000_job_1634`; preserve source split membership. Seed `20260922`, subset of 4,000 train / 500 val / 500 test, persisted IDs. Never train or evaluate the remaining 45,000 frames.
- Input 180x320; four indices (1,2,3,4), fusion widths (32,32,32,32); heatmap 90x160, F1 45x80, nine named keypoints from annotations.
- Global SSM state_dim=4, expand=1, directions=1; local state_dim=16, directions=4, backend=torch.
- Threshold .30, margin .25, overlap .50, corners+center with stable deduplication, patch microbatch <=8. Crop by slicing, no resizing/padding/ROIAlign.
- All computational programs, numerical tests, data validation, benchmarks and training run through Slurm. Login node is for reads/edits/submission/monitoring only. CPU partition compute, CUDA partition gpu, --gres=gpu:1.
- `forward(rgb,depth,K)` remains global only; `forward_stages(...,return_patches=False)` shares one encoder/fusion pass. Detach only region heatmap, preserve F1 gradient.
- Config v2 with strict same-architecture reload. Only local keys may be missing during A1->local warm-start.
- Annotation GT drives targets; support is a separate bool mask. Local loss balances pixels, then patches per keypoint, then valid keypoints.

## Review focus

- Missing baseline and old checkpoint provenance: same-version roundtrip is tested; actual historical state names cannot be inferred.
- Split leakage or accidentally indexing all 50k: validate disjoint persisted subset and source split membership, reject limits above 5000.
- Empty/negative/nonfinite maps and missing labels: explicit validation, finite graph-connected zero loss, masked argmax.
- Native patch microbatches and local-only training: exact variable shapes, BatchNorm global eval, local train, no global-head gradient.
- Long torch scan runtime and CPU feature cache size: batch-one benchmark first, bounded cache memory, do not launch expensive full campaign without measured feasibility.

## Task 1: Regions, decoding and loss

**Files:** models/keypoints/regions.py, heatmaps.py, local_losses.py; tests/test_keypoint_regions.py, test_local_losses.py, test_keypoint_heatmaps.py.
**Interfaces:** Box frozen dataclass(x1,y1,x2,y2) with width,height; peak_component_region(heatmap,level=.30,margin_ratio=.25), scale_box(box,src_hw,dst_hw), make_patch_boxes(region,layout='five',overlap_ratio=.50), extract_regions(global_heatmaps,feature_hw,...) returns records with batch_index,keypoint_index,global_box,feature_box,patch_boxes. transform_points(points,src_hw,dst_hw) center convention for torch/NumPy. gaussian_heatmaps(points,hw,sigma,weights=None). decode_heatmaps(heatmaps,support=None,subpixel=True) -> DecodedKeypoints(coords,argmax,peak_response,subpixel_used). weighted_heatmap_loss(pred,target,weights,positive_weight=100). local_patch_loss(patches,points,weights,feature_hw,sigma=1,positive_weight=100) -> scalar; points already in F1. Patch record prediction shape [1,1,h,w].
- [x] Write tests first and observe missing-function failures on compute partition.
- [x] Implement exact flood fill, bounds, masks, log-parabola and balancing from spec.
- [x] Run all covering numerical tests via Slurm; record job ID/result.

## Task 2: Model and local refinement

**Files:** models/keypoints/model.py, local_refinement.py, config.py; tests/test_keypoints.py, test_local_refinement.py.
**Interfaces:** DEFAULT_CONFIG nested dict input_size,heatmap_size,out_indices,geometry_channels,fusion_channels,decoder_width,num_keypoints,pretrained,global_ssm,local; make_config(overrides=None) deep merge. RGBDKeypointModel(config=None) with config,local_enabled,local_refinement,encode_fused(rgb,depth,K),global_from_fused(fused),forward_stages,refine_cached(f1,global_maps,return_patches=False),set_phase(phase). StageOutput(global_heatmaps,local). LocalRefinementOutput(heatmaps,support,coverage,regions,patches); PatchPrediction(batch_index,keypoint_index,box,prediction). FourDirectionLocalMamba(channels=32,state_dim=16,backend='torch'). AdaptiveLocalRefinement(channels,num_keypoints, **local_config).
- [x] Write model/native patch/gradient tests first; observe failure on compute partition.
- [x] Build both encoders with selected indices directly, existing fusion, checked decoder projections.
- [x] Implement shared SSM, shape grouping, embedding, patch head, differentiable overlap merge.
- [x] Verify default shapes, odd grids, gradient isolation, local-only eval behavior on Slurm.

## Task 3: Dataset and artifacts

**Files:** models/keypoints/data.py, outputs.py; tests/test_keypoint_data.py, test_keypoint_outputs.py.
**Interfaces:** create_subset_manifest(root,path,limit=5000,seed=20260922) -> dict with keypoint_names and train/val/test ID lists. RGBDKeypointDataset(root,ids,keypoint_names=None,input_size=(180,320),heatmap_size=(90,160),sigma=2) -> mapping frame_id,rgb [3,H,W],depth [1,H,W],K [3,3],points [K,2] input coords,weights [K],original_points [K,2],original_size [2],target [K,Hh,Wh]. load_manifest validates persisted splits. save_prediction_artifact(path,sample,stage,config,keypoint_names) -> npz+JSON; decode global and local both ways, retain raw maps/support/coverage/boxes, original and input coords. compute_metrics(predictions,targets,weights) -> mean,median,p95,per_keypoint. visualize_artifact(path,output_path), aggregate_region_statistics(...).
- [x] Read annotations/metric depth metadata and source splits; write synthetic loader and subset tests first.
- [x] Preserve 80/10/10 source split; persist seeded 5000 IDs without copying source images.
- [x] Implement consistent center transforms/K update, visibility/weight policy, saved-artifact visualizer.
- [x] Verify exact subset and artifact reconstruction via compute Slurm job.

## Task 4: Experiment, checkpoint and Slurm operation

**Files:** models/keypoints/experiment.py, checkpoints.py, ablation.py; scripts/slurm/test.slurm, benchmark.slurm, campaign.slurm, smoke.slurm; tests/test_keypoint_experiment.py; docs/KEYPOINT_EXPERIMENT.md.
**Interfaces:** CLI prepare,train,evaluate,infer,visualize,benchmark,ablate; phases global/local/joint, split train/val/test, local on/off, patches one/five, embedding on/off, decoder argmax/subpixel. Strict config-v2 checkpoint, optimizer + epoch/phase/metric/splits/keypoint_names. A0 two-scale vs A1 four-scale from scratch; A2-A4 same A1 and same local shared initialization, region configuration locked by val recall tuning.
- [x] Run original baseline tests using Slurm before integrating changes.
- [x] Implement phase parameter groups, no decay SSM metadata, accumulation, clipping, evaluation schedule/early stop and bounded CPU F1 cache.
- [x] Implement checkpoint roundtrip, strict warmstart/resume, train/evaluate/infer/artifact integration and ROI tuning.
- [x] Supply Slurm scripts enforcing compute-node execution; benchmark batch=1 before full campaign, log IDs and runtime/VRAM evidence.
- [x] Verify full suite and real-data smoke train/reload/evaluate/infer/visualize on 5000-only manifest.
- [x] Review whole implementation with a separate agent, fix important findings, publish evidence and remaining experiment status.

Execution evidence: originaltests2013, complete165-test suite2077, all5,000-frame audit2071, freshfullstage/reload/infer/visualize smoke2053, realcontinuation2076→2078. Full5,000-frame A0–A4 campaign is RUNNING as2082; this is ongoing research work and not a completed holdout result. Source snapshot/protocol/launch record and detailed evidence are in runs/ and docs/IMPLEMENTATION_STATUS.md.
