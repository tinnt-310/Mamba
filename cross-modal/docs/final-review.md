# Independent implementation review

Reviewed the 2026-10-01 RGB-D local Mamba specification and implementation plan against `models/keypoints`, the Slurm scripts and the focused tests. The review was read-only except for this report. No numerical program or test was run on the login node, and existing tests were not repeated. An independent reviewer inspected dataset splits, artifacts and campaign operation.

## Findings sent to the implementation owner

1. **Important: campaign auto-resume can mix parent checkpoints and initializations.** `models/keypoints/ablation.py:92` reuses phase checkpoints based on architecture, split/name metadata, sample scope and phase epoch budget, without checking the parent A1 checkpoint or campaign seed. Re-running a partly completed campaign with increased global epochs can replace A1 while reusing completed A2–A4 stages trained from the prior A1. Changing the seed after A2 completes can likewise leave A2 and newly started A3/A4 with different common initialization. Persist and validate the campaign protocol before changing any existing stage, and associate downstream stages with the selected A1 parent.

2. **Important: standalone checkpoint commands do not check ordered keypoint names.** `models/keypoints/experiment.py:258`, `:403` and `:417` check split IDs for resume, warm-start and evaluation, but do not check `keypoint_names`. A valid manifest with unchanged IDs and reordered semantic names silently assigns checkpoint channels to different GT labels and artifact labels. Compare the ordered names at each checkpoint entry path, as the campaign already does.

3. **Important: statistics can include stale prediction artifacts.** `models/keypoints/experiment.py:184` aggregates every existing JSON file in the artifact directory. Running evaluation for validation and then test with the same default output, or lowering the sample count, mixes prior artifacts into the current ROI/subpixel/runtime statistics. Collect and aggregate only the paths written by the current invocation.

4. **Important: resuming into a new output directory can report a missing best checkpoint.** `models/keypoints/experiment.py:262` restores the historical best metric, but `:287` only writes a new `best.pt` when the resumed model improves. If no subsequent epoch improves, or the checkpoint already reached the requested epoch limit, `:303` marks completion and `:306` returns the new output's nonexistent `best.pt`. Require and carry the historical best checkpoint, retain its valid source path, or reject an unsupported relocation before training begins.

## Scoped review of fixes

The campaign provenance, ordered semantic names, and stale-artifact findings are resolved. `ablation.py:201` validates an immutable complete campaign protocol before training; `:205` and `:270` validate the A1 SHA256, local initialization seed and locked validation ROI. Completed phase reuse also checks phase/training metadata and companion best existence. Changing global epoch budgets, seed, decoder, sample scope, splits or base configuration requires a new output directory, so the original mixed-parent trigger fails before existing training changes. Locked ROI selection is reused on an identical resume.

`experiment.py:59` validates exact split IDs and ordered keypoint names. Resume, warm-start, evaluate and infer use it before training or metric computation. `:152` collects saved artifact paths from the current invocation and `:194` aggregates that list, so prior files cannot enter current statistics.

The original missing-best-path trigger is resolved by copying the historical companion `best.pt` into a relocated output before resumed training. A remaining strict-resume concern was sent to the owner: `experiment.py:281` currently loads the companion directly and validates only IDs/names/epoch. It should also verify its architecture/state, phase and recorded metric before returning it as the selected result. A same-epoch companion from an unrelated compatible-ID run can currently pass this guard.

The fix review remained scoped to these four findings and their covering tests. No critical issue was found in the initial native patch, overlap merge, loss, subpixel, or fresh-run split/ROI/common-initialization review.

## Evidence and limits

- Existing baseline verification: Slurm job 2013, 20 passed and one skipped.
- Existing complete-suite verification: job 2043, 129 passed and one skipped in 11.63 seconds; inspected the stored log. This precedes the recovery fixes requested by this review.
- Standalone regression verification: job 2047, 12 passed in 8.61 seconds; includes semantic-order rejection, relocated best preservation, current-only artifact statistics and the phase-A global metric selection regression.
- Campaign provenance regressions: job 2051, 25 passed in 5.94 seconds; includes protocol/parent/region changes rejected before training and completed campaign ROI reuse.
- Updated full suite: job 2052, 158 passed and one skipped in 12.65 seconds. The skip is the optional CUDA backend test on the CPU compute node.
- Existing GPU benchmark: job 2031, reported A4 training pass approximately 10.4 seconds per image and 2.37 GB peak VRAM.
- Real-data A0–A4 smoke job 2042 produced its campaign report; final status and visualization/reload evidence are maintained by the implementation owner.
- Fresh real-data smoke job 2053 produced its A0–A4 report and A4 inference metrics with one current artifact. Final Slurm status and visualization evidence are maintained by the implementation owner.
- The real manifest is constrained to the authorized 5,000 frames, preserving 4,000/500/500 original source split membership. Numerical validation was performed by the implementation tasks on Slurm compute nodes.
- No imported keypoint baseline source, old checkpoint, or Git repository exists. Current same-version strict reload is tested; historical checkpoint compatibility remains unverified.
- Unit tests and a one-sample engineering smoke do not establish holdout improvement. The full 5,000-frame A0–A4 research campaign remains separate from implementation correctness.

## Owner verification after scoped review

The residual companion-best concern was fixed: independent strict reload plus config/phase/metric/epoch/training/names/splits agreement is required before relocation. Parameter/config/phase/metric corruption regressions were observed failing on2068 and passed after the fix; final suite2077 passed165tests,1skip. The shared exception identity/monotonic budget fix was additionally checked by real end-to-end Slurm continuation2076→2078, completing report, reload/inference and preview. These are owner verification results after the scoped reviewer report; no further independent review is claimed.
