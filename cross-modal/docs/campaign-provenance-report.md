# Campaign provenance review fix

The campaign resume issue is fixed in `models/keypoints/ablation.py`, with
focused coverage in `tests/test_keypoint_ablation.py`. No experiment or
checkpoint module was edited by this task.

Before model construction or training, `campaign_protocol.json` immutably
records the canonical complete base configuration, training seed, decoder,
global/local/joint epoch budgets, sample cap, resolved dataset root and the
complete persisted manifest. This preserves exact ordered split IDs and
keypoint names, source split fingerprints, manifest seed and declared limit.
Tuple/list representations of an otherwise identical config compare equally.
Repeated identical protocols are allowed; any change requires a new output
directory. A populated output without a protocol lock is rejected because
its earlier experiment provenance cannot be reconstructed safely.

`local_provenance.json` immutably records the SHA256 of the A1 best checkpoint,
the shared local initialization seed, and the full validation region selection
record. Existing local artifacts require this provenance. Resuming a campaign
validates the parent before any global training and checks it again before each
local stage. Changed A1 bytes, local seed, ROI artifact or explicitly supplied
region settings reject with a clear new-output diagnostic. Existing locked ROI
selection is reused without rerunning validation tuning. Malformed lock JSON
and missing/invalid selected region settings also fail clearly.

Reports include a campaign protocol SHA256, A1 checkpoint SHA256 and local
initialization seed. Each A2–A4 row records the same parent hash and seed plus
the locked level, margin and overlap configuration. Completed phase shortcuts
also validate phase/seed/decoder/epoch metadata, companion best existence and
last/best split, semantic-name and training consistency.

## Slurm evidence

All imports, hashes, temporary fixture work and tests ran on Slurm compute
nodes using `scripts/slurm/test.slurm`. Only reads, edits, submissions and log
inspection ran on the login node.

| Job | Focused command | Result |
|---|---|---|
| 2048 | `tests/test_keypoint_ablation.py` before implementation | RED: missing protocol-lock interface |
| 2049 | Initial provenance implementation | 20 passed in 5.93 seconds |
| 2050 | Added recovery and malformed-lock regressions | RED: missing-region lock did not reject; JSON-list lock raised AttributeError; 23 passed |
| 2051 | Final complete focused suite | 25 passed in 5.94 seconds |

Logs: `runs/slurm_logs/test-<job>.out` and `.err`.

The final tests exercise immutable identical resumes, each changed protocol
field, reordered split IDs and keypoint names, legacy populated output,
changed parent bytes, local initialization seed and region selection,
pre-training rejection, unattributed local artifacts, altered region artifacts,
malformed local metadata, report lineage and completed campaign ROI reuse.
An independent read-only final review found no important remaining defects.

The campaign orchestration test substitutes expensive training, model loading,
evaluation and benchmarking while exercising actual filesystem locks, hashes,
resume guards and report persistence. This task did not rerun the full project
suite or a real-data campaign; the root integration task runs those checks
against a fresh output directory after this focused verification. Existing
campaign outputs created before these locks cannot be silently adopted.
