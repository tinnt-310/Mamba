from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.keypoints.ablation import (
    lock_campaign_protocol,
    lock_local_provenance,
    run_ablation,
)
from models.keypoints.config import make_config


def campaign_inputs(tmp_path):
    root = tmp_path / "data"
    args = SimpleNamespace(output=str(tmp_path / "campaign"), data_root=str(root),
                           seed=20260922, decoder="subpixel", device="cpu",
                           global_epochs=2, local_epochs=3, joint_epochs=4,
                           max_samples=1)
    manifest = {"manifest_version": 1, "dataset_root": str(root.resolve()),
                "limit": 10, "seed": 20260922,
                "source_split_sha256": {"train": "a" * 64, "val": "b" * 64, "test": "c" * 64},
                "keypoint_names": ["left", "right"],
                "train": [f"tr{i}" for i in range(8)], "val": ["v0"], "test": ["t0"]}
    config = make_config({"input_size": (16, 24), "heatmap_size": (8, 12),
                          "out_indices": (1, 2), "fusion_channels": (4, 4),
                          "geometry_channels": (4, 4, 4, 4, 4),
                          "decoder_width": 4, "num_keypoints": 2,
                          "local": {"state_dim": 2}})
    return args, manifest, config


def selected_region():
    return {"selected": {"level": .15, "margin_ratio": .5, "roi_recall": .96,
                          "mean_area": 7., "valid_keypoints": 2},
            "candidates": [], "target_recall": .95, "recall_target_met": True,
            "split": "val"}


def parent_checkpoint(args):
    path = Path(args.output) / "A1" / "global" / "best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"original-parent-checkpoint")
    return path


def test_protocol_same_canonical_config_preserves_immutable_lock(tmp_path):
    args, manifest, config = campaign_inputs(tmp_path)
    first = lock_campaign_protocol(args, manifest, config)
    lock_path = Path(args.output) / "campaign_protocol.json"
    original = lock_path.read_bytes()
    from_json = json.loads(json.dumps(config))
    assert lock_campaign_protocol(args, deepcopy(manifest), from_json) == first
    assert lock_path.read_bytes() == original


@pytest.mark.parametrize("changed", [
    "seed", "decoder", "global_epochs", "local_epochs", "joint_epochs", "max_samples",
    "base_config", "split_ids", "split_order", "keypoint_names", "dataset_root", "source_fingerprint",
])
def test_changed_campaign_protocol_requires_new_output(tmp_path, changed):
    args, manifest, config = campaign_inputs(tmp_path)
    lock_campaign_protocol(args, manifest, config)
    if changed in ("seed", "global_epochs", "local_epochs", "joint_epochs", "max_samples"):
        setattr(args, changed, getattr(args, changed) + 1)
    elif changed == "decoder":
        args.decoder = "argmax"
    elif changed == "base_config":
        config["global_ssm"]["state_dim"] += 1
    elif changed == "split_ids":
        manifest["val"] = ["other_validation_frame"]
    elif changed == "split_order":
        manifest["train"].reverse()
    elif changed == "keypoint_names":
        manifest["keypoint_names"].reverse()
    elif changed == "dataset_root":
        args.data_root = str(tmp_path / "other_data")
        manifest["dataset_root"] = str(Path(args.data_root).resolve())
    else:
        manifest["source_split_sha256"]["test"] = "d" * 64
    with pytest.raises(ValueError, match="new output"):
        lock_campaign_protocol(args, manifest, config)


def test_unlocked_existing_training_output_cannot_be_adopted(tmp_path):
    args, manifest, config = campaign_inputs(tmp_path)
    parent_checkpoint(args)
    with pytest.raises(ValueError, match="new output"):
        lock_campaign_protocol(args, manifest, config)
    assert not (Path(args.output) / "campaign_protocol.json").exists()


def test_local_provenance_locks_parent_seed_and_region(tmp_path):
    args, manifest, config = campaign_inputs(tmp_path)
    lock_campaign_protocol(args, manifest, config)
    checkpoint = parent_checkpoint(args)
    first = lock_local_provenance(args.output, checkpoint, args.seed,
                                  region_selection=selected_region())
    assert len(first["a1_checkpoint_sha256"]) == 64
    assert first["local_init_seed"] == args.seed
    assert first["region_selection"] == selected_region()
    assert lock_local_provenance(args.output, checkpoint, args.seed) == first


@pytest.mark.parametrize("changed", ["parent", "seed", "region"])
def test_changed_local_lineage_is_rejected(tmp_path, changed):
    args, manifest, config = campaign_inputs(tmp_path)
    lock_campaign_protocol(args, manifest, config)
    checkpoint = parent_checkpoint(args)
    lock_local_provenance(args.output, checkpoint, args.seed,
                          region_selection=selected_region())
    region = selected_region()
    if changed == "parent":
        checkpoint.write_bytes(b"replaced-parent-checkpoint")
    elif changed == "seed":
        args.seed += 1
    else:
        region["selected"]["level"] = .5
    with pytest.raises(ValueError, match="new output"):
        lock_local_provenance(args.output, checkpoint, args.seed, region_selection=region)


def test_campaign_changed_protocol_fails_before_training(tmp_path, monkeypatch):
    args, manifest, config = campaign_inputs(tmp_path)
    lock_campaign_protocol(args, manifest, config)
    args.global_epochs += 1
    import models.keypoints.experiment as experiment
    monkeypatch.setattr(experiment, "train_run", lambda *a, **kw: pytest.fail("training ran before protocol validation"))
    with pytest.raises(ValueError, match="new output"):
        run_ablation(args, manifest, config)


def test_campaign_changed_parent_fails_before_training(tmp_path, monkeypatch):
    args, manifest, config = campaign_inputs(tmp_path)
    lock_campaign_protocol(args, manifest, config)
    checkpoint = parent_checkpoint(args)
    lock_local_provenance(args.output, checkpoint, args.seed,
                          region_selection=selected_region())
    checkpoint.write_bytes(b"changed-parent")
    import models.keypoints.experiment as experiment
    monkeypatch.setattr(experiment, "train_run", lambda *a, **kw: pytest.fail("training ran before lineage validation"))
    with pytest.raises(ValueError, match="new output"):
        run_ablation(args, manifest, config)


def test_local_artifacts_without_parent_lock_fail_before_training(tmp_path, monkeypatch):
    args, manifest, config = campaign_inputs(tmp_path)
    lock_campaign_protocol(args, manifest, config)
    parent_checkpoint(args)
    local = Path(args.output) / "A2" / "local"
    local.mkdir(parents=True)
    (local / "last.pt").write_bytes(b"unattributed-local-state")
    import models.keypoints.experiment as experiment
    monkeypatch.setattr(experiment, "train_run", lambda *a, **kw: pytest.fail("unattributed local artifacts reached training"))
    with pytest.raises(ValueError, match="new output"):
        run_ablation(args, manifest, config)


def test_changed_region_artifact_fails_before_training(tmp_path, monkeypatch):
    args, manifest, config = campaign_inputs(tmp_path)
    lock_campaign_protocol(args, manifest, config)
    checkpoint = parent_checkpoint(args)
    lock_local_provenance(args.output, checkpoint, args.seed, region_selection=selected_region())
    region = selected_region()
    region["selected"]["margin_ratio"] = 1.
    (Path(args.output) / "region_selection.json").write_text(json.dumps(region))
    import models.keypoints.experiment as experiment
    monkeypatch.setattr(experiment, "train_run", lambda *a, **kw: pytest.fail("modified region artifact reached training"))
    with pytest.raises(ValueError, match="new output"):
        run_ablation(args, manifest, config)


@pytest.mark.parametrize("corruption", ["missing_region", "list"])
def test_malformed_local_lock_fails_with_clear_provenance_error(tmp_path, corruption):
    args, manifest, config = campaign_inputs(tmp_path)
    lock_campaign_protocol(args, manifest, config)
    checkpoint = parent_checkpoint(args)
    metadata = lock_local_provenance(args.output, checkpoint, args.seed, region_selection=selected_region())
    if corruption == "missing_region":
        del metadata["region_selection"]
    else:
        metadata = []
    (Path(args.output) / "local_provenance.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="new output"):
        lock_local_provenance(args.output, checkpoint, args.seed)


def test_campaign_resume_preserves_roi_and_reports_parent_lineage(tmp_path, monkeypatch):
    args, manifest, config = campaign_inputs(tmp_path)
    import models.keypoints.ablation as ablation
    import models.keypoints.benchmark as benchmark
    import models.keypoints.checkpoints as checkpoints
    import models.keypoints.experiment as experiment
    import models.keypoints.model as model_module

    class LightweightModel:
        def __init__(self, configuration):
            self.config = configuration

        def to(self, device):
            return self

    def train(model, root, subset, output, *, phase, epochs, **options):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        for name in ("best.pt", "last.pt"):
            (output / name).write_bytes(f"{output.parent.name}-{phase}".encode())
        (output / "completed.json").write_text("{}")
        return output / "best.pt"

    def load(path, *, model=None, **options):
        path = Path(path)
        if model is None:
            run = path.parent.parent.name
            configs = ablation.ablation_configs(config, level=.15, margin=.5)
            model = LightweightModel(configs[run])
        phase = path.parent.name
        payload = {"split_ids": {s: manifest[s] for s in ("train", "val", "test")},
                   "keypoint_names": manifest["keypoint_names"],
                   "phase": phase,
                   "training": {"max_samples": args.max_samples,
                                "seed": args.seed, "decoder": args.decoder,
                                "epochs": getattr(args, phase + "_epochs")}}
        return model, payload

    def metrics(*a, **kw):
        return {"global_argmax": {"mean": 3.}, "global_subpixel": {"mean": 3.},
                "local_argmax": {"mean": 2.}, "local_subpixel": {"mean": 2.}}

    monkeypatch.setattr(model_module, "RGBDKeypointModel", LightweightModel)
    monkeypatch.setattr(experiment, "train_run", train)
    monkeypatch.setattr(checkpoints, "load_checkpoint", load)
    monkeypatch.setattr(experiment, "_dataset", lambda *a, **kw: [])
    monkeypatch.setattr(experiment, "evaluate_model", metrics)
    monkeypatch.setattr(benchmark, "benchmark_model", lambda *a, **kw: {})
    monkeypatch.setattr(ablation, "initialize_local", lambda *a: None)
    monkeypatch.setattr(ablation, "select_region_config", lambda *a, **kw: selected_region())
    run_ablation(args, manifest, config)
    report = json.loads((Path(args.output) / "report.json").read_text())
    assert len(report["a1_checkpoint_sha256"]) == 64
    assert report["local_init_seed"] == args.seed
    assert report["region_selection"] == selected_region()
    for run in ("A2", "A3", "A4"):
        assert report["runs"][run]["a1_checkpoint_sha256"] == report["a1_checkpoint_sha256"]
        assert report["runs"][run]["local_init_seed"] == args.seed
        assert report["runs"][run]["region"] == {"level": .15, "margin_ratio": .5, "overlap_ratio": .5}
    monkeypatch.setattr(ablation, "select_region_config", lambda *a, **kw: pytest.fail("locked ROI was recomputed"))
    monkeypatch.setattr(experiment, "train_run", lambda *a, **kw: pytest.fail("completed campaign retrained"))
    run_ablation(args, manifest, config)
    assert json.loads((Path(args.output) / "report.json").read_text())["region_selection"] == selected_region()
