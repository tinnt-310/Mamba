"""Dataset tests run on Slurm, including all synthetic image generation."""
import json

import numpy as np
import pytest
import torch
from PIL import Image

from models.keypoints.data import RGBDKeypointDataset, create_subset_manifest, load_manifest


def write_frame(root, frame_id="000000", *, visibility=2, points=None):
    for directory in ("rgb", "depth_sensor", "poses", "annotations"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((4, 8, 3), dtype=np.uint8)).save(root / "rgb" / f"{frame_id}.png")
    np.save(root / "depth_sensor" / f"{frame_id}.npy", np.arange(1, 33, dtype=np.float32).reshape(4, 8))
    (root / "poses" / f"{frame_id}.json").write_text(json.dumps({"K": [[10, 0, 3.5], [0, 12, 1.5], [0, 0, 1]]}))
    annotation = {"frame_id": frame_id, "paths": {"rgb": f"rgb/{frame_id}.png", "depth_sensor": f"depth_sensor/{frame_id}.npy", "poses": f"poses/{frame_id}.json"},
                  "keypoints_2d": points or [{"name": "left", "uv": [2.5, 1.5], "visibility": visibility}, {"name": "right", "uv": [7, 3], "visibility": 0}]}
    (root / "annotations" / f"{frame_id}.json").write_text(json.dumps(annotation))
    return annotation


def write_source_splits(root):
    ids = [f"{i:06d}" for i in range(10)]
    for split, values in (("train", ids[:8]), ("val", ids[8:9]), ("test", ids[9:])):
        (root / f"{split}.txt").write_text("\n".join(values) + "\n")
    for frame_id in ids:
        write_frame(root, frame_id)
    return ids


def test_loader_keeps_metric_depth_and_center_transforms(tmp_path):
    write_frame(tmp_path)
    sample = RGBDKeypointDataset(tmp_path, ["000000"], ["left", "right"], input_size=(2, 4), heatmap_size=(2, 4))[0]
    assert sample["frame_id"] == "000000"
    assert sample["rgb"].shape == (3, 2, 4)
    assert sample["depth"].dtype == torch.float32
    torch.testing.assert_close(sample["depth"], torch.tensor([[[1., 3., 5., 7.], [17., 19., 21., 23.]]]))
    torch.testing.assert_close(sample["K"], torch.tensor([[5., 0., 1.5], [0., 6., .5], [0., 0., 1.]]))
    torch.testing.assert_close(sample["points"][0], torch.tensor([1., .5]))
    torch.testing.assert_close(sample["original_points"][0], torch.tensor([2.5, 1.5]))
    assert sample["original_size"].tolist() == [4, 8]
    assert sample["weights"].tolist() == [1., 0.]
    assert sample["target"][0].max() > 0
    assert torch.count_nonzero(sample["target"][1]) == 0


def test_loader_matches_labels_by_name_and_masks_missing_outside_points(tmp_path):
    write_frame(tmp_path, points=[{"name": "right", "uv": [20, 1], "visibility": 2}, {"name": "left", "uv": [2.5, 1.5], "visibility": 1}])
    sample = RGBDKeypointDataset(tmp_path, ["000000"], ["left", "right", "missing"], input_size=(2, 4))[0]
    assert sample["weights"].tolist() == [1., 0., 0.]
    assert sample["visibility"].tolist() == [1, 2, 0]
    assert torch.count_nonzero(sample["target"][1:]) == 0


@pytest.mark.parametrize("change", ["wrong_id", "wrong_path", "depth_shape", "bad_intrinsics", "duplicate_label"])
def test_loader_rejects_mispaired_or_malformed_frames(tmp_path, change):
    annotation = write_frame(tmp_path)
    if change == "wrong_id":
        annotation["frame_id"] = "000001"
    elif change == "wrong_path":
        annotation["paths"]["rgb"] = "../rgb/000000.png"
    elif change == "depth_shape":
        np.save(tmp_path / "depth_sensor/000000.npy", np.ones((3, 8), dtype=np.float32))
    elif change == "bad_intrinsics":
        (tmp_path / "poses/000000.json").write_text(json.dumps({"K": [[0, 0, 0], [0, 1, 0], [0, 0, 1]]}))
    else:
        annotation["keypoints_2d"].append(annotation["keypoints_2d"][0])
    (tmp_path / "annotations/000000.json").write_text(json.dumps(annotation))
    with pytest.raises(ValueError):
        RGBDKeypointDataset(tmp_path, ["000000"], ["left", "right"])[0]


def test_subset_preserves_source_splits_and_persisted_ids(tmp_path):
    ids = write_source_splits(tmp_path)
    manifest = create_subset_manifest(tmp_path, tmp_path / "manifest.json", limit=10)
    assert [len(manifest[key]) for key in ("train", "val", "test")] == [8, 1, 1]
    assert set(manifest["train"]) == set(ids[:8])
    assert manifest["keypoint_names"] == ["left", "right"]
    assert load_manifest(tmp_path / "manifest.json") == manifest
    again = create_subset_manifest(tmp_path, tmp_path / "again.json", limit=10)
    assert again == manifest


@pytest.mark.parametrize("change", ["overlap", "traversal", "wrong_membership", "oversized"])
def test_manifest_rejects_leakage_traversal_and_limits(tmp_path, change):
    write_source_splits(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = create_subset_manifest(tmp_path, path, limit=10)
    if change == "overlap":
        manifest["val"] = [manifest["train"][0]]
    elif change == "traversal":
        manifest["train"][0] = "../000000"
    elif change == "wrong_membership":
        manifest["train"][0], manifest["val"][0] = manifest["val"][0], manifest["train"][0]
    else:
        manifest["limit"] = 5001
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_subset_rejects_above_user_budget_and_duplicate_dataset_ids(tmp_path):
    with pytest.raises(ValueError):
        create_subset_manifest(tmp_path, tmp_path / "manifest.json", limit=5001)
    with pytest.raises(ValueError):
        RGBDKeypointDataset(tmp_path, ["000000", "000000"], ["left"])


def test_manifest_rejects_different_dataset_root(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    write_source_splits(root)
    path = tmp_path / "manifest.json"
    create_subset_manifest(root, path, limit=10)
    with pytest.raises(ValueError):
        load_manifest(path, root=tmp_path / "another_dataset")


def test_manifest_detects_source_split_changes_after_selection(tmp_path):
    write_source_splits(tmp_path)
    path = tmp_path / "manifest.json"
    create_subset_manifest(tmp_path, path, limit=10)
    with (tmp_path / "train.txt").open("a") as file:
        file.write("000010\n")
    with pytest.raises(ValueError):
        load_manifest(path)


def test_manifest_rejects_changed_80_10_10_subset_counts(tmp_path):
    write_source_splits(tmp_path)
    write_frame(tmp_path, "000010")
    (tmp_path / "val.txt").write_text("000008\n000010\n")
    path = tmp_path / "manifest.json"
    manifest = create_subset_manifest(tmp_path, path, limit=10)
    manifest["train"].pop()
    manifest["val"] = ["000008", "000010"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_manifest_real_dataset_is_exactly_5000_with_source_membership(tmp_path):
    from pathlib import Path
    root = Path("/work/tinnt/BlenderProc/output/keypoints_bbox_50000_job_1634")
    if not root.exists():
        pytest.skip("cluster dataset unavailable")
    manifest = create_subset_manifest(root, tmp_path / "subset.json")
    assert [len(manifest[key]) for key in ("train", "val", "test")] == [4000, 500, 500]
    assert len(set(manifest["train"] + manifest["val"] + manifest["test"])) == 5000
    assert len(manifest["keypoint_names"]) == 9
    assert load_manifest(tmp_path / "subset.json") == manifest


def test_full_subset_validation_checks_every_selected_frame(tmp_path):
    from models.keypoints.validation import validate_subset
    write_source_splits(tmp_path)
    manifest=create_subset_manifest(tmp_path,tmp_path/'manifest.json',limit=10)
    report=validate_subset(tmp_path,manifest)
    assert report['images_checked']==10 and report['passed']
    assert report['counts']=={'train':8,'val':1,'test':1}
    assert report['valid_labels_per_keypoint']=={'left':10,'right':0}
    (tmp_path/'rgb'/f"{manifest['test'][0]}.png").unlink()
    with pytest.raises(FileNotFoundError): validate_subset(tmp_path,manifest)
