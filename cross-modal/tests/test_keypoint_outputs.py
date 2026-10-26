import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from models.keypoints.outputs import aggregate_region_statistics, compute_metrics, save_prediction_artifact, visualize_artifact


def artifact_fixture(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(rgb_path)
    sample = {"frame_id": "000001", "rgb": torch.zeros(3, 4, 6), "rgb_path": str(rgb_path), "original_size": torch.tensor([8, 12]),
              "points": torch.tensor([[2.2, 1.6]]), "original_points": torch.tensor([[4.9, 3.7]]), "weights": torch.ones(1)}
    yy, xx = torch.meshgrid(torch.arange(4), torch.arange(6), indexing="ij")
    global_maps = torch.exp(-((xx - 2.2).square() + (yy - 1.6).square()) / 2)[None, None]
    box = SimpleNamespace(x1=0, y1=0, x2=3, y2=2)
    global_box = SimpleNamespace(x1=0, y1=0, x2=6, y2=4)
    region = SimpleNamespace(batch_index=0, keypoint_index=0, global_box=global_box, feature_box=box, patch_boxes=[box])
    local = SimpleNamespace(heatmaps=torch.tensor([[[[-9., -4., -6.], [-8., -7., 999.]]]]),
                            support=torch.tensor([[[[True, True, True], [True, True, False]]]]),
                            coverage=torch.tensor([[[[1., 1., 1.], [1., 1., 0.]]]]), regions=[region], patches=[])
    return sample, SimpleNamespace(global_heatmaps=global_maps, local=local)


def test_artifact_keeps_raw_maps_and_both_coordinate_decoders(tmp_path):
    sample, stage = artifact_fixture(tmp_path)
    paths = save_prediction_artifact(tmp_path / "frame", sample, stage, {"config_version": 2}, ["left"])
    arrays = np.load(paths["npz"], allow_pickle=False)
    np.testing.assert_array_equal(arrays["global_heatmaps"], stage.global_heatmaps[0].numpy())
    np.testing.assert_array_equal(arrays["support"], stage.local.support[0].numpy())
    np.testing.assert_allclose(arrays["global_subpixel_input"], [[2.2, 1.6]], atol=1e-5)
    np.testing.assert_allclose(arrays["global_subpixel_original"], [[4.9, 3.7]], atol=1e-5)
    np.testing.assert_allclose(arrays["local_argmax_grid"], [[1, 0]])
    np.testing.assert_allclose(arrays["local_argmax_input"], [[2.5, .5]])
    metadata = json.loads(open(paths["json"]).read())
    assert metadata["frame_id"] == "000001"
    assert metadata["regions"][0]["patch_boxes"] == [[0, 0, 3, 2]]
    assert metadata["config"]["config_version"] == 2
    assert metadata["coordinates"]["global"]["subpixel"]["original"][0] == pytest.approx([4.9, 3.7], abs=1e-5)


def test_visualization_uses_saved_coordinates_and_source_image(tmp_path):
    sample, stage = artifact_fixture(tmp_path)
    paths = save_prediction_artifact(tmp_path / "frame", sample, stage, {}, ["left"])
    image_path = visualize_artifact(paths["npz"], tmp_path / "visualization.png")
    assert Image.open(image_path).width > 12


def test_metrics_exclude_missing_labels_and_report_keypoint_errors():
    predictions = np.array([[[3, 4], [30, 40]], [[0, 0], [0, 5]]], dtype=float)
    result = compute_metrics(predictions, np.zeros_like(predictions), np.array([[1, 0], [1, 1]]), ["left", "right"])
    assert result["count"] == 3
    assert result["mean"] == pytest.approx(10 / 3)
    assert result["median"] == 5
    assert result["p95"] == 5
    assert result["per_keypoint"]["left"]["mean"] == 2.5
    assert result["per_keypoint"]["right"]["mean"] == 5


def test_metrics_handle_no_labels_and_reject_nonfinite_valid_predictions():
    result = compute_metrics(np.zeros((1, 1, 2)), np.zeros((1, 1, 2)), np.zeros((1, 1)))
    assert result["count"] == 0 and result["mean"] is None
    with pytest.raises(ValueError):
        compute_metrics(np.array([[[np.nan, 0]]]), np.zeros((1, 1, 2)), np.ones((1, 1)))


def test_region_statistics_count_recall_sequence_lengths_and_fallbacks(tmp_path):
    sample, stage = artifact_fixture(tmp_path)
    paths = save_prediction_artifact(tmp_path / "frame", sample, stage, {}, ["left"])
    stats = aggregate_region_statistics([paths["json"]])
    assert stats["roi_recall"] == 1
    assert stats["region_count"] == 1
    assert stats["patch_count"] == 1
    assert stats["mean_region_area"] == 6
    assert stats["mean_sequence_length"] == stats["max_sequence_length"] == 6
    assert stats["local_subpixel_axis_rate"] == 0
    assert stats["local_fallback_axis_rate"] == 1


def test_artifact_rejects_batch_sizes_above_one(tmp_path):
    sample, stage = artifact_fixture(tmp_path)
    stage.global_heatmaps = stage.global_heatmaps.expand(2, -1, -1, -1)
    with pytest.raises(ValueError):
        save_prediction_artifact(tmp_path / "frame", sample, stage, {}, ["left"])
