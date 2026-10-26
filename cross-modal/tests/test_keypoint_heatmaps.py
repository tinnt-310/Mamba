import numpy as np
import pytest
import torch

from models.keypoints.heatmaps import decode_heatmaps, gaussian_heatmaps, transform_points


def test_point_center_transform_round_trip_and_literal_coordinates():
    points = torch.tensor([[0.0, 0.0], [319.0, 179.0], [111.25, 61.75]], dtype=torch.float64)
    smaller = transform_points(points, (180, 320), (45, 80))
    torch.testing.assert_close(smaller[:2], torch.tensor([[-0.375, -0.375], [79.375, 44.375]], dtype=torch.float64))
    torch.testing.assert_close(transform_points(smaller, (45, 80), (180, 320)), points, atol=1e-5, rtol=0)
    array = points.numpy()
    assert isinstance(transform_points(array, (180, 320), (45, 80)), np.ndarray)
    np.testing.assert_allclose(transform_points(transform_points(array, (180, 320), (45, 80)), (45, 80), (180, 320)), array, atol=1e-5)


def test_gaussian_unbatched_and_batched_missing_keypoint_targets():
    points = torch.tensor([[2.0, 1.0], [float("nan"), float("nan")]], dtype=torch.float64)
    result = gaussian_heatmaps(points, (3, 5), 1, weights=torch.tensor([1.0, 0.0]))
    assert result.shape == (2, 3, 5)
    assert result.dtype == torch.float64
    assert result[0, 1, 2] == 1
    assert result[0, 1, 1].item() == pytest.approx(np.exp(-0.5))
    assert torch.count_nonzero(result[1]) == 0
    assert gaussian_heatmaps(points[None], (3, 5), 1, weights=torch.tensor([[1.0, 0.0]])).shape == (1, 2, 3, 5)


def test_gaussian_rejects_invalid_visible_points_and_sigma():
    with pytest.raises(ValueError, match="finite"):
        gaussian_heatmaps(torch.tensor([[float("nan"), 0.0]]), (3, 4), 1)
    with pytest.raises(ValueError):
        gaussian_heatmaps(torch.zeros(1, 2), (3, 4), 0)


def test_subpixel_recovers_fractional_gaussian_center():
    yy, xx = torch.meshgrid(torch.arange(7, dtype=torch.float64), torch.arange(9, dtype=torch.float64), indexing="ij")
    heatmap = torch.exp(-((xx - 4.31) ** 2 + (yy - 2.74) ** 2) / (2 * 1.1 ** 2))
    decoded = decode_heatmaps(heatmap[None, None])
    torch.testing.assert_close(decoded.coords[0, 0], torch.tensor([4.31, 2.74], dtype=torch.float64), atol=1e-3, rtol=0)
    assert decoded.argmax.tolist() == [[[4, 3]]]
    assert decoded.subpixel_used.tolist() == [[[True, True]]]
    assert decoded.peak_response.item() == heatmap[3, 4]


def test_masked_argmax_excludes_uncovered_zero_for_negative_responses():
    heatmap = torch.zeros(1, 1, 4, 5)
    support = torch.zeros_like(heatmap, dtype=torch.bool)
    heatmap[0, 0, 1:3, 2:4] = torch.tensor([[-5.0, -4.0], [-3.0, -2.0]])
    support[0, 0, 1:3, 2:4] = True
    decoded = decode_heatmaps(heatmap, support=support)
    assert decoded.coords.tolist() == [[[3.0, 2.0]]]
    assert decoded.peak_response.item() == -2
    assert not decoded.subpixel_used.any()


def test_subpixel_falls_back_per_axis_at_support_boundary():
    yy, xx = torch.meshgrid(torch.arange(5, dtype=torch.float64), torch.arange(7, dtype=torch.float64), indexing="ij")
    heatmap = torch.exp(-((xx - 3.25) ** 2 + (yy - 2.2) ** 2) / 2)[None, None]
    support = torch.ones_like(heatmap, dtype=torch.bool)
    support[0, 0, 2, 2] = False
    decoded = decode_heatmaps(heatmap, support=support)
    torch.testing.assert_close(decoded.coords[0, 0], torch.tensor([3.0, 2.2], dtype=torch.float64), atol=1e-3, rtol=0)
    assert decoded.subpixel_used.tolist() == [[[False, True]]]


def test_argmax_mode_ties_border_and_flat_fallback():
    heatmaps = torch.ones(2, 1, 3, 4)
    heatmaps[1, 0, 0, 3] = 2
    decoded = decode_heatmaps(heatmaps)
    assert decoded.argmax.tolist() == [[[0, 0]], [[3, 0]]]
    assert decoded.coords.tolist() == [[[0.0, 0.0]], [[3.0, 0.0]]]
    assert not decoded.subpixel_used.any()
    assert not decode_heatmaps(heatmaps, subpixel=False).subpixel_used.any()


def test_decoder_rejects_empty_support_and_nonfinite_maps():
    with pytest.raises(ValueError, match="support"):
        decode_heatmaps(torch.zeros(1, 1, 3, 4), support=torch.zeros(1, 1, 3, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="finite"):
        decode_heatmaps(torch.full((1, 1, 3, 4), float("nan")))
    with pytest.raises(ValueError):
        decode_heatmaps(torch.empty(1, 1, 0, 4))
