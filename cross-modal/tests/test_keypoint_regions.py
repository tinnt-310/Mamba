import numpy as np
import pytest
import torch

from models.keypoints.regions import (
    Box,
    extract_regions,
    make_patch_boxes,
    peak_component_region,
    scale_box,
)


def test_region_uses_peak_component_four_connectivity_and_row_major_tie():
    heatmap = torch.zeros(7, 8)
    heatmap[1:3, 1:3] = 4
    heatmap[3, 3] = 3  # Diagonal touch must not join the component.
    heatmap[5, 6] = 4
    assert peak_component_region(heatmap, margin_ratio=0) == Box(1, 1, 3, 3)


def test_region_margin_uses_ceil_and_clamps_at_border():
    heatmap = np.zeros((7, 8), dtype=np.float32)
    heatmap[0:2, 0:3] = 1
    assert peak_component_region(heatmap) == Box(0, 0, 4, 3)


@pytest.mark.parametrize("value", [-4.0, 0.0, 8.0])
def test_constant_maps_use_full_grid(value):
    assert peak_component_region(torch.full((3, 5), value)) == Box(0, 0, 5, 3)


def test_tiny_dynamic_range_uses_full_grid():
    heatmap = torch.zeros(3, 4)
    heatmap[1, 2] = 5e-7
    assert peak_component_region(heatmap) == Box(0, 0, 4, 3)


def test_negative_map_is_minmax_normalized():
    heatmap = torch.full((5, 6), -10.0)
    heatmap[2, 3] = -2
    assert peak_component_region(heatmap, margin_ratio=0) == Box(3, 2, 4, 3)


def test_finite_large_response_range_retains_component_threshold():
    heatmap = torch.full((5, 6), -2e38)
    heatmap[2, 2:4] = 2e38
    assert peak_component_region(heatmap, margin_ratio=0) == Box(2, 2, 4, 3)


def test_wider_gaussian_produces_wider_region():
    yy, xx = torch.meshgrid(torch.arange(25), torch.arange(31), indexing="ij")
    narrow = torch.exp(-((xx - 15) ** 2 + (yy - 12) ** 2) / 2)
    wide = torch.exp(-((xx - 15) ** 2 + (yy - 12) ** 2) / 18)
    assert peak_component_region(wide).width > peak_component_region(narrow).width
    assert peak_component_region(wide).height > peak_component_region(narrow).height


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_map_is_rejected(bad):
    heatmap = torch.zeros(3, 4)
    heatmap[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        peak_component_region(heatmap)


def test_scale_box_maps_edges_outward_on_odd_grids():
    assert scale_box(Box(1, 1, 7, 5), (7, 11), (3, 5)) == Box(0, 0, 4, 3)
    assert scale_box(Box(10, 6, 11, 7), (7, 11), (3, 5)) == Box(4, 2, 5, 3)


def test_five_patches_have_native_shape_and_cover_region():
    region = Box(3, 2, 10, 7)
    patches = make_patch_boxes(region)
    assert patches == [
        Box(3, 2, 8, 6), Box(5, 2, 10, 6),
        Box(3, 3, 8, 7), Box(5, 3, 10, 7), Box(4, 2, 9, 6),
    ]
    covered = np.zeros((5, 7), dtype=bool)
    for patch in patches:
        covered[patch.y1 - 2:patch.y2 - 2, patch.x1 - 3:patch.x2 - 3] = True
    assert covered.all()


def test_small_regions_are_stably_deduplicated_and_one_layout_is_full_region():
    assert make_patch_boxes(Box(4, 7, 5, 8)) == [Box(4, 7, 5, 8)]
    assert make_patch_boxes(Box(4, 7, 6, 9)) == [Box(4, 7, 6, 9)]
    assert make_patch_boxes(Box(0, 0, 6, 4), layout="one") == [Box(0, 0, 6, 4)]


def test_extract_regions_preserves_batch_keypoint_indices_and_detaches_maps():
    heatmaps = torch.zeros(2, 2, 5, 7, requires_grad=True)
    with torch.no_grad():
        heatmaps[0, 0, 2, 3] = 1
        heatmaps[1, 1, 0, 6] = 1
    regions = extract_regions(heatmaps, (3, 4), margin_ratio=0)
    assert [(r.batch_index, r.keypoint_index) for r in regions] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert regions[0].global_box == Box(3, 2, 4, 3)
    assert regions[0].feature_box == Box(1, 1, 3, 2)
    assert regions[1].feature_box == Box(0, 0, 4, 3)
    assert regions[3].feature_box == Box(3, 0, 4, 1)
    assert heatmaps.grad is None


@pytest.mark.parametrize("kwargs", [{"level": 0}, {"level": 1.1}, {"margin_ratio": -1}])
def test_invalid_region_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        peak_component_region(torch.zeros(3, 3), **kwargs)


@pytest.mark.parametrize("kwargs", [{"layout": "invalid"}, {"overlap_ratio": -0.1}, {"overlap_ratio": 1.1}])
def test_invalid_patch_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        make_patch_boxes(Box(0, 0, 4, 5), **kwargs)


def test_empty_or_invalid_grids_and_boxes_rejected():
    with pytest.raises(ValueError):
        peak_component_region(torch.empty(0, 3))
    with pytest.raises(ValueError):
        Box(1, 0, 1, 2)
    with pytest.raises(ValueError):
        scale_box(Box(0, 0, 1, 1), (0, 3), (3, 3))
