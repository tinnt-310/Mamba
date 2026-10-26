import pytest
import torch

from models.fusion.ssm import SimpleSelectiveSSM
from models.keypoints.local_refinement import (
    AdaptiveLocalRefinement,
    FourDirectionLocalMamba,
    PatchPrediction,
    merge_patch_predictions,
)
from models.keypoints.regions import Box


@pytest.mark.parametrize("hw", [(1, 1), (1, 7), (5, 1), (3, 7), (23, 40)])
def test_local_mamba_keeps_native_patch_shape_and_finite_gradients(hw):
    torch.manual_seed(8)
    block = FourDirectionLocalMamba(channels=4, state_dim=2)
    x = torch.randn(2, 4, *hw, requires_grad=True)
    out = block(x)
    assert out.shape == x.shape
    out.square().mean().backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for p in block.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert block.ssm.x_proj.weight.grad.abs().sum() > 0
    if hw[0] * hw[1] > 1:
        assert block.ssm.A_log.grad.abs().sum() > 0


def test_one_shared_ssm_receives_four_distinct_spatial_directions():
    block = FourDirectionLocalMamba(channels=2, state_dim=2)
    calls = []
    handle = block.ssm.register_forward_pre_hook(
        lambda module, args: calls.append(args[0].detach().clone()))
    x = torch.arange(12.).reshape(1, 2, 2, 3)
    try:
        block(x)
    finally:
        handle.remove()
    assert sum(isinstance(m, SimpleSelectiveSSM) for m in block.modules()) == 1
    assert len(calls) == 1
    assert calls[0].shape == (4, 2, 6)
    assert calls[0][:, 0].tolist() == [
        [0, 1, 2, 3, 4, 5], [0, 3, 1, 4, 2, 5],
        [5, 4, 3, 2, 1, 0], [5, 2, 4, 1, 3, 0],
    ]


def test_overlap_merge_averages_negative_predictions_and_preserves_gradients():
    a = torch.full((1, 1, 2, 2), -2., requires_grad=True)
    b = torch.full((1, 1, 2, 2), -6., requires_grad=True)
    patches = [PatchPrediction(0, 0, Box(0, 0, 2, 2), a),
               PatchPrediction(0, 0, Box(1, 0, 3, 2), b)]
    maps, support, coverage = merge_patch_predictions(
        patches, batch_size=1, num_keypoints=2, feature_hw=(3, 4))
    torch.testing.assert_close(maps[0, 0], torch.tensor([
        [-2., -4., -6., 0.], [-2., -4., -6., 0.], [0., 0., 0., 0.]]))
    assert coverage[0, 0].tolist() == [[1, 2, 1, 0], [1, 2, 1, 0], [0, 0, 0, 0]]
    assert support.dtype == torch.bool
    assert torch.equal(support, coverage > 0)
    assert not support[0, 1].any()
    maps.sum().backward()
    torch.testing.assert_close(a.grad, torch.tensor([[[[1., .5], [1., .5]]]]))
    torch.testing.assert_close(b.grad, torch.tensor([[[[.5, 1.], [.5, 1.]]]]))


def test_overlap_merge_keeps_large_finite_mean_finite():
    prediction = torch.full((1, 1, 2, 2), 2e38, requires_grad=True)
    patches = [PatchPrediction(0, 0, Box(0, 0, 2, 2), prediction) for _ in range(5)]
    maps, support, coverage = merge_patch_predictions(
        patches, batch_size=1, num_keypoints=1, feature_hw=(2, 2))
    assert torch.isfinite(maps).all()
    torch.testing.assert_close(maps, prediction)
    maps.mean().backward()
    assert torch.isfinite(prediction.grad).all()


def test_adaptive_native_shape_groups_and_microbatch_bound():
    torch.manual_seed(14)
    layer = AdaptiveLocalRefinement(4, 3, enabled=True, microbatch_size=2,
                                    state_dim=2, embedding=True)
    f1 = torch.randn(2, 4, 5, 8, requires_grad=True)
    maps = torch.zeros(2, 3, 10, 16, requires_grad=True)
    with torch.no_grad():
        maps[0, 0, 2, 3] = 1
        maps[0, 1, 2:5, 3:9] = 1
        maps[1, 0, 8, 15] = 1
    shapes = []
    handle = layer.local_mamba.register_forward_pre_hook(
        lambda module, args: shapes.append(tuple(args[0].shape)))
    try:
        out = layer(maps, f1, return_patches=True)
    finally:
        handle.remove()
    assert out.heatmaps.shape == (2, 3, 5, 8)
    assert torch.equal(out.support, out.coverage > 0)
    assert out.patches and len(out.regions) == 6
    assert all(n <= 2 for n, c, h, w in shapes)
    observed_shapes = {(h, w) for n, c, h, w in shapes}
    expected_shapes = {(p.box.height, p.box.width) for p in out.patches}
    assert observed_shapes == expected_shapes and len(observed_shapes) > 1
    for p in out.patches:
        assert p.prediction.shape == (1, 1, p.box.height, p.box.width)
    out.heatmaps.square().mean().backward()
    assert maps.grad is None
    assert f1.grad is not None and f1.grad.abs().sum() > 0
    assert layer.keypoint_embedding.weight.grad.abs().sum() > 0


def test_embedding_off_and_patch_return_off_still_have_differentiable_merge():
    layer = AdaptiveLocalRefinement(4, 2, state_dim=2, embedding=False, layout="one")
    f1 = torch.randn(1, 4, 3, 5, requires_grad=True)
    out = layer(torch.ones(1, 2, 6, 10), f1)
    assert out.patches == [] and out.support.all()
    assert layer.keypoint_embedding is None
    out.heatmaps.sum().backward()
    assert torch.isfinite(f1.grad).all() and f1.grad.abs().sum() > 0


def test_opt_in_profile_reports_region_scan_and_merge_timings():
    layer = AdaptiveLocalRefinement(4, 1, state_dim=2, embedding=False)
    assert layer.profile is False
    layer.profile = True
    layer(torch.ones(1, 1, 6, 10), torch.randn(1, 4, 3, 5))
    assert set(layer.last_timings) == {"regions_seconds", "scan_seconds", "merge_seconds"}
    assert all(value >= 0 for value in layer.last_timings.values())


def test_full_overlap_deduplicates_to_one_native_region_patch():
    layer = AdaptiveLocalRefinement(4, 1, state_dim=2, overlap_ratio=1.)
    out = layer(torch.ones(1, 1, 6, 10), torch.randn(1, 4, 3, 5), return_patches=True)
    assert len(out.patches) == 1
    assert out.patches[0].box == Box(0, 0, 5, 3)
    assert out.support.all() and out.coverage.eq(1).all()


@pytest.mark.parametrize("options", [
    {"directions": 1}, {"microbatch_size": 0}, {"microbatch_size": 9},
    {"layout": "invalid"}, {"state_dim": 0}, {"backend": "invalid"},
])
def test_local_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        AdaptiveLocalRefinement(4, 2, **options)
