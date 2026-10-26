import math
from types import SimpleNamespace

import pytest
import torch

from models.keypoints.local_losses import local_patch_loss, weighted_heatmap_loss
from models.keypoints.regions import Box


def patch(batch, keypoint, box, values):
    return SimpleNamespace(batch_index=batch, keypoint_index=keypoint, box=box, prediction=values)


def test_weighted_global_loss_uses_one_plus_positive_minus_one_times_target():
    prediction = torch.zeros(1, 1, 1, 2, requires_grad=True)
    target = torch.tensor([[[[1.0, 0.5]]]])
    loss = weighted_heatmap_loss(prediction, target, torch.ones(1, 1), positive_weight=3)
    assert loss.item() == pytest.approx(1.75)  # (3*1 + 2*.25)/2
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[[[-3.0, -1.0]]]]))


def test_global_loss_masks_missing_labels_and_zero_loss_keeps_graph():
    prediction = torch.tensor([[[[2.0]], [[100.0]]]], requires_grad=True)
    assert weighted_heatmap_loss(prediction, torch.zeros_like(prediction), torch.tensor([[1.0, 0.0]])).item() == 4
    loss = weighted_heatmap_loss(prediction, torch.zeros_like(prediction), torch.zeros(1, 2))
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert prediction.grad is not None and torch.count_nonzero(prediction.grad) == 0


def test_local_loss_balances_pixels_then_patches_then_keypoints():
    a = torch.tensor([[[[1.0]]]], requires_grad=True)
    b = torch.tensor([[[[3.0, 3.0], [3.0, 3.0]]]], requires_grad=True)
    c = torch.tensor([[[[2.0]]]], requires_grad=True)
    predictions = [patch(0, 0, Box(0, 0, 1, 1), a), patch(0, 0, Box(0, 0, 2, 2), b), patch(0, 1, Box(0, 0, 1, 1), c)]
    points = torch.tensor([[[3.0, 3.0], [3.0, 3.0]]])
    loss = local_patch_loss(predictions, points, torch.ones(1, 2), (4, 4))
    assert loss.item() == pytest.approx(4.5)  # ((1+9)/2 + 4)/2
    loss.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in (a, b, c))


def test_local_gaussian_target_comes_from_annotation_in_patch_coordinates():
    prediction = torch.zeros(1, 1, 1, 1, requires_grad=True)
    predictions = [patch(0, 0, Box(2, 3, 3, 4), prediction)]
    loss = local_patch_loss(predictions, torch.tensor([[[2.0, 3.0]]]), torch.ones(1, 1), (5, 6), positive_weight=3)
    assert loss.item() == 3


@pytest.mark.parametrize("x,included", [(-0.5, True), (-0.50001, False), (0.49999, True), (0.5, False)])
def test_local_target_uses_half_open_pixel_center_footprint(x, included):
    predictions = [patch(0, 0, Box(0, 0, 1, 1), torch.zeros(1, 1, 1, 1, dtype=torch.float64))]
    loss = local_patch_loss(predictions, torch.tensor([[[x, 0.0]]], dtype=torch.float64), torch.ones(1, 1), (3, 3), positive_weight=3)
    # exp(-x²/2) target, loss = (1+2*T)*T² for an included point.
    want = math.exp(-x * x) + 2 * math.exp(-1.5 * x * x) if included else 0.0
    assert loss.item() == pytest.approx(want, abs=1e-8)


def test_local_missing_annotations_zero_loss_preserves_all_patch_graphs():
    values = [torch.randn(1, 1, 1, 1, requires_grad=True), torch.randn(1, 1, 2, 2, requires_grad=True)]
    predictions = [patch(0, i, Box(0, 0, value.shape[-1], value.shape[-2]), value) for i, value in enumerate(values)]
    points = torch.full((1, 2, 2), float("nan"))
    loss = local_patch_loss(predictions, points, torch.zeros(1, 2), (3, 3))
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert all(value.grad is not None and torch.count_nonzero(value.grad) == 0 for value in values)


def test_local_missing_keypoint_does_not_change_valid_loss():
    predictions = [patch(0, 0, Box(0, 0, 1, 1), torch.tensor([[[[2.0]]]])), patch(0, 1, Box(0, 0, 1, 1), torch.tensor([[[[90.0]]]]))]
    assert local_patch_loss(predictions, torch.tensor([[[2.0, 2.0], [float("nan"), float("nan")]]]), torch.tensor([[1.0, 0.0]]), (3, 3)).item() == 4


def test_local_positive_annotation_weights_have_equal_keypoint_votes():
    predictions = [patch(0, 0, Box(0, 0, 1, 1), torch.tensor([[[[1.0]]]])), patch(0, 1, Box(0, 0, 1, 1), torch.tensor([[[[3.0]]]]))]
    assert local_patch_loss(predictions, torch.tensor([[[2.0, 2.0], [2.0, 2.0]]]), torch.tensor([[0.01, 1.0]]), (3, 3)).item() == 5


def test_unlabeled_large_finite_predictions_produce_finite_graph_zero():
    prediction = torch.full((1, 1, 2, 2), 2e38, requires_grad=True)
    global_loss = weighted_heatmap_loss(prediction, torch.zeros_like(prediction), torch.zeros(1, 1))
    local_loss = local_patch_loss([patch(0, 0, Box(0, 0, 2, 2), prediction)], torch.full((1, 1, 2), float("nan")), torch.zeros(1, 1), (2, 2))
    assert torch.isfinite(global_loss) and global_loss.item() == 0
    assert torch.isfinite(local_loss) and local_loss.item() == 0
    (global_loss + local_loss).backward()
    assert torch.isfinite(prediction.grad).all() and torch.count_nonzero(prediction.grad) == 0


def test_global_masking_happens_before_invalid_keypoint_error_can_overflow():
    prediction = torch.tensor([[[[2.0]], [[2e38]]]], requires_grad=True)
    loss = weighted_heatmap_loss(prediction, torch.zeros_like(prediction), torch.tensor([[1.0, 0.0]]))
    assert torch.isfinite(loss) and loss.item() == 4
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[[[4.0]], [[0.0]]]]))


def test_loss_rejects_nonfinite_predictions_invalid_weights_and_shapes():
    with pytest.raises(ValueError, match="finite"):
        weighted_heatmap_loss(torch.full((1, 1, 1, 1), float("nan")), torch.zeros(1, 1, 1, 1), torch.ones(1, 1))
    with pytest.raises(ValueError):
        weighted_heatmap_loss(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1), -torch.ones(1, 1))
    with pytest.raises(ValueError):
        local_patch_loss([patch(0, 0, Box(0, 0, 2, 2), torch.zeros(1, 1, 1, 2))], torch.zeros(1, 1, 2), torch.ones(1, 1), (3, 3))
