"""Annotation-derived weighted global and equally balanced local patch losses."""

from collections import defaultdict
import math

import torch

from .heatmaps import _point_weights, gaussian_heatmaps
from .regions import _grid_hw


def _positive_weight(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("positive_weight must be finite and positive")


def weighted_heatmap_loss(pred, target, weights, positive_weight=100):
    """Mean weighted pixel MSE, normalized over annotation keypoint weights."""
    _positive_weight(positive_weight)
    if pred.ndim < 3 or pred.shape != target.shape or not all(pred.shape):
        raise ValueError("Prediction and target must have matching nonempty [..., K, H, W] shapes")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise ValueError("Prediction and target must be floating-point tensors")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise ValueError("Prediction and target responses must be finite")
    keypoint_weights = torch.as_tensor(weights, dtype=pred.dtype, device=pred.device)
    if keypoint_weights.shape != pred.shape[:-2]:
        raise ValueError("Keypoint weights must match heatmap leading dimensions")
    if not torch.isfinite(keypoint_weights).all() or (keypoint_weights < 0).any():
        raise ValueError("Keypoint weights must be finite and nonnegative")
    total_weight = keypoint_weights.sum()
    if not (total_weight > 0):
        # Summing finite large values can overflow before multiplication by zero.
        return pred.reshape(-1)[0] * 0
    valid = (keypoint_weights > 0)[..., None, None]
    safe_prediction = torch.where(valid, pred, torch.zeros_like(pred))
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    pixel_weight = 1 + (positive_weight - 1) * safe_target
    per_keypoint = (pixel_weight * (safe_prediction - safe_target).square()).mean(dim=(-2, -1))
    return (per_keypoint * keypoint_weights).sum() / total_weight


def local_patch_loss(patches, points, weights, feature_hw, sigma=1, positive_weight=100):
    """Mean pixels, then patches per (batch,keypoint), then valid keypoints.

    ``points`` already uses F1 point-center coordinates. Annotation weights
    determine validity; a valid keypoint has one vote regardless of patch count.
    """
    _positive_weight(positive_weight)
    feature_h, feature_w = _grid_hw(feature_hw)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be finite and positive")
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("Local points must have shape [B, K, 2]")
    _, valid = _point_weights(points, weights)
    grouped = defaultdict(list)
    anchors = []
    for patch in patches:
        batch_index, keypoint_index = patch.batch_index, patch.keypoint_index
        if not (0 <= batch_index < points.shape[0] and 0 <= keypoint_index < points.shape[1]):
            raise ValueError("Patch batch/keypoint index is out of range")
        box, prediction = patch.box, patch.prediction
        if not (0 <= box.x1 < box.x2 <= feature_w and 0 <= box.y1 < box.y2 <= feature_h):
            raise ValueError("Patch box must be inside the feature grid")
        if prediction.shape != (1, 1, box.height, box.width):
            raise ValueError("Patch prediction must have shape [1, 1, box.height, box.width]")
        if not prediction.is_floating_point() or not torch.isfinite(prediction).all():
            raise ValueError("Patch prediction must contain finite floating-point responses")
        anchors.append(prediction.reshape(-1)[0] * 0)
        if not valid[batch_index, keypoint_index]:
            continue
        point = points[batch_index, keypoint_index].to(device=prediction.device, dtype=prediction.dtype)
        inside = (box.x1 - 0.5 <= point[0]) & (point[0] < box.x2 - 0.5) & (box.y1 - 0.5 <= point[1]) & (point[1] < box.y2 - 0.5)
        if inside:
            local_point = point - point.new_tensor((box.x1, box.y1))
            target = gaussian_heatmaps(local_point[None], (box.height, box.width), sigma)[None]
        else:
            target = torch.zeros_like(prediction)
        pixel_weight = 1 + (positive_weight - 1) * target
        grouped[(batch_index, keypoint_index)].append((pixel_weight * (prediction - target).square()).mean())
    if grouped:
        return torch.stack([torch.stack(losses).mean() for losses in grouped.values()]).mean()
    if anchors:
        return torch.stack(anchors).sum()
    # Empty prediction sets have no model graph to attach; allow safe backward.
    return torch.zeros((), dtype=points.dtype, device=points.device, requires_grad=True)
