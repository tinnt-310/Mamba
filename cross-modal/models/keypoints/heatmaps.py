"""Pixel-center transforms, annotation targets, and masked subpixel decoding."""

from dataclasses import dataclass
import math

import numpy as np
import torch

from .regions import _grid_hw


@dataclass(frozen=True)
class DecodedKeypoints:
    coords: torch.Tensor
    argmax: torch.Tensor
    peak_response: torch.Tensor
    subpixel_used: torch.Tensor


def transform_points(points, src_hw, dst_hw):
    """Map (x,y) point centers with the align_corners=False convention."""
    src_h, src_w = _grid_hw(src_hw)
    dst_h, dst_w = _grid_hw(dst_hw)
    if isinstance(points, torch.Tensor):
        if points.ndim < 1 or points.shape[-1] != 2:
            raise ValueError("Points must have final dimension 2 (x, y)")
        values = points if points.is_floating_point() else points.to(torch.get_default_dtype())
        scale = values.new_tensor((dst_w / src_w, dst_h / src_h))
        return (values + 0.5) * scale - 0.5
    if isinstance(points, np.ndarray):
        if points.ndim < 1 or points.shape[-1] != 2:
            raise ValueError("Points must have final dimension 2 (x, y)")
        dtype = points.dtype if np.issubdtype(points.dtype, np.floating) else np.float64
        values = points.astype(dtype, copy=False)
        scale = np.asarray((dst_w / src_w, dst_h / src_h), dtype=dtype)
        return (values + 0.5) * scale - 0.5
    raise TypeError("Points must be a torch Tensor or NumPy array")


def _point_weights(points, weights):
    if weights is None:
        values = torch.ones(points.shape[:-1], dtype=points.dtype, device=points.device)
    else:
        values = torch.as_tensor(weights, dtype=points.dtype, device=points.device)
    if values.shape != points.shape[:-1]:
        raise ValueError("Weights must match all point dimensions except (x, y)")
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Keypoint weights must be finite and nonnegative")
    valid = values > 0
    if not torch.isfinite(points[valid]).all():
        raise ValueError("Visible keypoint coordinates must be finite")
    return values, valid


def gaussian_heatmaps(points, hw, sigma, weights=None):
    """Render [...,K,H,W] Gaussians; missing labels produce exactly zero."""
    height, width = _grid_hw(hw)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian sigma must be finite and positive")
    values = torch.as_tensor(points)
    if values.ndim < 2 or values.shape[-1] != 2:
        raise ValueError("Points must have shape [..., K, 2]")
    if not values.is_floating_point():
        values = values.to(torch.get_default_dtype())
    _, valid = _point_weights(values, weights)
    safe_points = torch.where(valid[..., None], values, torch.zeros_like(values))
    yy, xx = torch.meshgrid(torch.arange(height, device=values.device, dtype=values.dtype),
                            torch.arange(width, device=values.device, dtype=values.dtype), indexing="ij")
    distance = (xx - safe_points[..., 0, None, None]).square() + (yy - safe_points[..., 1, None, None]).square()
    return torch.exp(-distance / (2 * sigma * sigma)) * valid[..., None, None]


def decode_heatmaps(heatmaps, support=None, subpixel=True):
    """Decode first row-major masked peak, refining each eligible axis in log space."""
    values = torch.as_tensor(heatmaps)
    if values.ndim < 2 or not all(values.shape):
        raise ValueError("Heatmaps must have nonempty shape [..., H, W]")
    if not values.is_floating_point():
        values = values.to(torch.get_default_dtype())
    if not torch.isfinite(values).all():
        raise ValueError("Heatmap responses must be finite")
    if support is None:
        mask = torch.ones_like(values, dtype=torch.bool)
    else:
        mask = torch.as_tensor(support, device=values.device)
        if mask.shape != values.shape or mask.dtype != torch.bool:
            raise ValueError("Support must be a bool tensor matching heatmap shape")
    height, width = values.shape[-2:]
    leading_shape = values.shape[:-2]
    flat_values = values.reshape(-1, height * width)
    flat_mask = mask.reshape(-1, height * width)
    if not flat_mask.any(dim=-1).all():
        raise ValueError("Every heatmap must have at least one pixel in support")
    flat_peak = flat_values.masked_fill(~flat_mask, -torch.inf).argmax(dim=-1)
    peak = flat_values.gather(1, flat_peak[:, None]).squeeze(1)
    x, y = flat_peak.remainder(width), flat_peak.div(width, rounding_mode="floor")
    argmax = torch.stack((x, y), dim=-1)
    coords = argmax.to(values.dtype)
    used = torch.zeros_like(argmax, dtype=torch.bool)
    if subpixel:
        for axis, step, position, bound in ((0, 1, x, width), (1, width, y, height)):
            inside = (position > 0) & (position < bound - 1)
            minus = (flat_peak - step).clamp(0, height * width - 1)
            plus = (flat_peak + step).clamp(0, height * width - 1)
            left = flat_values.gather(1, minus[:, None]).squeeze(1)
            right = flat_values.gather(1, plus[:, None]).squeeze(1)
            neighbors_supported = flat_mask.gather(1, minus[:, None]).squeeze(1) & flat_mask.gather(1, plus[:, None]).squeeze(1)
            eligible = inside & neighbors_supported & (left > 1e-6) & (peak > 1e-6) & (right > 1e-6)
            # Guard logs and the quotient even for axes which will fall back.
            log_left = torch.where(eligible, left, torch.ones_like(left)).log()
            log_peak = torch.where(eligible, peak, torch.ones_like(peak)).log()
            log_right = torch.where(eligible, right, torch.ones_like(right)).log()
            curvature = log_left - 2 * log_peak + log_right
            concave = eligible & (curvature < -1e-6)
            denominator = torch.where(concave, curvature, torch.ones_like(curvature))
            offset = 0.5 * (log_left - log_right) / denominator
            accepted = concave & torch.isfinite(offset) & (offset.abs() <= 0.5)
            coords[:, axis] = coords[:, axis] + torch.where(accepted, offset, torch.zeros_like(offset))
            used[:, axis] = accepted
    return DecodedKeypoints(coords.reshape(*leading_shape, 2), argmax.reshape(*leading_shape, 2),
                            peak.reshape(leading_shape), used.reshape(*leading_shape, 2))
