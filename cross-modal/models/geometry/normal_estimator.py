"""PyTorch central-difference normals with explicit metric validity rules."""
import math

import torch
import torch.nn.functional as F


def xyz_to_normals(xyz: torch.Tensor, valid_mask: torch.Tensor, *,
                   radius: int = 1, depth_threshold: float = 0.02,
                   eps: float = 1e-12, point_into_surface: bool = False):
    """Return (normal, normal_valid), [B,3,H,W] and boolean [B,1,H,W].

    Four axial neighbors at +/- radius must be valid and within
    depth_threshold meters of the center. Borders are invalid (no wraparound).
    eps thresholds cross-product magnitude in m^2. Defaults face the camera;
    point_into_surface=True reverses orientation, yielding +Z on a front plane.
    This is inspired by normalSpeed's validity/orientation, not its estimator.
    """
    if xyz.ndim != 4 or xyz.shape[1] != 3 or min(xyz.shape) <= 0 or not xyz.is_floating_point():
        raise ValueError("xyz must be nonempty floating [B,3,H,W]")
    if valid_mask.shape != (xyz.shape[0], 1, *xyz.shape[2:]) or valid_mask.dtype != torch.bool or valid_mask.device != xyz.device:
        raise ValueError("valid_mask must be boolean [B,1,H,W] on the XYZ device")
    if not isinstance(radius, int) or radius < 1:
        raise ValueError("radius must be a positive integer")
    if not math.isfinite(depth_threshold) or depth_threshold < 0 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("depth_threshold must be finite/nonnegative and eps finite/positive")
    p = xyz if xyz.dtype == torch.float64 else xyz.float()
    valid = valid_mask & torch.isfinite(p).all(dim=1, keepdim=True) & (p[:, 2:3] > 0)
    p = torch.where(valid, p, 0)
    if min(p.shape[2:]) <= 2 * radius:
        return p * 0, torch.zeros_like(valid)
    r = radius
    center = p[:, :, r:-r, r:-r]
    slices = ((slice(r, -r), slice(2*r, None)), (slice(r, -r), slice(None, -2*r)),
              (slice(2*r, None), slice(r, -r)), (slice(None, -2*r), slice(r, -r)))
    neighbors = [p[:, :, y, x] for y, x in slices]
    good = valid[:, :, r:-r, r:-r].clone()
    for neighbor, (y, x) in zip(neighbors, slices):
        good = good & valid[:, :, y, x] & ((neighbor[:, 2:3] - center[:, 2:3]).abs() <= depth_threshold)
    dx, dy = neighbors[0] - neighbors[1], neighbors[2] - neighbors[3]
    # Mask before arithmetic/normalization to keep invalid gradients finite.
    dx, dy = torch.where(good, dx, 0), torch.where(good, dy, 0)
    cross = torch.linalg.cross(dx, dy, dim=1)
    finite = torch.isfinite(cross).all(dim=1, keepdim=True)
    cross = torch.where(finite, cross, 0)
    length = torch.linalg.vector_norm(cross, dim=1, keepdim=True)
    good = good & finite & torch.isfinite(length) & (length > eps)
    normal = cross / length.clamp_min(eps)
    # Fix orientation using the camera-to-point ray, including tilted surfaces.
    facing_away = (normal * center).sum(dim=1, keepdim=True) > 0
    normal = torch.where(facing_away, -normal, normal)
    if point_into_surface:
        normal = -normal
    normal = torch.where(good & torch.isfinite(normal).all(dim=1, keepdim=True), normal, 0)
    return F.pad(normal, (r, r, r, r)), F.pad(good, (r, r, r, r), value=False)
