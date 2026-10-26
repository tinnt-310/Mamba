"""Dense camera geometry. Depth and XYZ are meters, never image-normalized."""
import torch

from .normal_estimator import xyz_to_normals


def build_valid_mask(depth: torch.Tensor) -> torch.Tensor:
    """Return boolean [B,1,H,W] mask for finite, positive optical-axis Z."""
    if depth.ndim != 4 or depth.shape[1] != 1 or min(depth.shape) <= 0:
        raise ValueError("depth must be nonempty [B,1,H,W]")
    if not depth.is_floating_point():
        raise ValueError("depth must be floating point in meters")
    return torch.isfinite(depth) & (depth > 0)


def depth_to_xyz(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Back-project using each frame's [B,3,3] pinhole K; invalid XYZ is zero.

    Pixel centers use integer u/v. Axes: X right, Y down, Z forward.
    Half precision inputs are promoted to float32 for geometry calculations.
    K must describe the current depth grid (including any crop/resize).
    """
    valid = build_valid_mask(depth)
    if K.shape != (depth.shape[0], 3, 3) or K.device != depth.device:
        raise ValueError("K must be [B,3,3] on the depth device")
    if not K.is_floating_point() or not torch.isfinite(K).all():
        raise ValueError("K must contain finite floating-point values")
    if (K[:, (0, 1), (0, 1)] <= 0).any():
        raise ValueError("fx and fy must be positive")
    expected = K.new_tensor([0, 0, 1]).expand(K.shape[0], -1)
    if not torch.allclose(K[:, 2], expected) or (K[:, 0, 1] != 0).any() or (K[:, 1, 0] != 0).any():
        raise ValueError("K must be a zero-skew pinhole intrinsic matrix")
    dtype = torch.float64 if depth.dtype == torch.float64 else torch.float32
    z = torch.where(valid, depth, 0).to(dtype)
    k = K.to(dtype)
    u = torch.arange(depth.shape[3], device=depth.device, dtype=dtype)[None, None, None, :]
    v = torch.arange(depth.shape[2], device=depth.device, dtype=dtype)[None, None, :, None]
    fx, fy, cx, cy = (k[:, i, j, None, None, None] for i, j in ((0, 0), (1, 1), (0, 2), (1, 2)))
    xyz = torch.cat(((u - cx) / fx * z, (v - cy) / fy * z, z), dim=1)
    if not torch.isfinite(xyz).all():
        raise ValueError("Depth/K magnitudes overflow metric XYZ")
    return torch.where(valid, xyz, 0)


def build_geometry_tensor(depth: torch.Tensor, K: torch.Tensor, **normal_options) -> torch.Tensor:
    """Return [B,7,H,W] = XYZ, normal, depth-valid mask.

    Normal validity is stricter than depth validity; undefined normals are zero
    while their valid center XYZ and mask remain available to the encoder.
    """
    mask = build_valid_mask(depth)
    xyz = depth_to_xyz(depth, K)
    normal, _ = xyz_to_normals(xyz, mask, **normal_options)
    return torch.cat((xyz, normal, mask.to(xyz.dtype)), dim=1)
