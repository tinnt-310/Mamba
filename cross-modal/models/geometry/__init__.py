"""Dense metric geometry features aligned with SE-ResNet fusion scales."""
from .depth_to_geometry import build_geometry_tensor, build_valid_mask, depth_to_xyz
from .normal_estimator import xyz_to_normals
from .geometry_encoder import (
    DenseGeometryEncoder, DepthGeometryBranch, GeometryProjector, GeometryResidualBlock,
)

__all__ = ["build_geometry_tensor", "build_valid_mask", "depth_to_xyz", "xyz_to_normals",
           "DenseGeometryEncoder", "DepthGeometryBranch", "GeometryProjector", "GeometryResidualBlock"]
