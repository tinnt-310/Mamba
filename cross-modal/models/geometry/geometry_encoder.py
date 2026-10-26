"""Learned dense geometry stages aligned to the existing timm SE-ResNet."""
import torch
from torch import nn

from .depth_to_geometry import build_geometry_tensor


class GeometryResidualBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False),
            nn.GroupNorm(1, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels))
        self.skip = (nn.Identity() if in_channels == channels and stride == 1 else
                     nn.Conv2d(in_channels, channels, 1, stride, bias=False))
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class DenseGeometryEncoder(nn.Module):
    """[B,7,H,W] -> tuple of learned NCHW maps, same ordering as RGB.

    A 7x7/2 stem and 3x3/2 maxpool follow the RGB sampling grid. The four
    residual stages have strides 1,2,2,2 with symmetric padding. Consequently
    output sizes are ceil(H/reduction), ceil(W/reduction), including odd sizes.
    GroupNorm supports small batches; no normalization is applied to input XYZ.
    """
    def __init__(self, *, channels=(32, 48, 96, 128, 192), out_indices=(0, 1, 2, 3, 4)):
        super().__init__()
        channels, indices = tuple(channels), tuple(out_indices)
        if len(channels) != 5 or any(not isinstance(c, int) or c < 2 for c in channels):
            raise ValueError("channels must contain five integers >= 2")
        if not indices or len(set(indices)) != len(indices) or any(not isinstance(i, int) or i not in range(5) for i in indices):
            raise ValueError("out_indices must be unique integers in [0,4]")
        # timm's feature extraction returns outputs in traversal order.
        if indices != tuple(sorted(indices)):
            raise ValueError("out_indices must be in increasing stage order")
        self.out_indices = indices
        self.stage_channels = tuple(channels[i] for i in indices)
        self.stage_reductions = tuple(2 ** (i + 1) for i in indices)
        self.stem = nn.Sequential(nn.Conv2d(7, channels[0], 7, 2, 3, bias=False),
                                  nn.GroupNorm(1, channels[0]), nn.SiLU())
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.stages = nn.ModuleList(
            GeometryResidualBlock(channels[i-1], channels[i], 1 if i == 1 else 2)
            for i in range(1, max(indices) + 1))

    @classmethod
    def from_rgb_backbone(cls, rgb_backbone, **kwargs):
        """Read selected scales from the existing RGB instance; do not own it."""
        encoder = cls(out_indices=rgb_backbone.out_indices, **kwargs)
        if tuple(rgb_backbone.stage_reductions) != encoder.stage_reductions:
            raise ValueError("RGB reductions do not match the supported SE-ResNet sampling grid")
        return encoder

    def forward(self, geometry):
        if geometry.ndim != 4 or geometry.shape[1] != 7 or min(geometry.shape) <= 0:
            raise ValueError("geometry must be nonempty [B,7,H,W]")
        if not geometry.is_floating_point() or not torch.isfinite(geometry).all():
            raise ValueError("geometry must be finite floating point")
        x = self.stem(geometry)
        features = [x]
        if self.stages:
            x = self.pool(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return tuple(features[i] for i in self.out_indices)


class DepthGeometryBranch(nn.Module):
    """Per-frame metric depth + K -> learned feature tuple for Mamba fusion."""
    def __init__(self, encoder: DenseGeometryEncoder | None = None, **normal_options):
        super().__init__()
        self.encoder = DenseGeometryEncoder() if encoder is None else encoder
        self.normal_options = normal_options
        self.stage_channels = self.encoder.stage_channels
        self.stage_reductions = self.encoder.stage_reductions
        self.out_indices = self.encoder.out_indices

    def forward(self, depth, K):
        geometry = build_geometry_tensor(depth, K, **self.normal_options)
        return self.encoder(geometry)


class GeometryProjector(nn.Module):
    """Pair of per-scale 1x1 projections for direct MS2FusionBranches input.

    MultiScaleRGBDCompensation already owns equivalent projections; use its
    built-in projections instead when connecting to that module.
    """
    def __init__(self, rgb_channels, geo_channels, channels):
        super().__init__()
        r, g, c = map(tuple, (rgb_channels, geo_channels, channels))
        if not c or not (len(r) == len(g) == len(c)) or any(not isinstance(v, int) or v <= 0 for v in r + g + c):
            raise ValueError("channel lists must have equal nonzero length and positive integer entries")
        self.rgb_projectors = nn.ModuleList(nn.Conv2d(a, b, 1) for a, b in zip(r, c))
        self.geo_projectors = nn.ModuleList(nn.Conv2d(a, b, 1) for a, b in zip(g, c))

    def forward(self, rgb_features, geo_features):
        if len(rgb_features) != len(self.rgb_projectors) or len(geo_features) != len(self.geo_projectors):
            raise ValueError("feature count must match projector scale count")
        rgb_out, geo_out = [], []
        for r, g, rp, gp in zip(rgb_features, geo_features, self.rgb_projectors, self.geo_projectors):
            if r.ndim != 4 or g.ndim != 4 or r.shape[0] != g.shape[0] or r.shape[2:] != g.shape[2:]:
                raise ValueError("RGB and geometry batch/spatial shapes must match")
            if r.shape[1] != rp.in_channels or g.shape[1] != gp.in_channels:
                raise ValueError("feature channels do not match projectors")
            if r.device != g.device or r.dtype != g.dtype or not r.is_floating_point():
                raise ValueError("RGB and geometry need matching floating dtype/device")
            rgb_out.append(rp(r))
            geo_out.append(gp(g))
        return tuple(rgb_out), tuple(geo_out)
