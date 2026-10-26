"""CP + SP feature paths matching DBISSF_SS(scan=False, dropout=0).

This is the dual-path feature block, not the complete upstream SSF fusion
network. Source names, commit and scope are documented in SOURCE_REVIEW.md.
"""
import math
import torch
from torch import nn
from .cp_ssm import CPSSM
from .sp_ssm import SPSSM


class MS2FusionBranches(nn.Module):
    """Aligned [B,C,H,W] pair -> four [B,C,H,W] CP/SP feature maps.

    Matches the default raster path of DBISSF_SS: independent modality input
    projections, one shared embedding, one depthwise convolution reused for
    all four streams, CP/SP scans, per-modality CP norms/projections, and a
    shared SP norm/projection. SP is always present and evaluated.
    """
    def __init__(self, channels: int, state_dim: int = 16, expand: int = 2, *,
                 dt_rank: int | None = None, backend: str = "torch"):
        super().__init__()
        if channels <= 0 or not isinstance(expand, int) or expand <= 0:
            raise ValueError("channels and integer expand must be positive")
        self.channels = channels
        inner = channels * expand
        # Upstream auto rank is derived from d_model, not expanded d_inner.
        rank = math.ceil(channels / 16) if dt_rank is None else dt_rank
        self.rgb_in = nn.Conv2d(channels, inner, 1, bias=False)
        self.depth_in = nn.Conv2d(channels, inner, 1, bias=False)
        self.shared_in = nn.Conv2d(channels, inner, 1, bias=False)
        self.local = nn.Sequential(nn.Conv2d(inner, inner, 3, padding=1, groups=inner), nn.SiLU())
        self.cp_ssm = CPSSM(inner, state_dim, dt_rank=rank, backend=backend)
        self.sp_ssm = SPSSM(inner, state_dim, dt_rank=rank, backend=backend)
        self.rgb_norm, self.depth_norm = nn.LayerNorm(inner), nn.LayerNorm(inner)
        self.shared_norm = nn.LayerNorm(inner)
        self.rgb_out = nn.Conv2d(inner, channels, 1, bias=False)
        self.depth_out = nn.Conv2d(inner, channels, 1, bias=False)
        self.shared_out = nn.Conv2d(inner, channels, 1, bias=False)

    def forward(self, rgb, depth):
        if rgb.ndim != 4 or min(rgb.shape) <= 0 or rgb.shape != depth.shape or rgb.shape[1] != self.channels:
            raise ValueError(f"Expected matching nonempty [B,{self.channels},H,W] features")
        if rgb.device != depth.device or rgb.dtype != depth.dtype or not rgb.is_floating_point():
            raise ValueError("Inputs require matching devices and floating-point dtypes")
        b, _, h, w = rgb.shape
        r = self.local(self.rgb_in(rgb)).flatten(2)
        d = self.local(self.depth_in(depth)).flatten(2)
        sr = self.local(self.shared_in(rgb)).flatten(2)
        sd = self.local(self.shared_in(depth)).flatten(2)
        rc, dc = self.cp_ssm(r, d)
        rs, ds = self.sp_ssm(r, d, sr, sd)

        def finish(x, norm, projection):
            return projection(norm(x.transpose(1, 2)).transpose(1, 2).reshape(b, -1, h, w))

        return {
            "rgb_cross": finish(rc, self.rgb_norm, self.rgb_out),
            "rgb_shared": finish(rs, self.shared_norm, self.shared_out),
            "depth_cross": finish(dc, self.depth_norm, self.depth_out),
            "depth_shared": finish(ds, self.shared_norm, self.shared_out),
        }
