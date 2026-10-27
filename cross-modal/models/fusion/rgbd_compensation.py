"""Feature-only fusion; no backbone, decoder or additional Mamba fusion stage."""
import torch
from torch import nn
from torch.nn import functional as F
from .ms2fusion import CrossParametricSSM
from .reliability import ResidualCompensation


class ChannelNorm(nn.Module):
    """Per-pixel LayerNorm, also defined for a single spatial position."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class RGBDCompensation(nn.Module):
    def __init__(self, rgb_channels: int, depth_channels: int, channels: int,
                 out_channels: int | None = None, *, state_dim: int = 16,
                 expand: int = 2, directions: int = 1, backend: str = "torch",
                 validity_aware: bool = False):
        super().__init__()
        out_channels = channels if out_channels is None else out_channels
        if min(rgb_channels, depth_channels, channels, out_channels) <= 0:
            raise ValueError("Channel counts must be positive")
        self.rgb_channels, self.depth_channels = rgb_channels, depth_channels
        self.rgb_project = nn.Conv2d(rgb_channels, channels, 1) if rgb_channels != channels else nn.Identity()
        self.depth_project = nn.Conv2d(depth_channels, channels, 1) if depth_channels != channels else nn.Identity()
        self.cross_parametric = CrossParametricSSM(channels, state_dim, expand, directions, backend)
        self.compensation = ResidualCompensation(channels)
        self.validity_aware=validity_aware
        if validity_aware:
            from .reliability.reliability_gate import ReliabilityGate
            self.compensation.d2r=ReliabilityGate(channels,quality_channels=1)
            self.compensation.r2d=ReliabilityGate(channels,quality_channels=1)
        self.fuse = nn.Sequential(nn.Conv2d(2 * channels, out_channels, 1, bias=False),
                                  ChannelNorm(out_channels), nn.GELU())

    @property
    def cp(self):
        """Compatibility accessor for the image-level self/cross coordinator."""
        return self.cross_parametric

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, validity=None):
        if rgb.ndim != 4 or depth.ndim != 4 or min(rgb.shape) <= 0 or min(depth.shape) <= 0:
            raise ValueError("Expected nonempty [B,C,H,W] inputs")
        if rgb.shape[0] != depth.shape[0] or rgb.shape[2:] != depth.shape[2:]:
            raise ValueError("RGB/Depth batch and spatial dimensions must match")
        if rgb.shape[1] != self.rgb_channels or depth.shape[1] != self.depth_channels:
            raise ValueError("Input channels differ from configured channel counts")
        if rgb.device != depth.device or rgb.dtype != depth.dtype or not rgb.is_floating_point():
            raise ValueError("Inputs require matching devices and floating-point dtypes")
        aux = self.cross_parametric(self.rgb_project(rgb), self.depth_project(depth))
        quality=None
        if self.validity_aware:
            if validity is None:
                raise ValueError('Validity-aware fusion requires measured depth validity')
            quality=F.interpolate(validity.to(rgb.dtype),size=rgb.shape[-2:],mode='area')
        r, d, gd, gr = self.compensation(aux,quality=quality)
        aux.update(rgb_compensated=r, depth_compensated=d, gate_d2r=gd, gate_r2d=gr)
        fused=self.fuse(torch.cat((r,d),dim=1))
        if self.validity_aware:
            present=validity.flatten(1).any(1)[:,None,None,None]
            # Measured-depth absence falls back to the learned RGB self pathway.
            rgb_only=self.fuse(torch.cat((aux['rgb_self'],torch.zeros_like(d)),dim=1))
            fused=torch.where(present,fused,rgb_only)
        return fused, aux


class MultiScaleRGBDCompensation(nn.Module):
    """One independent compensation block per encoder output scale."""
    def __init__(self, rgb_channels, depth_channels, channels, out_channels=None, **kwargs):
        super().__init__()
        rgb_channels, depth_channels, channels = map(tuple, (rgb_channels, depth_channels, channels))
        out_channels = channels if out_channels is None else tuple(out_channels)
        if not channels or not (len(rgb_channels) == len(depth_channels) == len(channels) == len(out_channels)):
            raise ValueError("Channel lists must have the same nonzero number of scales")
        self.blocks = nn.ModuleList(RGBDCompensation(r, d, c, o, **kwargs)
                                   for r, d, c, o in zip(rgb_channels, depth_channels, channels, out_channels))

    def forward(self, rgb_features, depth_features):
        if len(rgb_features) != len(self.blocks) or len(depth_features) != len(self.blocks):
            raise ValueError("Feature scale count differs from the configured scale count")
        results = [block(r, d) for block, r, d in zip(self.blocks, rgb_features, depth_features)]
        return tuple(x[0] for x in results), tuple(x[1] for x in results)
