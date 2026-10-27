"""Task-specific learned gates; not original MS2Fusion/Fusion-Mamba modules."""
import torch
from torch import nn


class ReliabilityGate(nn.Module):
    def __init__(self, channels: int, quality_channels: int = 0):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.quality_channels=quality_channels
        self.net = nn.Sequential(nn.Conv2d(3 * channels+quality_channels, channels, 1), nn.GELU(),
                                 nn.Conv2d(channels, channels, 1), nn.Sigmoid())

    def forward(self, self_feature, cross_feature, quality=None):
        if self_feature.shape != cross_feature.shape:
            raise ValueError("Self and cross features must have matching shapes")
        parts=[self_feature,cross_feature,(cross_feature-self_feature).abs()]
        if self.quality_channels:
            if quality is None or quality.shape!=(self_feature.shape[0],self.quality_channels,*self_feature.shape[-2:]):
                raise ValueError('Quality grid differs from gate features')
            parts.append(quality)
        return self.net(torch.cat(parts,dim=1))
