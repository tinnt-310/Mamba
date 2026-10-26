"""Task-specific learned gates; not original MS2Fusion/Fusion-Mamba modules."""
import torch
from torch import nn


class ReliabilityGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.net = nn.Sequential(nn.Conv2d(3 * channels, channels, 1), nn.GELU(),
                                 nn.Conv2d(channels, channels, 1), nn.Sigmoid())

    def forward(self, self_feature, cross_feature):
        if self_feature.shape != cross_feature.shape:
            raise ValueError("Self and cross features must have matching shapes")
        return self.net(torch.cat((self_feature, cross_feature,
                                   (cross_feature - self_feature).abs()), dim=1))
