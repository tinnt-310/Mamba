"""Depth Quality Optimization adapted from TAWNet commit 6e92c6c7.

Reference:
https://github.com/Karry-Wu/TAWNet/blob/6e92c6c7d4f2bd59f408e40e46b542cd2abd5092/models/TAWNet.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthQualityOptimization(nn.Module):
    """TAWNet-style Depth Quality Optimization (DQO).

    For a depth feature ``D`` this computes::

        F = PReLU(Conv3x3(D))
        W = sigmoid(max_channel(F * D))
        D_optimized = F + W * F

    The implementation deliberately keeps equal input/output channels, matching
    every DQO use in the referenced TAWNet source.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        depth_feature: torch.Tensor,
        *,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if depth_feature.ndim != 4 or depth_feature.shape[1] != self.channels:
            raise ValueError(
                f"Expected [B,{self.channels},H,W], got {tuple(depth_feature.shape)}"
            )

        refined = self.refine(depth_feature)
        correlation = (refined * depth_feature).amax(dim=1, keepdim=True)
        quality_gate = self.sigmoid(correlation)
        optimized = refined + quality_gate * refined
        if return_gate:
            return optimized, quality_gate
        return optimized
