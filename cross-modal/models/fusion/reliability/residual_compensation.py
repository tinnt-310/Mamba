"""Task-specific directional interpolation of self and cross readouts."""
from torch import nn
from .reliability_gate import ReliabilityGate


def compensate(self_feature, cross_feature, gate):
    """G=0 gives self; G=1 gives cross. Caller supplies a sigmoid gate."""
    if self_feature.shape != cross_feature.shape or gate.shape != self_feature.shape:
        raise ValueError("Self, cross and gate must have matching shapes")
    return self_feature + gate * (cross_feature - self_feature)


class ResidualCompensation(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.d2r = ReliabilityGate(channels)
        self.r2d = ReliabilityGate(channels)

    def forward(self, features, quality=None):
        rs, rc = features["rgb_self"], features["rgb_cross"]
        ds, dc = features["depth_self"], features["depth_cross"]
        gd, gr = self.d2r(rs, rc,quality), self.r2d(ds, dc,quality)
        return compensate(rs, rc, gd), compensate(ds, dc, gr), gd, gr
