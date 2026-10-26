"""Standalone CP-SSM adapted from MS2Fusion DBISSF_Attention(Cross=True).

CPSSM is a local class name. Upstream combines CP and SP in DBISSF_Attention;
this block isolates the two cross-parametric scans, without SP or fusion.
"""
import torch
from torch import nn
from ..ssm import SimpleSelectiveSSM, SelectiveParameters


def exchange_readouts(rgb: SelectiveParameters, depth: SelectiveParameters):
    """Keep recipient Delta/B and substitute the donor readout C."""
    if rgb.C.shape != depth.C.shape:
        raise ValueError("Readouts must have the same [B,N,L] shape")
    return (SelectiveParameters(rgb.delta, rgb.B, depth.C),
            SelectiveParameters(depth.delta, depth.B, rgb.C))


class CPSSM(nn.Module):
    """Two [B,C,L] sequences -> RGB-cross and Depth-cross sequences.

    Owns independent modality parameter generators and A/D parameters.
    RGB: scan(R; A_R, Delta_R, B_R, C_D, D_R).
    Depth: scan(D; A_D, Delta_D, B_D, C_R, D_D).
    Optional precomputed parameters let self/cross branches use the exact same
    token-dependent parameters without projecting twice. Omit them for standalone use.
    """
    def __init__(self, channels: int, state_dim: int = 16, *, backend: str = "torch",
                 dt_rank: int | None = None):
        super().__init__()
        self.channels = channels
        self.rgb_ssm = SimpleSelectiveSSM(channels, state_dim, dt_rank=dt_rank, backend=backend)
        self.depth_ssm = SimpleSelectiveSSM(channels, state_dim, dt_rank=dt_rank, backend=backend)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, *, parameters=None):
        if rgb.ndim != 3 or rgb.shape != depth.shape or rgb.shape[1] != self.channels or min(rgb.shape) <= 0:
            raise ValueError(f"Expected matching nonempty [B,{self.channels},L] sequences")
        if rgb.device != depth.device or rgb.dtype != depth.dtype or not rgb.is_floating_point():
            raise ValueError("Inputs require matching devices and floating-point dtypes")
        if parameters is None:
            rp = self.rgb_ssm.parameterize(rgb)
            dp = self.depth_ssm.parameterize(depth)
        else:
            rp, dp = parameters
        rc, dc = exchange_readouts(rp, dp)
        return self.rgb_ssm.run(rgb, rc), self.depth_ssm.run(depth, dc)
