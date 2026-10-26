"""Shared-parameter scans extracted from MS2Fusion DBISSF_Attention.

Spatial shared embedding belongs to MS2FusionBranches, matching DBISSF_SS.
This core accepts already embedded sequences in [B,C,L] layout.
"""
import torch
from torch import nn
from ..ssm import SimpleSelectiveSSM


class SPSSM(nn.Module):
    """Jointly generate Delta/B/C; scan two distinct inputs with shared A/D.

    shared_rgb/shared_depth are produced by the same embedding weights.
    Their sum parameterizes the SSM. The scan inputs remain rgb/depth, exactly
    as in DBISSF_Attention; neither shared embedding replaces the scan input.
    Output normalization is applied by the surrounding feature block.
    """
    def __init__(self, channels: int, state_dim: int = 16, *, backend="torch", dt_rank=None):
        super().__init__()
        self.channels = channels
        self.shared_ssm = SimpleSelectiveSSM(channels, state_dim, dt_rank=dt_rank, backend=backend)

    def forward(self, rgb, depth, shared_rgb, shared_depth):
        tensors = (rgb, depth, shared_rgb, shared_depth)
        if rgb.ndim != 3 or min(rgb.shape) <= 0 or rgb.shape[1] != self.channels:
            raise ValueError(f"Expected nonempty [B,{self.channels},L] sequences")
        if any(x.shape != rgb.shape or x.device != rgb.device or x.dtype != rgb.dtype
               or not x.is_floating_point() for x in tensors):
            raise ValueError("All four sequences require matching shapes, devices and floating-point dtypes")
        shared = self.shared_ssm.parameterize(shared_rgb + shared_depth)
        return self.shared_ssm.run(rgb, shared), self.shared_ssm.run(depth, shared)
