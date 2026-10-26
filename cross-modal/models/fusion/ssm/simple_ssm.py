"""Modality-agnostic selective SSM, independently written from the Mamba recurrence.

x_t is an input token; h_t is state [batch, channel, state_dim].
A is the continuous transition, B_t input-to-state, C_t state-to-output,
Delta_t a positive selective time step, and D a learned direct skip.
As in official Mamba: A_bar=exp(Delta*A), B_bar=Delta*B (not exact ZOH B).
h_t = A_bar_t*h_(t-1) + Delta_t*B_t*x_t; y_t = C_t*h_t + D*x_t.
See docs/SOURCE_REVIEW.md for pinned source references and design differences.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SelectiveParameters:
    delta: torch.Tensor  # [B,C,L], already positive
    B: torch.Tensor  # [B,N,L]
    C: torch.Tensor  # [B,N,L]


def selective_scan(x, delta, A, B, C, D, *, backend="torch"):
    """Real, variable B/C scan. CPU reference or explicitly selected Mamba CUDA.

    Accumulates half/bfloat16 in float32 and preserves float64 for grad checks.
    The torch loop is a correctness baseline, not a high-resolution training kernel.
    """
    if backend not in ("torch", "mamba"):
        raise ValueError("backend must be 'torch' or 'mamba'")
    if x.ndim != 3 or min(x.shape) <= 0 or delta.shape != x.shape:
        raise ValueError("x and delta must have matching nonempty [B,C,L] shapes")
    batch, channels, length = x.shape
    if A.ndim != 2 or A.shape[0] != channels or A.shape[1] <= 0:
        raise ValueError("A must have shape [C,N] with N positive")
    if B.shape != (batch, A.shape[1], length) or C.shape != B.shape:
        raise ValueError("B and C must have matching [B,N,L] shapes")
    if D.shape != (channels,):
        raise ValueError("D must have shape [C]")
    if any(t.device != x.device for t in (delta, A, B, C, D)):
        raise ValueError("All scan tensors must be on the same device")
    if not all(t.is_floating_point() for t in (x, delta, A, B, C, D)):
        raise ValueError("Scan requires real floating-point tensors")
    dtype = x.dtype
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    x, delta, A, B, C, D = (t.to(work_dtype) for t in (x, delta, A, B, C, D))
    if backend == "mamba":
        if not x.is_cuda or work_dtype == torch.float64:
            raise ValueError("Mamba backend requires CUDA and float32/float16/bfloat16 input")
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        return selective_scan_fn(
            x.contiguous(), delta.contiguous(), A.contiguous(), B.contiguous(),
            C.contiguous(), D.contiguous(), delta_softplus=False,
        ).to(dtype)
    state = x.new_zeros(batch, channels, A.shape[1])
    outputs = []
    for t in range(length):
        step = delta[:, :, t, None]
        state = (step * A).exp() * state + step * B[:, None, :, t] * x[:, :, t, None]
        outputs.append((state * C[:, None, :, t]).sum(-1) + D * x[:, :, t])
    return torch.stack(outputs, dim=-1).to(dtype)


class SimpleSelectiveSSM(nn.Module):
    """Standalone [B,C,L] -> [B,C,L] selective state-space operation."""

    def __init__(self, channels: int, state_dim: int = 16, dt_rank: int | None = None,
                 backend: str = "torch"):
        super().__init__()
        if channels <= 0 or state_dim <= 0 or (dt_rank is not None and dt_rank <= 0):
            raise ValueError("channels, state_dim and dt_rank must be positive")
        if backend not in ("torch", "mamba"):
            raise ValueError("backend must be 'torch' or 'mamba'")
        self.channels, self.state_dim = channels, state_dim
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(channels / 16)
        self.backend = backend
        self.x_proj = nn.Linear(channels, self.dt_rank + 2 * state_dim, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, channels)
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
        dt = torch.exp(torch.empty(channels).uniform_(math.log(0.001), math.log(0.1)))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.arange(1, state_dim + 1).float().log().repeat(channels, 1))
        self.D = nn.Parameter(torch.ones(channels))
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True

    def parameterize(self, x: torch.Tensor) -> SelectiveParameters:
        if x.ndim != 3 or x.shape[1] != self.channels or min(x.shape) <= 0:
            raise ValueError(f"Expected nonempty [B,{self.channels},L]")
        dt, B, C = self.x_proj(x.transpose(1, 2)).split(
            (self.dt_rank, self.state_dim, self.state_dim), dim=-1
        )
        dt = self.dt_proj(dt)
        dt = F.softplus(dt.double() if dt.dtype == torch.float64 else dt.float())
        return SelectiveParameters(dt.transpose(1, 2), B.transpose(1, 2), C.transpose(1, 2))

    def run(self, x: torch.Tensor, parameters: SelectiveParameters) -> torch.Tensor:
        A_log = self.A_log.double() if x.dtype == torch.float64 else self.A_log.float()
        return selective_scan(x, parameters.delta, -A_log.exp(), parameters.B,
                              parameters.C, self.D, backend=self.backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.run(x, self.parameterize(x))
