"""Image-level orchestration of self SSM and the standalone CPSSM block."""
import torch
from torch import nn
from ..ssm import scan_image, merge_image
from .cp_ssm import CPSSM, exchange_readouts


class CrossParametricSSM(nn.Module):
    """Return self/cross maps for two aligned [B,C,H,W] inputs.

    Independent modality projections and SSM parameters, with a shared spatial
    depthwise convolution as in DBISSF_SS. Each recipient uses identical output
    norm/projection for its self and cross readouts. Four-direction mode shares
    weights across directions and averages maps; it is a local extension.
    """
    def __init__(self, channels: int, state_dim: int = 16, expand: int = 2,
                 directions: int = 1, backend: str = "torch"):
        super().__init__()
        if channels <= 0 or not isinstance(expand, int) or expand <= 0:
            raise ValueError("channels and integer expand must be positive")
        if directions not in (1, 4):
            raise ValueError("directions must be 1 or 4")
        self.channels, self.directions = channels, directions
        inner = channels * expand
        self.rgb_in = nn.Conv2d(channels, inner, 1, bias=False)
        self.depth_in = nn.Conv2d(channels, inner, 1, bias=False)
        self.local = nn.Sequential(nn.Conv2d(inner, inner, 3, padding=1, groups=inner), nn.SiLU())
        self.cp_ssm = CPSSM(inner, state_dim, backend=backend)
        self.rgb_norm, self.depth_norm = nn.LayerNorm(inner), nn.LayerNorm(inner)
        self.rgb_out = nn.Conv2d(inner, channels, 1, bias=False)
        self.depth_out = nn.Conv2d(inner, channels, 1, bias=False)

    @property
    def rgb_ssm(self):
        return self.cp_ssm.rgb_ssm

    @property
    def depth_ssm(self):
        return self.cp_ssm.depth_ssm

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> dict[str, torch.Tensor]:
        if rgb.ndim != 4 or rgb.shape != depth.shape or rgb.shape[1] != self.channels:
            raise ValueError(f"Expected matching [B,{self.channels},H,W] inputs")
        if rgb.device != depth.device or rgb.dtype != depth.dtype:
            raise ValueError("Inputs must have matching device and dtype")
        b, _, h, w = rgb.shape
        r = scan_image(self.local(self.rgb_in(rgb)), self.directions).flatten(0, 1)
        d = scan_image(self.local(self.depth_in(depth)), self.directions).flatten(0, 1)
        rp, dp = self.rgb_ssm.parameterize(r), self.depth_ssm.parameterize(d)
        rgb_cross, depth_cross = self.cp_ssm(r, d, parameters=(rp, dp))

        def finish(y, norm, projection):
            y = norm(y.transpose(1, 2)).transpose(1, 2)
            return projection(merge_image(y.reshape(b, self.directions, -1, h * w), h, w))

        return {
            "rgb_self": finish(self.rgb_ssm.run(r, rp), self.rgb_norm, self.rgb_out),
            "rgb_cross": finish(rgb_cross, self.rgb_norm, self.rgb_out),
            "depth_self": finish(self.depth_ssm.run(d, dp), self.depth_norm, self.depth_out),
            "depth_cross": finish(depth_cross, self.depth_norm, self.depth_out),
        }
