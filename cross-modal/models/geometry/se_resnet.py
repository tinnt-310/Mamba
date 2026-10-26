"""Feature-only SE-ResNet backbones provided by :mod:`timm`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:  # pragma: no cover - exercised only in broken envs
    raise ImportError(
        "SE-ResNet backbones require timm. Install project requirements first."
    ) from exc


SE_RESNET_MODELS = ("seresnet50", "seresnet101", "seresnet152")
DEFAULT_OUT_INDICES = (0, 1, 2, 3, 4)


class SEResNetBackbone(nn.Module):
    """Thin, feature-only wrapper around timm's SE-ResNet implementation.

    ``seresnet50`` is SE-ResNet-50 (not ResNet-50, SE-ResNeXt, or SENet154).
    The default output indices expose the stem and all four residual stages.
    """

    def __init__(
        self,
        name: str = "seresnet50",
        *,
        in_channels: int = 3,
        pretrained: bool = True,
        out_indices: Sequence[int] = DEFAULT_OUT_INDICES,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        normalized_name = name.lower().replace("-", "")
        if normalized_name not in SE_RESNET_MODELS:
            choices = ", ".join(SE_RESNET_MODELS)
            raise ValueError(f"Unknown SE-ResNet backbone {name!r}; choose: {choices}")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")

        indices = tuple(int(index) for index in out_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("out_indices must contain unique feature indices")
        if any(index < 0 or index > 4 for index in indices):
            raise ValueError("SE-ResNet out_indices must be between 0 and 4")

        self.name = normalized_name
        self.input_channels = in_channels
        self.out_indices = indices
        self.encoder = timm.create_model(
            normalized_name,
            pretrained=pretrained or checkpoint_path is not None,
            features_only=True,
            out_indices=indices,
            in_chans=in_channels,
            # Use timm's pretrained loader before wrapping feature extraction;
            # it safely removes the ImageNet classifier from full RGB weights.
            pretrained_cfg_overlay={"file": str(checkpoint_path)} if checkpoint_path else None,
        )
        self.stage_channels = tuple(self.encoder.feature_info.channels())
        self.stage_reductions = tuple(self.encoder.feature_info.reduction())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected [B,{self.input_channels},H,W], got {tuple(x.shape)}"
            )
        return tuple(self.encoder(x))


def build_se_resnet_backbone(
    name: str = "seresnet50",
    *,
    in_channels: int = 3,
    pretrained: bool = True,
    out_indices: Sequence[int] = DEFAULT_OUT_INDICES,
    checkpoint_path: str | Path | None = None,
) -> SEResNetBackbone:
    """Build an independent timm SE-ResNet feature encoder."""
    return SEResNetBackbone(
        name=name,
        in_channels=in_channels,
        pretrained=pretrained,
        out_indices=out_indices,
        checkpoint_path=checkpoint_path,
    )
