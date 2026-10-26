"""SE-ResNet RGB-D fusion and global/native-local 2D keypoint heads.

The imported source did not include a keypoint model. The global decoder here
implements the documented projection/mean/two-convolution baseline; historical
checkpoint parameter names cannot be reconstructed from that description.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.geometry.se_resnet import SEResNetBackbone
from models.geometry.geometry_encoder import DenseGeometryEncoder, DepthGeometryBranch
from models.fusion.rgbd_compensation import MultiScaleRGBDCompensation
from .config import make_config
from .local_refinement import AdaptiveLocalRefinement, LocalRefinementOutput


@dataclass(frozen=True)
class StageOutput:
    global_heatmaps: torch.Tensor
    local: LocalRefinementOutput | None
    final_heatmaps: torch.Tensor | None = None
    residual: object | None = None


class MeanMultiScaleDecoder(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.projections = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels)
        self.refine = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(1, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(1, width), nn.GELU())

    def forward(self, features, output_hw):
        if len(features) != len(self.projections):
            raise ValueError("feature scale count differs from decoder projection count")
        resized = []
        for feature, projection in zip(features, self.projections):
            if feature.ndim != 4 or min(feature.shape) <= 0 or feature.shape[1] != projection.in_channels:
                raise ValueError("feature shape/channels differ from decoder projection")
            if resized and feature.shape[0] != resized[0].shape[0]:
                raise ValueError("decoder feature batch sizes must match")
            resized.append(F.interpolate(projection(feature), size=output_hw,
                                         mode="bilinear", align_corners=False))
        return self.refine(torch.stack(resized, dim=0).mean(dim=0))


class RGBDKeypointModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        # Explicit older configs without a local section remain global-only.
        # make_config(sparse_overrides) produces the full current config.
        if config is not None and "local" not in config:
            config = deepcopy(config)
            config["local"] = {"enabled": False}
        self.config = make_config(config)
        selected = self.config["out_indices"]
        detail_needed = self.config['detail_decoder'] or (
            self.config['local']['mode']=='residual' and self.config['local']['feature_source']=='detail')
        self._rgb_indices = tuple(sorted(set((0,)+selected))) if detail_needed else selected
        self.rgb_encoder = SEResNetBackbone(pretrained=self.config["pretrained"],
                                            out_indices=self._rgb_indices,
                                            checkpoint_path=self.config['pretrained_path'])
        rgb_channels = tuple(self.rgb_encoder.stage_channels[self._rgb_indices.index(i)] for i in selected)
        self.geometry_encoder = DepthGeometryBranch(DenseGeometryEncoder(
            channels=self.config["geometry_channels"], out_indices=selected))
        if tuple(self.rgb_encoder.stage_reductions[self._rgb_indices.index(i)] for i in selected) != self.geometry_encoder.stage_reductions:
            raise ValueError("RGB and geometry encoder feature grids must align")
        self.fusion = MultiScaleRGBDCompensation(
            rgb_channels, self.geometry_encoder.stage_channels,
            self.config["fusion_channels"], **self.config["global_ssm"])
        self.rgb_projections = None
        self.quality_blocks = None
        if self.config['modality']=='rgb' or self.config['skip_fusion_scales']:
            self.rgb_projections = nn.ModuleList(nn.Conv2d(c,o,1) for c,o in zip(rgb_channels,self.config['fusion_channels']))
            if self.config['modality']!='rgb':
                for i in range(len(selected)):
                    if i not in self.config['skip_fusion_scales']:
                        self.rgb_projections[i]=nn.Identity()
                    else:
                        self.fusion.blocks[i]=nn.Identity()
        if self.config['fusion_mode']=='quality':
            from .quality_fusion import QualityFusion
            self.quality_blocks = nn.ModuleList(QualityFusion(r,d,c,**self.config['global_ssm'])
                for r,d,c in zip(rgb_channels,self.geometry_encoder.stage_channels,self.config['fusion_channels']))
        # Unused legacy fusion/geometry parameters are excluded in new modalities.
        if self.config['modality']=='rgb' or self.quality_blocks is not None:
            self.fusion = nn.Identity()
        if self.config['modality']=='rgb':
            self.geometry_encoder = nn.Identity()
        self.decoder = MeanMultiScaleDecoder(self.config["fusion_channels"],
                                             self.config["decoder_width"])
        self.head = nn.Conv2d(self.config["decoder_width"], self.config["num_keypoints"], 1)
        self.detail_projection = (nn.Conv2d(self.rgb_encoder.stage_channels[self._rgb_indices.index(0)],
                                 self.config['decoder_width'],1) if self.config['detail_decoder'] else None)
        self.local_enabled = self.config["local"]["enabled"]
        self.local_refinement = (AdaptiveLocalRefinement(
            self.config["fusion_channels"][0], self.config["num_keypoints"],
            **self.config["local"]) if self.local_enabled and self.config['local']['mode']=='legacy' else None)
        self.residual_local = None
        if self.local_enabled and self.config['local']['mode']=='residual':
            from .residual_local import ResidualLocal
            self.local_refinement = None
            local = self.config['local']
            channels = (self.rgb_encoder.stage_channels[self._rgb_indices.index(0)]
                        if local['feature_source']=='detail' else self.config['fusion_channels'][0])
            self.residual_local = ResidualLocal(channels,self.config['fusion_channels'][0],
                self.config['num_keypoints'],roi_size=local['roi_size'],conditioning=local['conditioning'],
                state_dim=local['state_dim'],backend=local['backend'])
        self.phase = "joint" if self.local_enabled else "global"

    def _global_modules(self):
        return tuple(m for m in (self.rgb_encoder,self.geometry_encoder,self.fusion,self.decoder,self.head,
                     self.rgb_projections,self.quality_blocks,self.detail_projection) if m is not None)

    def set_phase(self, phase):
        if phase not in ("global", "local", "joint"):
            raise ValueError("phase must be global, local or joint")
        if phase in ("local", "joint") and not self.local_enabled:
            raise ValueError("local/joint phase requires local refinement enabled")
        self.phase = phase
        for module in self._global_modules():
            module.requires_grad_(phase != "local")
        if self.local_refinement is not None:
            self.local_refinement.requires_grad_(phase != "global")
        if self.residual_local is not None:
            self.residual_local.requires_grad_(phase != 'global')
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.phase == "local":
            for module in self._global_modules():
                module.eval()
        if self.phase == "global" and self.local_refinement is not None:
            self.local_refinement.eval()
        if self.phase == 'global' and self.residual_local is not None:
            self.residual_local.eval()
        return self

    def _encode(self, rgb, depth, K):
        if (rgb.ndim != 4 or depth.ndim != 4 or rgb.shape[0] != depth.shape[0] or
                rgb.shape[-2:] != depth.shape[-2:]):
            raise ValueError("RGB and depth must have matching nonempty batch/spatial shapes")
        all_rgb = self.rgb_encoder(rgb)
        rgb_features = tuple(all_rgb[self._rgb_indices.index(i)] for i in self.config['out_indices'])
        detail = all_rgb[self._rgb_indices.index(0)] if 0 in self._rgb_indices else None
        if self.config['modality']=='rgb':
            fused = tuple(p(r) for p,r in zip(self.rgb_projections,rgb_features))
        else:
            # Geometry and scan retain FP32 under AMP; RGB/decoder can autocast.
            amp = torch.is_autocast_enabled(rgb.device.type)
            with torch.autocast(rgb.device.type, enabled=False):
                if amp:
                    rgb_features = tuple(r.float() for r in rgb_features)
                    depth, K = depth.float(), K.float()
                geometry_features = self.geometry_encoder(depth,K)
                if self.quality_blocks is not None:
                    valid = torch.isfinite(depth)&(depth>0)
                    fused = tuple(block(r,d,valid) for block,r,d in zip(self.quality_blocks,rgb_features,geometry_features))
                else:
                    fused = tuple(self.rgb_projections[i](r) if i in self.config['skip_fusion_scales']
                                  else block(r,d)[0] for i,(block,r,d) in enumerate(zip(self.fusion.blocks,rgb_features,geometry_features)))
        return fused,detail

    def encode_fused(self,rgb,depth,K):
        return self._encode(rgb,depth,K)[0]

    def global_from_fused(self, fused, detail=None):
        decoded = self.decoder(fused,self.config['heatmap_size'])
        if self.detail_projection is not None:
            if detail is None:
                raise ValueError('Detail decoder needs the RGB stem feature')
            decoded = decoded+F.interpolate(self.detail_projection(detail),size=decoded.shape[-2:],mode='bilinear',align_corners=False)
        return self.head(decoded)

    def refine_cached(self, f1, global_maps, *, return_patches=False, detail=None):
        if self.residual_local is not None:
            feature = detail if self.config['local']['feature_source']=='detail' else f1
            if feature is None:
                raise ValueError('Residual local needs its configured feature source')
            residual = self.residual_local(global_maps,feature)
            return StageOutput(global_maps,None,residual.heatmaps,residual)
        local = (self.local_refinement(global_maps.detach(), f1,
                                       return_patches=return_patches)
                 if self.local_enabled else None)
        return StageOutput(global_maps, local,local.heatmaps if local else global_maps)

    def forward_stages(self, rgb, depth, K, *, return_patches=False):
        fused,detail = self._encode(rgb,depth,K)
        global_maps = self.global_from_fused(fused,detail)
        if self.residual_local is not None and self.phase=='global':
            return StageOutput(global_maps,None,global_maps)
        return self.refine_cached(fused[0], global_maps, return_patches=return_patches,detail=detail)

    def forward(self, rgb, depth, K):
        """Keep the global-only tensor API regardless of local configuration."""
        fused,detail = self._encode(rgb,depth,K)
        return self.global_from_fused(fused,detail)
