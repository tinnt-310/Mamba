"""Additive FPN and RGB-D coarse-to-fine heads for controlled ablations."""
import torch
from torch import nn
from torch.nn import functional as F


def conv(width,depthwise=False):
    layers=[nn.Conv2d(width,width,3,padding=1,groups=width if depthwise else 1,bias=False)]
    if depthwise: layers.append(nn.Conv2d(width,width,1,bias=False))
    return nn.Sequential(*layers,nn.GroupNorm(1,width),nn.GELU())


class FPNDecoder(nn.Module):
    def __init__(self,channels,width):
        super().__init__()
        self.projections=nn.ModuleList(nn.Conv2d(c,width,1) for c in channels)
        self.refine=nn.ModuleList(conv(width,True) for _ in channels)

    def forward(self,features,output_hw,detail=None):
        current=None
        for i in reversed(range(len(features))):
            lateral=self.projections[i](features[i])
            if current is not None:
                lateral=lateral+F.interpolate(current,size=lateral.shape[-2:],mode='bilinear',align_corners=False)
            current=self.refine[i](lateral)
        return F.interpolate(current,size=output_hw,mode='bilinear',align_corners=False)


class CoarseToFineDecoder(nn.Module):
    def __init__(self,channels,width,detail_channels,keypoints):
        super().__init__()
        from .model import MeanMultiScaleDecoder
        self.coarse=MeanMultiScaleDecoder(channels,width)
        self.coarse_head=nn.Conv2d(width,keypoints,1)
        self.rgb_detail=nn.Conv2d(detail_channels[0],width,1)
        self.depth_detail=nn.Conv2d(detail_channels[1],width,1)
        self.detail_gate=nn.Conv2d(2*width,width,1)
        nn.init.zeros_(self.detail_gate.weight); nn.init.zeros_(self.detail_gate.bias)
        self.combine=nn.Conv2d(2*width+keypoints,width,1)
        self.refine=nn.Sequential(conv(width),conv(width))

    def forward(self,features,output_hw,detail=None):
        if detail is None or len(detail)!=2:
            raise ValueError('D2 requires both shallow RGB and geometry features')
        coarse_hw=tuple((n+1)//2 for n in output_hw)
        coarse=self.coarse(features,coarse_hw)
        heatmaps=self.coarse_head(coarse)
        rgb,depth=self.rgb_detail(detail[0]),self.depth_detail(detail[1])
        if getattr(self,'mask_rgb',False): rgb=rgb*0
        if getattr(self,'mask_depth',False): depth=depth*0
        gate=self.detail_gate(torch.cat((rgb,depth),1)).sigmoid()
        fused=gate*rgb+(1-gate)*depth
        up=lambda x:F.interpolate(x,size=output_hw,mode='bilinear',align_corners=False)
        fine=self.refine(self.combine(torch.cat((up(coarse),up(heatmaps),up(fused)),1)))
        return fine,heatmaps
