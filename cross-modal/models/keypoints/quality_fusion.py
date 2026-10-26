"""RGB identity path with explicitly masked depth residual compensation."""
import torch
from torch import nn
from torch.nn import functional as F
from models.fusion.rgbd_compensation import RGBDCompensation


class QualityFusion(nn.Module):
    def __init__(self,rgb_channels,depth_channels,channels,**ssm):
        super().__init__()
        self.rgb = nn.Conv2d(rgb_channels,channels,1)
        self.cross = RGBDCompensation(rgb_channels,depth_channels,channels,**ssm)
        self.quality = nn.Sequential(nn.Conv2d(channels+1,channels,1),nn.GELU(),nn.Conv2d(channels,1,1))
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self,rgb,depth,validity):
        base = self.rgb(rgb)
        valid = F.interpolate(validity.float(),size=rgb.shape[-2:],mode='area')
        cross,_ = self.cross(rgb,depth)
        gate = torch.sigmoid(self.quality(torch.cat((base,valid),1)))
        return base+self.scale.tanh()*valid*gate*cross
