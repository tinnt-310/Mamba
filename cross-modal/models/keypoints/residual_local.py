"""Batched semantic residual refinement on the global grid, without CPU ROIs."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .local_refinement import FourDirectionLocalMamba


@dataclass(frozen=True)
class ResidualOutput:
    heatmaps: torch.Tensor
    delta: torch.Tensor
    support: torch.Tensor
    gate: torch.Tensor


class ResidualLocal(nn.Module):
    def __init__(self, in_channels, channels, num_keypoints, *, roi_size=13,
                 conditioning=False, state_dim=16, backend='torch'):
        super().__init__()
        self.roi_size, self.num_keypoints = roi_size, num_keypoints
        self.conditioning = conditioning
        self.project = nn.Conv2d(in_channels,channels,1)
        self.embedding = nn.Embedding(num_keypoints,channels)
        self.context = nn.Conv2d(3,channels,1) if conditioning else None
        self.mamba = FourDirectionLocalMamba(channels,state_dim,backend)
        self.delta_head = nn.Conv2d(channels,1,1)
        self.gate = nn.Parameter(torch.zeros(num_keypoints))
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, global_maps, feature, *, gate_override=None):
        b,k,h,w = global_maps.shape
        if k != self.num_keypoints or feature.shape[0] != b:
            raise ValueError('Residual batch/semantic mismatch')
        size, radius = self.roi_size, self.roi_size//2
        centers = global_maps.detach().flatten(-2).argmax(-1).reshape(-1)
        yy,xx = torch.meshgrid(torch.arange(size,device=feature.device)-radius,
                              torch.arange(size,device=feature.device)-radius,indexing='ij')
        x = centers.remainder(w)[:,None,None]+xx
        y = centers.div(w,rounding_mode='floor')[:,None,None]+yy
        valid = (x>=0)&(x<w)&(y>=0)&(y<h)
        grid = torch.stack(((x+.5)*2/w-1,(y+.5)*2/h-1),-1).to(feature.dtype)
        projected = self.project(feature)
        # Pack K grids along height; sample the full image feature only once.
        packed = grid.reshape(b,k*size,size,2)
        crops = F.grid_sample(projected,packed,align_corners=False,padding_mode='zeros')
        crops = crops.reshape(b,projected.shape[1],k,size,size).permute(0,2,1,3,4).reshape(b*k,projected.shape[1],size,size)
        ids = torch.arange(k,device=feature.device).repeat(b)
        crops = crops+self.embedding(ids)[:,:,None,None]
        if self.context is not None:
            heat = F.grid_sample(global_maps.detach().reshape(b*k,1,h,w),grid,
                                 align_corners=False,padding_mode='zeros')
            xy = torch.stack((xx,yy),0).to(feature.dtype)/max(radius,1)
            crops = crops+self.context(torch.cat((heat,xy[None].expand(b*k,-1,-1,-1)),1))
        patch_delta = self.delta_head(self.mamba(crops)).reshape(b*k,-1)
        gates = self.gate.sigmoid().repeat(b) if gate_override is None else torch.as_tensor(
            gate_override,device=feature.device,dtype=feature.dtype).expand(b*k)
        indices = (y.clamp(0,h-1)*w+x.clamp(0,w-1)).reshape(b*k,-1)
        values = patch_delta*valid.reshape(b*k,-1)*gates[:,None]
        delta = values.new_zeros(b*k,h*w).scatter_add(1,indices,values).reshape(b,k,h,w)
        coverage = torch.zeros(b*k,h*w,device=feature.device,dtype=torch.int32).scatter_add(
            1,indices,valid.reshape(b*k,-1).to(torch.int32))
        support = coverage.reshape(b,k,h,w)>0
        return ResidualOutput(global_maps+delta,delta,support,gates.reshape(b,k))
