"""Four-direction RGB-D search blocks; legacy parameter paths stay untouched."""
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
from .rgbd_compensation import RGBDCompensation


class WindowCrossC(nn.Module):
    """Batch equal-sized valid windows; padding never enters a recurrence."""
    def __init__(self, block, size=8):
        super().__init__()
        self.block, self.size = block, size

    def forward(self, rgb, depth):
        b,c,h,w = rgb.shape
        size = self.size
        nh,nw = (h+size-1)//size,(w+size-1)//size
        def partition(x):
            x=F.pad(x,(0,nw*size-w,0,nh*size-h))
            return x.reshape(b,c,nh,size,nw,size).permute(0,2,4,1,3,5).reshape(-1,c,size,size)
        r,d=partition(rgb),partition(depth)
        groups=defaultdict(list)
        for batch in range(b):
            for y in range(nh):
                for x in range(nw):
                    groups[min(size,h-y*size),min(size,w-x*size)].append((batch*nh+y)*nw+x)
        result={}
        for (hh,ww),ids in groups.items():
            indices=torch.tensor(ids,device=rgb.device)
            outputs=self.block(r[indices,:,:hh,:ww],d[indices,:,:hh,:ww])
            for key,value in outputs.items():
                padded=F.pad(value,(0,size-ww,0,size-hh))
                if key not in result:
                    result[key]=padded.new_zeros(b*nh*nw,padded.shape[1],size,size)
                result[key]=result[key].index_copy(0,indices,padded)
        return {key:value.reshape(b,nh,nw,-1,size,size).permute(0,3,1,4,2,5)
                .reshape(b,-1,nh*size,nw*size)[:,:,:h,:w] for key,value in result.items()}


class SearchRGBDCompensation(RGBDCompensation):
    def __init__(self,*args,topology='g0',window_size=8,**kwargs):
        super().__init__(*args,**kwargs)
        if self.cross_parametric.directions != 4:
            raise ValueError('GDL Global requires real four-direction scans')
        self.topology=topology
        if topology=='g1':
            self.cross_parametric=WindowCrossC(self.cross_parametric,window_size)
        if topology=='g2':
            channels=self.cross_parametric.channels
            self.cnn=nn.Sequential(nn.Conv2d(2*channels,channels,1),nn.GELU(),
                nn.Conv2d(channels,channels,3,padding=1,groups=channels),nn.GELU())
            self.cnn_output=nn.Conv2d(channels,channels,1,bias=False)
            self.cnn_gate=nn.Conv2d(2*channels,channels,1)
            nn.init.zeros_(self.cnn_output.weight)
            nn.init.zeros_(self.cnn_gate.weight); nn.init.zeros_(self.cnn_gate.bias)

    def forward(self,rgb,depth,validity=None):
        fused,aux=super().forward(rgb,depth,validity)
        if self.topology=='g2':
            cnn=self.cnn(torch.cat((self.rgb_project(rgb),self.depth_project(depth)),1))
            gate=self.cnn_gate(torch.cat((fused,cnn),1)).sigmoid()
            if getattr(self,'mask_cnn',False): cnn=cnn*0
            fused=fused+gate*self.cnn_output(cnn)
            aux['cnn_gate']=gate
        return fused,aux


class SearchMultiScaleFusion(nn.Module):
    def __init__(self,rgb_channels,depth_channels,channels,*,topology,window_size=8,**kwargs):
        super().__init__()
        self.blocks=nn.ModuleList(SearchRGBDCompensation(r,d,c,topology=(
            'g0' if topology=='g1' and i>=2 else topology),window_size=window_size,**kwargs)
            for i,(r,d,c) in enumerate(zip(rgb_channels,depth_channels,channels)))
