# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from collections import defaultdict
from ..builder import MODELS
from einops.layers.torch import Rearrange

class Interpolate(nn.Module):
    def __init__(self, size=(1, 1)):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size)

@MODELS.register_module()
class ConvAdapter(BaseModule):
    def __init__(self,
                 dim=256,
                 interpolate_size=(1,1),
                 interpolate_fn='interpolate',
                 transpose=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        if interpolate_fn == 'interpolate':
            interp_layer = Interpolate(interpolate_size) 
        elif interpolate_fn == 'avgpool':
            interp_layer = nn.AdaptiveAvgPool2d(interpolate_size) 
        self.layers = [
            nn.Conv2d(dim, dim, kernel_size=1, stride=(1,1), padding=(0,0)),
            nn.GELU(),
        ]
        if transpose:
            self.layers.append(Rearrange('b c h w -> b c w h')) 
        self.layers.append(interp_layer)
        self.layers.append(nn.Conv2d(dim, dim, kernel_size=1, stride=(1,1), padding=(0,0)))
        self.layers.append(nn.GELU())
        self.layers = nn.Sequential(*self.layers)
    
    #x has shape B x in_len x D
    def forward(self, x, pos_embeds=None):
        return self.layers(x)

