# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models.backbones.resnet import Bottleneck, ResNet
from mmdet.models.builder import BACKBONES
from mmcls.models.backbones.convnext import ConvNeXtBlock
from mmcv.runner import BaseModule, auto_fp16
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
import torch

#@BACKBONES.register_module()
class ResNet50Stem(nn.Sequential):
    def __init__(self, frozen=True):
        r50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        super().__init__(
            r50.conv1,
            r50.bn1,
            r50.relu,
            r50.maxpool
        )
        self.frozen = frozen
        if self.frozen:
            self.forward = self._forward_frozen
    
    @torch.no_grad()
    def _forward_frozen(self, x):
        for layer in self:
            x = layer.eval()(x)
        return x

@BACKBONES.register_module()
class TVResNet50(BaseModule):
    def __init__(self, 
                 out_channels=256,
                 norm_cfg=dict(type='BN')
        ):
        super().__init__()
        self.stem = ResNet50Stem(frozen=True)
        self.layers = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=7, stride=2, padding=3),
            build_norm_layer(norm_cfg, out_channels)[1],

            ConvNeXtBlock(out_channels, layer_scale_init_value=0.0),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=2, padding=3),
            build_norm_layer(norm_cfg, out_channels)[1],
            
            ConvNeXtBlock(out_channels, layer_scale_init_value=0.0),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=2, padding=3),
            build_norm_layer(norm_cfg, out_channels)[1]
        )
               
    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        return (x, )
