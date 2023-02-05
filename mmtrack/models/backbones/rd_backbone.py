# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models.backbones.resnet import Bottleneck, ResNet
from mmdet.models.builder import BACKBONES
from mmcls.models.backbones.convnext import ConvNeXtBlock
from mmcv.runner import BaseModule, auto_fp16


@BACKBONES.register_module()
class RangeDopplerBackbone(BaseModule):
    def __init__(self, in_channels=1, 
                 out_channels=256,
                 norm_cfg=dict(type='BN')):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3),
            build_norm_layer(norm_cfg, out_channels)[1],
            ConvNeXtBlock(out_channels, layer_scale_init_value=0.0),
            nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=2, padding=3),
            build_norm_layer(norm_cfg, out_channels)[1],
            ConvNeXtBlock(out_channels, layer_scale_init_value=0.0),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1,7), stride=(1,2), padding=(0,3)),
            build_norm_layer(norm_cfg, out_channels)[1]
        )
    
    def forward(self, x):
        return (self.layers(x), )


