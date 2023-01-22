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



@BACKBONES.register_module()
class TVResNet50(BaseModule):
    def __init__(self, in_channels=1, 
                 out_channels=256,
                 norm_cfg=dict(type='BN')):
        super().__init__()
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layers = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=7, stride=2, padding=3),
            build_norm_layer(norm_cfg, out_channels)[1],
            ConvNeXtBlock(out_channels, layer_scale_init_value=0.0),
            nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=2, padding=3),
            build_norm_layer(norm_cfg, out_channels)[1],
            ConvNeXtBlock(out_channels, layer_scale_init_value=0.0),
            nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=2, padding=3),
            build_norm_layer(norm_cfg, out_channels)[1]
        )

    
               
    def forward(self, x):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        x = self.layers(x)
        return (x, )

        try:
            return (self.layers(x), )
        except:
            import ipdb; ipdb.set_trace() # noqa


