# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lap 
from mmdet.models import build_detector, build_head, build_backbone, build_neck
from collections import OrderedDict

import torch.distributed as dist
from mmtrack.core import outs2results, results2outs
# from mmtrack.models.mot import BaseMultiObjectTracker
from mmcv.runner import BaseModule, auto_fp16
from ..builder import MODELS, build_tracker

from mmdet.core import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy, reduce_mean
import copy

import torch.distributions as D
from .base import BaseMocapModel
from mmdet.models import build_loss
from cad.pos import AnchorEncoding
from cad.attn import ResCrossAttn, ResSelfAttn
# from cad.models.detr import DETRDecoder
from collections import defaultdict
from mmcv.cnn.bricks.registry import FEEDFORWARD_NETWORK
from mmcv import build_from_cfg
from ..builder import MODELS, build_tracker, build_model

@MODELS.register_module()
class SingleModalityModel(BaseModule):
    def __init__(self,
                 backbone_cfg=None,
                 neck_cfg=None,
                 cross_attn_cfg=dict(type='QKVAttention',
                     qk_dim=256,
                     num_heads=8, 
                     in_proj=True, 
                     out_proj=True,
                     attn_drop=0.1, 
                     seq_drop=0.0,
                     return_weights=False,
                     v_dim=None
                 ),
                 ffn_cfg=dict(type='SLP', in_channels=256),
                 bg_cfg=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        
        self.backbone = backbone_cfg
        if self.backbone is not None:
            self.backbone = build_backbone(backbone_cfg)

        self.neck = neck_cfg
        if self.neck is not None:
            self.neck = build_neck(neck_cfg)
        
        self.pos_encoding = AnchorEncoding(dim=256, learned=False, out_proj=False)
        self.cross_attn = ResCrossAttn(cross_attn_cfg)
        
        self.ffn = None
        if ffn_cfg is not None:
            self.ffn = build_from_cfg(ffn_cfg, FEEDFORWARD_NETWORK)

        self.bg_model = None
        if bg_cfg is not None:
            self.bg_model = build_model(bg_cfg)
        
    
    #def forward(self, data, return_loss=True, **kwargs):
    def forward(self, x):
        if self.backbone:
            feats = self.backbone(x)
        else:
            feats = x
        if self.neck:
            feats = self.neck(feats)
        if len(feats) > 1:
            target_shape = (feats[2].shape[2], feats[2].shape[3])
            feats = [F.interpolate(f, target_shape) for f in feats] 
            feats = torch.cat(feats, dim=1)
        else:
            feats = feats[0]
        feats = feats.permute(0, 2, 3, 1) #feat dim to end
        B, H, W, D = feats.shape

        pos_embeds = self.pos_encoding(None).unsqueeze(0)
        pos_embeds = pos_embeds.expand(B, -1, -1, -1)
        output_embeds = self.cross_attn(pos_embeds, feats)
        output_embeds = output_embeds.reshape(B, -1, D)
        if self.ffn is not None:
            output_embeds = self.ffn(output_embeds)
        return output_embeds
