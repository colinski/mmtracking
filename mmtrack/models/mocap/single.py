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
from cad.pos import AnchorEncoding, SineEncoding2d
from cad.attn import ResCrossAttn, ResSelfAttn
from cad.models.detr import DETRDecoder
from collections import defaultdict
from mmcv.cnn.bricks.registry import FEEDFORWARD_NETWORK
from mmcv import build_from_cfg
from ..builder import MODELS, build_tracker, build_model
from cad.pos.sine import SineEncoding2d

@MODELS.register_module()
class DETRModalityModel(BaseModule):
    def __init__(self,
                 backbone_cfg=None,
                 neck_cfg=None,
                 decoder_cfg=dict(type='DETRDecoder',
                    num_layers=2,
                    self_attn_cfg=dict(type='ResSelfAttn', attn_cfg=dict(type='QKVAttention', qk_dim=256, num_heads=8)),
                    cross_attn_cfg=dict(type='ResCrossAttn', attn_cfg=dict(type='QKVAttention', qk_dim=256, num_heads=8)),
                    ffn_cfg=dict(type='SLP', in_channels=256),
                    return_all_layers=False,
                 ),
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = build_from_cfg(decoder_cfg, FEEDFORWARD_NETWORK)
        
        
        self.backbone = backbone_cfg
        if self.backbone is not None:
            self.backbone = build_backbone(backbone_cfg)

        self.neck = neck_cfg
        if self.neck is not None:
            self.neck = build_neck(neck_cfg)
        
        self.feat_pos_encoder = SineEncoding2d(dim=256)
        self.anchor_encoder = AnchorEncoding(dim=256, learned=True, out_proj=True)
        
                
    
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
        feats_pos = self.feat_pos_encoder(feats)

        anchor_pos = self.anchor_encoder(None).unsqueeze(0)
        anchor_pos = anchor_pos.expand(B, -1, -1, -1)
        anchor_embeds = torch.zeros_like(anchor_pos)
        
        output_embeds = self.decoder(anchor_embeds, anchor_pos, feats, feats_pos)
        output_embeds = output_embeds.reshape(B, -1, D)
        return output_embeds

@MODELS.register_module()
class LinearEncoder(BaseModule):
    def __init__(self,
                 in_len=100,
                 out_len=1,
                 # in_dim=4,
                 # out_dim=6,
                 ffn_cfg=dict(type='SLP', in_channels=256),
                 use_pos_encodings=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lin_len = nn.Linear(in_len, out_len)
        #self.lin_dim = nn.Linear(in_dim, out_dim)
        self.ffn = build_from_cfg(ffn_cfg, FEEDFORWARD_NETWORK)
        self.use_pos_encodings = use_pos_encodings
        if use_pos_encodings:
            self.pos_encodings = SineEncoding2d(dim=256)
    
    #x has shape B x in_len x D
    def forward(self, x, pos_embeds=None):
        if self.use_pos_encodings:
            x = x.permute(0,2,3,1)
            encodings = self.pos_encodings.encode(x)
            x = torch.cat([x, encodings], dim=-1)
            x = x.permute(0,3,1,2)

        if len(x.shape) == 4: #cov feat map
            x = x.flatten(2)
            x = x.permute(0, 2, 1)
        x = self.ffn(x)
        x = x.permute(0, 2, 1)
        x = self.lin_len(x)
        x = x.permute(0, 2, 1)
        #x = self.lin_dim(x)
        return x

@MODELS.register_module()
class ModalityEncoder(BaseModule):
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
                     v_dim=None
                 ),
                 feat_pos_grid_size=(9,15),
                 ffn_cfg=None,
                 output_style='embeds',
                 bg_cfg=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.output_style = output_style
        
        self.backbone = backbone_cfg
        if self.backbone is not None:
            self.backbone = build_backbone(backbone_cfg)

        self.neck = neck_cfg
        if self.neck is not None:
            self.neck = build_neck(neck_cfg)
        
        self.room_pos_encoding = AnchorEncoding(dim=256, grid_size=(5, 7), learned=False, out_proj=False)
        self.feat_pos_encoding = AnchorEncoding(dim=256, grid_size=feat_pos_grid_size, learned=False, out_proj=False)
        self.cross_attn = ResCrossAttn(cross_attn_cfg)
        
        self.ffn = None
        if ffn_cfg is not None:
            self.ffn = build_from_cfg(ffn_cfg, FEEDFORWARD_NETWORK)

        self.bg_model = None
        if bg_cfg is not None:
            self.bg_model = build_model(bg_cfg)
        
    
    #def forward(self, data, return_loss=True, **kwargs):
    def forward(self, x, pos_embeds=None):
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
        
        room_pos_embeds = self.room_pos_encoding(None).unsqueeze(0)
        room_pos_embeds = room_pos_embeds.expand(B, -1, -1, -1)
        feat_pos_embeds = self.feat_pos_encoding(None).unsqueeze(0)
        feat_pos_embeds = feat_pos_embeds.expand(B, -1, -1, -1)
        
        #detr trick
        room_feats = torch.zeros_like(room_pos_embeds) 
        output_embeds = self.cross_attn(room_feats, feats, x_pos=room_pos_embeds, feats_pos=feat_pos_embeds)
        
        output_embeds = output_embeds.reshape(B, -1, D)
        if self.ffn is not None:
            output_embeds = self.ffn(output_embeds)
        return output_embeds

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
                 output_style='embeds',
                 bg_cfg=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.output_style = output_style
        
        
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
    def forward(self, x, pos_embeds=None):
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
        
        if pos_embeds is None:
            pos_embeds = self.pos_encoding(None).unsqueeze(0)
            pos_embeds = pos_embeds.expand(B, -1, -1, -1)
        if self.output_style == 'embeds':
            output_embeds = self.cross_attn(pos_embeds, feats)
        elif self.output_style == 'feats':
            output_embeds = self.cross_attn(feats, pos_embeds)
        else:
            print('output_style must be embeds or feats')
            assert 1==2
        output_embeds = output_embeds.reshape(B, -1, D)
        if self.ffn is not None:
            output_embeds = self.ffn(output_embeds)
        return output_embeds
