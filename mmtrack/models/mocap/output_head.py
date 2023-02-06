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
import torch.distributions as D

@MODELS.register_module()
class PoolingOutputHead(BaseModule):
    def __init__(self,
                 include_z=False,
                 predict_full_cov=True,
                 predict_rotation=False,
                 predict_velocity=False,
                 num_sa_layers=0,
                 num_objects=1,
                 input_dim=256,
                 mean_scale=[700,500],
                 to_cm=False,
                 cov_add=1,
                 mlp_dropout_rate=0.0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.include_z = include_z
        self.predict_full_cov = predict_full_cov
        self.predict_rotation = predict_rotation
        self.predict_velocity = predict_velocity
        self.predict_full_cov = predict_full_cov
        self.to_cm = to_cm
        #self.pooler = nn.AdaptiveAvgPool2d((1, num_objects))
        self.pooler = nn.AdaptiveMaxPool2d((1, num_objects))
        self.big_lin = nn.Linear(70*50, num_objects)


        # self.num_outputs = 2 + 1
        # if self.include_z:
            # self.num_outputs += 1
        
        # if predict_full_cov:
            # if self.include_z:
                # self.num_outputs += 9
            # else:
                # self.num_outputs += 3
        # else:
            # if self.include_z:
                # self.num_outputs += 3
            # else:
                # self.num_outputs += 2

        # if self.predict_rotation:
            # self.num_outputs += 9

        
        if include_z:
            self.register_buffer('cov_add', torch.eye(3) * cov_add)
        else:
            self.register_buffer('cov_add', torch.eye(2) * cov_add)

        self.register_buffer('mean_scale', torch.tensor(mean_scale))
         
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            #nn.Conv2d(input_dim, input_dim, kernel_size=7, padding=3, stride=2),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
            #nn.Conv2d(input_dim, input_dim, kernel_size=7, padding=3, stride=2),
            nn.GELU(),
            nn.Dropout(mlp_dropout_rate)
        )
        

        self.mean_head = nn.Linear(input_dim, 2)
        self.cov_head = nn.Linear(input_dim, 3)
        self.obj_prob_head = nn.Linear(input_dim, 1)
        
        self.num_outputs = 2 + 3 + 1
        
        if self.predict_rotation:
            self.rot_head = nn.Linear(input_dim, 2)
            self.num_outputs += 2

        if self.predict_velocity:
            self.vel_head = nn.Linear(input_dim, 2)
            self.num_outputs += 2

        output_sa_cfg=dict(type='QKVAttention',
             qk_dim=self.num_outputs,
             num_heads=1, 
             in_proj=True,
             out_proj=True,
             attn_drop=0.0, 
             seq_drop=0.0,
             return_weights=False,
             v_dim=None
        )
        
        if num_sa_layers > 0:
            self.output_sa = [ResSelfAttn(output_sa_cfg) for _ in range(num_sa_layers)]
            self.output_sa = nn.Sequential(*self.output_sa)
        else:
            self.output_sa = nn.Identity()

    
    #def forward(self, data, return_loss=True, **kwargs):
    #x has the shape B x num_object x D
    def forward(self, x):
        # x = self.pooler(x)[:, :, 0] #B C nO
        # x = x.permute(0, 2, 1)
        x = x.flatten(2)
        x = self.big_lin(x).transpose(-2,-1)
        #x = x.mean(dim=[2,3]).unsqueeze(0) #WARNING!!!!
        x = self.mlp(x)
        outputs = []
        result = {}
        outputs.append(self.mean_head(x))
        outputs.append(self.cov_head(x))
        outputs.append(self.obj_prob_head(x))
        if self.predict_rotation:
            outputs.append(self.rot_head(x))
        
        if self.predict_velocity:
            outputs.append(self.vel_head(x))

        outputs = torch.cat(outputs, dim=-1)
        outputs = self.output_sa(outputs)
        mean = outputs[..., 0:2]
        cov_logits = outputs[..., 2:5]
        obj_logits = outputs[..., 5]

        if self.predict_rotation:
            rot_logits = outputs[..., 5:7]
            rot = torch.stack([
                torch.sin(rot_logits[..., 0]),
                torch.cos(rot_logits[..., 1])
            ], dim=-1)
            result['rot'] = rot
        # mean = self.mean_head(x)
        # cov_logits = self.cov_head(x)
        
        # if self.include_z:
            # mean = output_vals[..., 0:3]
        # else:
            # mean = output_vals[..., 0:2]
        # if self.add_grid_to_mean:
            # mean[..., 0] += self.global_pos_encoding.unscaled_params_x.flatten()
            # mean[..., 1] += self.global_pos_encoding.unscaled_params_y.flatten()
        mean = mean.sigmoid()
        mean = mean * self.mean_scale
        
        cov_diag = F.softplus(cov_logits[..., 0:2])
        cov_off_diag = cov_logits[..., -1]
        cov = torch.diag_embed(cov_diag)
        cov[..., -1, 0] += cov_off_diag
        B, N, _, _ = cov.shape
        cov = cov.reshape(B*N, 2, 2)
        cov = torch.bmm(cov, cov.transpose(-2,-1))
        cov = cov.reshape(B, N, 2, 2)

        cov = cov + self.cov_add

        # if self.to_cm:
            # mean = mean*100
            # cov = cov*100

        result['dist'] = D.MultivariateNormal(mean, cov)
        result['obj_logits'] = obj_logits
        # if self.predict_rotation:
            # result['rot'] = self.rot_head(x).tanh()

        # if self.predict_velocity:
            # assert 1==2
            # result['vel'] = self.vel_head(x)
        return result


@MODELS.register_module()
class OutputHead(BaseModule):
    def __init__(self,
                 include_z=False,
                 predict_full_cov=True,
                 predict_rotation=False,
                 predict_velocity=False,
                 predict_obj_prob=False,
                 num_sa_layers=0,
                 input_dim=256,
                 mean_scale=[700,500],
                 to_cm=False,
                 cov_add=1,
                 mlp_dropout_rate=0.0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.include_z = include_z
        self.predict_full_cov = predict_full_cov
        self.predict_rotation = predict_rotation
        self.predict_velocity = predict_velocity
        self.predict_full_cov = predict_full_cov
        self.predict_obj_prob = predict_obj_prob
        self.to_cm = to_cm


        # self.num_outputs = 2 + 1
        # if self.include_z:
            # self.num_outputs += 1
        
        # if predict_full_cov:
            # if self.include_z:
                # self.num_outputs += 9
            # else:
                # self.num_outputs += 3
        # else:
            # if self.include_z:
                # self.num_outputs += 3
            # else:
                # self.num_outputs += 2

        # if self.predict_rotation:
            # self.num_outputs += 9

        
        if include_z:
            self.register_buffer('cov_add', torch.eye(3) * cov_add)
        else:
            self.register_buffer('cov_add', torch.eye(2) * cov_add)

        self.register_buffer('mean_scale', torch.tensor(mean_scale))
         
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout_rate)
        )
        

        self.num_outputs = 2 + 3 
        self.mean_head = nn.Linear(input_dim, 2)
        self.cov_head = nn.Linear(input_dim, 3)

        if self.predict_obj_prob:
            self.obj_prob_head = nn.Linear(input_dim, 1)
            self.num_outputs += 1
        
        
        if self.predict_rotation:
            self.rot_head = nn.Linear(input_dim, 2)
            self.num_outputs += 2

        if self.predict_velocity:
            self.vel_head = nn.Linear(input_dim, 2)
            self.num_outputs += 2

        output_sa_cfg=dict(type='QKVAttention',
             qk_dim=self.num_outputs,
             num_heads=1, 
             in_proj=True,
             out_proj=True,
             attn_drop=0.0, 
             seq_drop=0.0,
             v_dim=None
        )
        
        if num_sa_layers > 0:
            self.output_sa = [ResSelfAttn(output_sa_cfg) for _ in range(num_sa_layers)]
            self.output_sa = nn.Sequential(*self.output_sa)
        else:
            self.output_sa = nn.Identity()

    
    #def forward(self, data, return_loss=True, **kwargs):
    #x has the shape B x num_object x D
    def forward(self, x):
        x = self.mlp(x)
        outputs = []
        result = {}
        outputs.append(self.mean_head(x))
        outputs.append(self.cov_head(x))
        
        if self.predict_obj_prob:
            outputs.append(self.obj_prob_head(x))
        if self.predict_rotation:
            outputs.append(self.rot_head(x))
        
        if self.predict_velocity:
            outputs.append(self.vel_head(x))

        outputs = torch.cat(outputs, dim=-1)
        outputs = self.output_sa(outputs)
        mean = outputs[..., 0:2]
        cov_logits = outputs[..., 2:5]
        
        if self.predict_obj_prob:
            obj_logits = outputs[..., 5]
            result['obj_logits'] = obj_logits

        if self.predict_rotation:
            rot_logits = outputs[..., 5:7]
            rot = torch.stack([
                torch.sin(rot_logits[..., 0]),
                torch.cos(rot_logits[..., 1])
            ], dim=-1)
            result['rot'] = rot
        # mean = self.mean_head(x)
        # cov_logits = self.cov_head(x)
        
        # if self.include_z:
            # mean = output_vals[..., 0:3]
        # else:
            # mean = output_vals[..., 0:2]
        # if self.add_grid_to_mean:
            # mean[..., 0] += self.global_pos_encoding.unscaled_params_x.flatten()
            # mean[..., 1] += self.global_pos_encoding.unscaled_params_y.flatten()
        mean = mean.sigmoid()
        mean = mean * self.mean_scale
        
        cov_diag = F.softplus(cov_logits[..., 0:2])
        cov_off_diag = cov_logits[..., -1]
        cov = torch.diag_embed(cov_diag)
        cov[..., -1, 0] += cov_off_diag
        B, N, _, _ = cov.shape
        cov = cov.reshape(B*N, 2, 2)
        cov = torch.bmm(cov, cov.transpose(-2,-1))
        cov = cov.reshape(B, N, 2, 2)

        cov = cov + self.cov_add

        # if self.to_cm:
            # mean = mean*100
            # cov = cov*100

        result['dist'] = D.MultivariateNormal(mean, cov)
        
        # if self.predict_rotation:
            # result['rot'] = self.rot_head(x).tanh()

        # if self.predict_velocity:
            # assert 1==2
            # result['vel'] = self.vel_head(x)
        return result



        # if self.predict_full_cov and self.include_z:
            # cov = F.softplus(output_vals[..., 3:3+9])
            # cov = cov.view(B*Nt, L, 3,3).tril()
        # elif not self.predict_full_cov and self.include_z:
            # cov = F.softplus(output_vals[..., 3:6])
        # elif not self.predict_full_cov and not self.include_z:
            # cov = F.softplus(output_vals[..., 2:4])
            # cov = torch.diag_embed(cov)
        # elif self.predict_full_cov and not self.include_z:
            # cov = F.softplus(output_vals[..., 2:4])
            # cov = torch.diag_embed(cov)
            # cov[..., -1, 0] += output_vals[..., 4]
            # B, N, _, _ = cov.shape
            # cov = cov.reshape(B*N, 2, 2)
            # cov = torch.bmm(cov, cov.transpose(-2,-1))
            # cov = cov.reshape(B, N, 2, 2)
        # cov = cov + self.cov_add
        # obj_logits = output_vals[..., -1]
        # return dist, obj_logits
