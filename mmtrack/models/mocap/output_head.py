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

def shift(x, a, b):
    return a + (b - a) * x

def generate_intervals(length, interval_size=100):
    intervals = torch.stack([
        torch.arange(0, length, interval_size),
        torch.arange(interval_size, length + interval_size, interval_size)
    ]).t()
    return intervals

@MODELS.register_module()
class AnchorOutputHead(BaseModule):
    def __init__(self,
                 include_z=False,
                 predict_full_cov=True,
                 predict_rotation=False,
                 predict_velocity=False,
                 predict_obj_prob=False,
                 num_sa_layers=0,
                 input_dim=256,
                 room_size=[700,500,100],
                 interval_sizes=[700,500],
                 to_cm=False,
                 cov_add=1,
                 mlp_dropout_rate=0.0,
                 cov_only_train=False,
                 binary_prob=False,
                 scale_binary_prob=False,
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
        self.return_raw = False
        self.cov_only_train = cov_only_train
        self.binary_prob = binary_prob
        self.scale_binary_prob = scale_binary_prob
        if self.binary_prob and self.scale_binary_prob:
            self.alpha = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.zeros(1))

        x_intervals = generate_intervals(room_size[0], interval_sizes[0])
        y_intervals = generate_intervals(room_size[1], interval_sizes[1])
        self.register_buffer('x_intervals', x_intervals)
        self.register_buffer('y_intervals', y_intervals)
        
        self.room_size = room_size

        
        if include_z:
            self.register_buffer('cov_add', torch.eye(3) * cov_add)
        else:
            self.register_buffer('cov_add', torch.eye(2) * cov_add)

        # self.register_buffer('mean_scale', torch.tensor(mean_scale))
         
        self.mlp = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(mlp_dropout_rate)
        )

        self.num_outputs = 2 + 3 
        self.mean_head = nn.Conv2d(input_dim, 2, kernel_size=1)
        self.cov_head = nn.Conv2d(input_dim, 3, kernel_size=1)
        self.mix_head = nn.Conv2d(input_dim, 1, kernel_size=1)
        self.z_head = nn.Conv2d(input_dim, 1, kernel_size=1)

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
    def forward(self, x, node_pos, node_rot):
        with torch.set_grad_enabled(not self.cov_only_train):
            x = self.mlp(x)
            means = self.mean_head(x)
            mix_logits = self.mix_head(x)
            z_vals = self.z_head(x)[0]
            z_vals = z_vals.sigmoid() * self.room_size[-1]

            means = means.permute(0, 2, 3, 1)
                    
            means = means.sigmoid()

            x_vals = []
            for i, xi in enumerate(self.x_intervals):
                x_vals.append(shift(means[:, i, :, 0], xi[0], xi[1]))
            
            y_vals = []
            for i, yi in enumerate(self.y_intervals):
                y_vals.append(shift(means[:, :, i, 1], yi[0], yi[1]))

            x_vals = torch.stack(x_vals, dim=1)
            y_vals = torch.stack(y_vals, dim=-1)
            
            means = torch.stack([x_vals, y_vals, z_vals], dim=-1)
            _, H, W, _ = means.shape
            means = means.reshape(H*W, 3)

            node_rot = node_rot.reshape(3,3).t().cuda()
            node_pos = node_pos.unsqueeze(-1).cuda()

            means = torch.matmul(node_rot, means.t()) + node_pos
            means = means.t().reshape(1, H, W, 3)
            means = means[..., 0:2]

            mix_logits = mix_logits.flatten()

        outputs = []
        result = {}

                
        cov_logits = self.cov_head(x)
        cov_logits = cov_logits.permute(0, 2, 3, 1)

        cov_diag = F.softplus(cov_logits[..., 0:2])
        cov_off_diag = cov_logits[..., -1]
        
        eye = torch.eye(2).cuda()
        cov_diag = cov_diag.unsqueeze(-1) * eye

        reye = torch.flip(eye, dims=[1])
        cov_off_diag = cov_off_diag.unsqueeze(-1).unsqueeze(-1) * reye
        
        cov = cov_diag + cov_off_diag

        # cov = torch.diag_embed(cov_diag)
        # cov[..., -1, 0] += cov_off_diag
        
        B, H, W, _, _ = cov.shape
        cov = cov.reshape(B*H*W, 2, 2)
        cov = torch.bmm(cov, cov.transpose(-2,-1))
        cov = cov.reshape(B, H, W, 2, 2)

        if self.binary_prob:
            if self.scale_binary_prob:
                binary_probs = self.alpha * mix_logits + self.beta
            result['binary_probs'] = binary_probs.detach().cpu()
            binary_probs = binary_probs.sigmoid()
            binary_probs = binary_probs.reshape(B, H, W, 1, 1)
            cov = binary_probs * cov + (1 - binary_probs) * self.cov_add
        else: 
            cov = cov + self.cov_add

        means = means.reshape(B, H*W, 2)
        cov = cov.reshape(B, H*W, 2, 2)

        if self.return_raw:
            return means, cov, mix_logits.unsqueeze(0)

        mix_weights = torch.softmax(mix_logits, dim=0)
        normals = D.MultivariateNormal(means, cov)
        
        #mix = D.Categorical(torch.ones(35,).cuda())
        mix = D.Categorical(probs=mix_weights)
        dist = D.MixtureSameFamily(mix, normals)

        

        # if self.to_cm:
            # mean = mean*100
            # cov = cov*100

        result['dist'] = dist
        result['grid_size'] = (H, W)
        
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
        self.return_raw = False


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
        if self.return_raw:
            assert len(mean) == 1
            I = torch.eye(2).cuda()
            cov_diag = I * F.softplus(cov_logits[..., 0:2]).squeeze()
            cov_off_diag = cov_logits[..., -1]
            rI = torch.tensor([[0,1],[1,0]]).float().cuda()
            cov_off_diag = rI * cov_logits.squeeze()[-1]
            cov = cov_off_diag + cov_diag
            cov = cov @ cov.t()
            cov = cov + I
            return mean, cov
        
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
