# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
import lap 
from mmdet.models import build_detector, build_head, build_backbone, build_neck
from collections import OrderedDict
import torch.distributed as dist
from mmtrack.core import outs2results, results2outs
from mmcv.runner import BaseModule, auto_fp16
from ..builder import MODELS, build_tracker, build_model
from mmdet.core import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy, reduce_mean
import copy
import torch.distributions as D
from .base import BaseMocapModel
from mmdet.models import build_loss
from cad.pos import AnchorEncoding
from cad.attn import ResCrossAttn, ResSelfAttn
from cad.models.detr import DETRDecoder
from collections import defaultdict
from mmtrack.models.mot.kalman_track import MocapTrack
import time
from mmcv.cnn.bricks.registry import FEEDFORWARD_NETWORK
from mmcv import build_from_cfg
from pyro.contrib.tracking.measurements import PositionMeasurement

def linear_assignment(cost_matrix):
    cost_matrix = cost_matrix.cpu().detach().numpy()
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    assign_idx = np.array([[y[i], i] for i in x if i >= 0])
    sort_idx = np.argsort(assign_idx[:, 1])
    assign_idx = assign_idx[sort_idx]
    assign_idx = torch.from_numpy(assign_idx)
    return assign_idx.long()

class Tracker:
    def __init__(self, max_age=5, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.frame_count = 0 
        self.obj_prob_thres = 0.5

    def __call__(self, result):
        obj_probs = result['det_obj_probs']
        is_obj = obj_probs > self.obj_prob_thres
        means = result['det_means'][is_obj]
        covs = result['det_covs'][is_obj]
        slot_ids = torch.arange(len(obj_probs))[is_obj]
        if len(means) == 0:
            result.update({
                'track_means': torch.empty(0,3).cpu(),
                'track_covs': torch.empty(0,3,3).cpu(),
                'track_ids': torch.empty(0,1),
                'slot_ids': torch.empty(0,1)
            })
            return result

        self.frame_count += 1
        
        #get predictions from existing tracks
        for track in self.tracks:
            track.predict()
        
        log_probs = torch.zeros(len(self.tracks), len(means))
        for i, track in enumerate(self.tracks):
            for j, mean in enumerate(means):
                m = PositionMeasurement(means[j], covs[j], time=track.kf.time)
                log_probs[i, j] = track.kf.log_likelihood_of_update(m)
        
        if len(log_probs) == 0: #no tracks yet
            for j in range(len(means)):
                new_track = MocapTrack(means[j], covs[j])
                self.tracks.append(new_track)
                self.tracks[-1].slot_id = slot_ids[j]
        else:
            exp_probs = log_probs.exp()
            unassigned, assign_idx = [], []
            assign_idx = linear_assignment(-log_probs)
            for t, d in assign_idx:
                if exp_probs[t,d] >= 1e-16:
                    self.tracks[t].update(means[d], covs[d])
                    self.tracks[t].slot_id = slot_ids[d]
                else:
                    unassigned.append(d)
            for d in unassigned:
                new_track = MocapTrack(means[d], covs[d])
                self.tracks.append(new_track)
                self.tracks[-1].slot_id = slot_ids[d]

        ndim = len(means[0])
        track_means, track_covs, track_ids = [means.new_empty(0,ndim).cpu()], [means.new_empty(0,ndim,ndim).cpu()], []
        slots = []
        for t, track in enumerate(self.tracks):
            onstreak = track.hit_streak >= self.min_hits
            warmingup = self.frame_count <= self.min_hits
            if track.wasupdated and (onstreak or warmingup):
                track_means.append(track.mean.unsqueeze(0))
                track_covs.append(track.cov.unsqueeze(0))
                track_ids.append(track.id)
                slots.append(track.slot_id)
        
        track_means = torch.cat(track_means)
        track_covs = torch.cat(track_covs)
        track_ids = torch.tensor(track_ids)
        slots = torch.tensor(slots)

        self.tracks = [track for track in self.tracks\
                       if track.time_since_update < self.max_age]

        result.update({
            'track_means': track_means.detach().cpu(),
            'track_covs': track_covs.detach().cpu(),
            'track_ids': track_ids,
            'slot_ids': slot_ids
        })
        return result


@MODELS.register_module()
class DecoderMocapModel(BaseMocapModel):
    def __init__(self,
                 backbone_cfgs=None,
                 model_cfgs=None,
                 bce_target=0.99,
                 remove_zero_at_train=True,
                 cross_attn_cfg=dict(type='QKVAttention',
                     qk_dim=256,
                     num_heads=8, 
                     in_proj=True, 
                     out_proj=True,
                     attn_drop=0.0, 
                     seq_drop=0.0,
                     return_weights=False,
                     v_dim=None
                 ),
                 fusion_sa_cfg=dict(type='QKVAttention',
                     qk_dim=256,
                     num_heads=8, 
                     in_proj=True,
                     out_proj=True,
                     attn_drop=0.1, 
                     seq_drop=0.0,
                     return_weights=False,
                     v_dim=None
                 ),
                 fusion_ffn_cfg=dict(type='SLP', in_channels=256),
                 predict_corners=False,
                 num_output_sa_layers=6,
                 max_age=5,
                 min_hits=3,
                 track_eval=False,
                 mse_loss_weight=0.0,
                 pos_loss_weight=0.1,
                 mean_scale=[7,5,1],
                 predict_full_cov=False,
                 grid_loss=False,
                 include_z=True,
                 add_grid_to_mean=True,
                 num_queries=None,
                 grid_size=(10,10),
                 match_by_id=False,
                 autoregressive=False,
                 global_ca_layers=1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = Tracker()
        self.num_classes = 2
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = None
        self.frame_count = 0 
        self.track_eval = track_eval
        self.mse_loss_weight = mse_loss_weight
        self.pos_loss_weight = pos_loss_weight
        self.grid_loss = grid_loss
        self.include_z = include_z
        self.remove_zero_at_train = remove_zero_at_train
        self.bce_target = bce_target
        self.register_buffer('mean_scale', torch.tensor(mean_scale))
        self.add_grid_to_mean = add_grid_to_mean
        self.match_by_id = match_by_id
        self.autoregressive = autoregressive

        self.num_outputs = 2 + 1
        if self.include_z:
            self.num_outputs += 1
        if predict_corners:
            self.num_outputs += 4*2
        
        self.predict_full_cov = predict_full_cov
        if predict_full_cov:
            # self.corr_coef = nn.Parameter(torch.zeros(1)+0.1)
            if self.include_z:
                self.num_outputs += 9
            else:
                self.num_outputs += 3
        else:
            if self.include_z:
                self.num_outputs += 3
            else:
                self.num_outputs += 2

        # self.num_outputs += 8
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
        time_attn_cfg=dict(type='QKVAttention',
            qk_dim=self.num_outputs,
            num_heads=1, 
            in_proj=True,
            out_proj=True,
            attn_drop=0.0, 
            seq_drop=0.0,
            return_weights=False,
            v_dim=None
        )
        
        # if not self.add_grid_to_mean:
            # self.output_scale = nn.Linear(self.num_outputs, self.num_outputs)

        
        self.time_attn = None
        if time_attn_cfg is not None:
            self.time_attn = [ResSelfAttn(time_attn_cfg) for _ in range(6)]
            self.time_attn = nn.Sequential(*self.time_attn)

        if self.autoregressive:
            ar_attn_cfg=dict(type='QKVAttention',
                qk_dim=256,
                num_heads=8,
                in_proj=True,
                out_proj=True,
                attn_drop=0.1, 
                seq_drop=0.0,
                return_weights=False,
                v_dim=None
            )
            self.ar_cross_attn = [ResCrossAttn(ar_attn_cfg) for _ in range(6)]
            self.ar_cross_attn = nn.ModuleList(self.ar_cross_attn)


        self.backbones = nn.ModuleDict()
        for key, cfg in backbone_cfgs.items():
            self.backbones[key] = build_backbone(cfg)
        
        self.models = nn.ModuleDict()

        for key, cfg in model_cfgs.items():
            mod, node = key
            self.models[mod + '_' + node] = build_model(cfg)
        
        self.mse_loss = nn.MSELoss(reduction='none')
        
        self.predict_corners = predict_corners
        if predict_corners:
            self.corner_loss = nn.MSELoss(reduction='none')
        
        self.num_queries = num_queries
        if self.num_queries is not None:
            self.global_pos_encoding = nn.Embedding(self.num_queries, 256)
        else: 
            self.global_pos_encoding = AnchorEncoding(dim=256, grid_size=grid_size, learned=False, out_proj=False)
        
        self.global_ca_layers = global_ca_layers
        if global_ca_layers > 1:
            self.global_cross_attn = nn.ModuleList([ResCrossAttn(cross_attn_cfg)]*global_ca_layers)
        else:
            self.global_cross_attn = ResCrossAttn(cross_attn_cfg)
        
        
        self.ctn = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )
        self.output_sa = [ResSelfAttn(output_sa_cfg) for _ in range(6)]
        self.output_sa = nn.Sequential(*self.output_sa)
        self.output_head = nn.Linear(256, self.num_outputs)
        self.bce_loss = nn.BCELoss(reduction='none')

    def dist(self, mean, cov):
        if self.predict_full_cov:
            #dist = D.MultivariateNormal(mean, scale_tril=cov)
            dist = D.MultivariateNormal(mean, cov)
        else:
            dist = D.MultivariateNormal(mean, cov)
            # dist = D.Normal(mean, cov)
            # dist = D.Independent(dist, 1)
        return dist

    
    def forward(self, data, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(data, **kwargs)
        else:
            if self.track_eval:
                return self.forward_track(data, **kwargs)
            else:
                return self.forward_test(data, **kwargs)
    
    def _ar_forward_test(self, datas, return_unscaled=False, **kwargs):
        if self.tracks is None:
            self.tracks = self.global_pos_encoding.weight.unsqueeze(0) # 1 x L x D



        self.tracks = self._forward_single_ar(datas[-1], self.tracks)

        det_mean, det_cov, det_obj_probs = self.convert(self.ctn(self.tracks))
        det_mean, det_cov, det_obj_probs = det_mean[0], det_cov[0], det_obj_probs[0]
        
        result = {
            'det_means': det_mean.detach().cpu(),
            'det_covs': det_cov.detach().cpu(),
            'det_obj_probs': torch.ones(len(det_mean)).float()
            # 'track_ids': torch.arange(2).long(),
            # 'slot_ids': torch.arange(2).long()
        }
        result = self.tracker(result)


        return result

    def _ar_forward_testv2(self, datas, return_unscaled=False, **kwargs):
        all_embeds = [self._forward_single(data) for data in datas]
        all_embeds = torch.stack(all_embeds, dim=0)[-1] # 1 x L x D

        det_mean, det_cov, det_obj_probs = self.convert(all_embeds)
        det_mean, det_cov, det_obj_probs[0] = det_mean[0], det_cov[0], det_obj_probs[0]
        result = {
            'det_means': det_mean.cpu(),
            'det_covs': det_cov.cpu(),
        }

        if self.tracks is None:
            self.tracks = self.global_pos_encoding.weight.unsqueeze(0) # 1 x L x D
        

        if self.global_ca_layers > 1:
            for layer in self.global_cross_attn:
                self.tracks = layer(self.tracks, all_embeds)
        else:
            self.tracks = self.global_cross_attn(self.tracks, all_embeds)

        # self.tracks = self.global_cross_attn(self.tracks, all_embeds)
        curr = self.ctn(self.tracks)

        track_mean, track_cov, _ = self.convert(curr)
        # track_mean, track_cov = track_mean.squeeze(), track_cov.squeeze()
        track_mean, track_cov = track_mean[0], track_cov[0]

        result = {
            'det_means': track_mean.detach().cpu(),
            'det_covs': track_cov.detach().cpu(),
            'det_obj_probs': torch.ones(len(track_mean)).float()
            # 'track_ids': torch.arange(2).long(),
            # 'slot_ids': torch.arange(2).long()
        }
        result = self.tracker(result)


        return result

    def _ar_forward_testv1(self, datas, return_unscaled=False, **kwargs):
        all_embeds = [self._forward_single(data) for data in datas]
        all_embeds = torch.stack(all_embeds, dim=0) 
        Nt, B, L, D = all_embeds.shape
        all_embeds = all_embeds.reshape(Nt*B, L, D)
        
        if self.num_queries is not None:
            #global_pos_embeds = all_embeds.new_zeros(1, self.num_queries, 256)
            global_pos_embeds = self.global_pos_encoding.weight.unsqueeze(0)
        else:
            global_pos_embeds = self.global_pos_encoding(None).unsqueeze(0)
        global_pos_embeds = global_pos_embeds.expand(B*Nt, -1, -1, -1)

        final_embeds = self.global_cross_attn(global_pos_embeds, all_embeds)
        final_embeds = final_embeds.reshape(B*Nt, -1, D)
        _, L, D = final_embeds.shape
        
        output_vals = self.ctn(final_embeds)
        # output_vals = self.output_head(final_embeds) #B L No
        output_vals = output_vals.reshape(Nt, B, L, -1)[-1][0]
        output_vals = output_vals.detach()
        
        det_mean, det_cov, _ = self.convert(output_vals.unsqueeze(0))
        det_mean, det_cov = det_mean.squeeze(), det_cov.squeeze()
        result = {
            'det_means': det_mean.cpu(),
            'det_covs': det_cov.cpu(),
        }


        if self.tracks is None:
            self.tracks = output_vals
        else:
            for layer in self.ar_cross_attn:
                self.tracks = layer(self.tracks, output_vals)

        track_mean, track_cov, _ = self.convert(self.tracks.unsqueeze(0))
        track_mean, track_cov = track_mean.squeeze(), track_cov.squeeze()


        result.update({
            'track_means': track_mean.detach().cpu(),
            'track_covs': track_cov.detach().cpu(),
            'track_ids': torch.arange(2).long(),
            'slot_ids': torch.arange(2).long()
        })

        return result
    
    def _ar_forwardv2(self, datas, return_unscaled=False, **kwargs):
        losses = defaultdict(list)
        mocaps = [d[('mocap', 'mocap')] for d in datas]
        mocaps = mmcv.parallel.collate(mocaps)

        gt_positions = mocaps['gt_positions']
        gt_positions = gt_positions.transpose(0,1)
        # B, T, L, f = gt_positions.shape
        # gt_positions = gt_positions.reshape(B*T, L, f)
        # gt_positions = gt_positions.reshape(T*B, N, C)

        gt_ids = mocaps['gt_ids']
        T, B, f = gt_ids.shape
        gt_ids = gt_ids.transpose(0,1)
        # gt_ids = gt_ids.reshape(T*B, f)

        gt_grids = mocaps['gt_grids']
        T, B, N, Np, f = gt_grids.shape
        gt_grids = gt_grids.transpose(0,1)
        # gt_grids = gt_grids.reshape(T*B, N, Np, f)

        all_embeds = [self._forward_single(data) for data in datas]
        all_embeds = torch.stack(all_embeds, dim=0) 
        # T, B, L, D = all_embeds.shape
        # all_embeds = all_embeds.reshape(T*B, L, D)
        all_embeds = all_embeds.transpose(0, 1) #B T L D
        
        # output_vals = self.ctn(final_embeds)
        # output_vals = self.output_head(final_embeds) #B L No
        # output_vals = output_vals.reshape(Nt, B, L, -1)
        # output_vals = output_vals.transpose(0,1) #B x Nt x L x No

        # bs = len(output_vals)
        all_outputs = []
        for i in range(B):
            global_pos_embeds = self.global_pos_encoding.weight.unsqueeze(0) # 1 x 2 x 256
            for j in range(T):
                if self.global_ca_layers > 1:
                    for layer in self.global_cross_attn:
                        global_pos_embeds = layer(global_pos_embeds, all_embeds[i,j].unsqueeze(0))
                else:
                    global_pos_embeds = self.global_cross_attn(global_pos_embeds, all_embeds[i,j].unsqueeze(0))

                curr = self.ctn(global_pos_embeds)
                mean, cov, obj_logits = self.convert(curr)
                mean, cov, obj_logits = mean[0], cov[0], obj_logits[0]
                dist = self.dist(mean, cov)
                grid = gt_grids[i][j]
                No, G, f = grid.shape
                grid = grid.reshape(No*G, 2)
                log_grid_pdf = dist.log_prob(grid.unsqueeze(1)) * 1.5
                log_grid_pdf = log_grid_pdf.reshape(No, G, len(mean))
                logsum = torch.logsumexp(log_grid_pdf, dim=1).t()
                pos_neg_log_probs = -logsum
                if self.match_by_id:
                    assign_idx = torch.stack([gt_ids[i,j]]*2, dim=-1).cpu()
                else:
                    assign_idx = linear_assignment(pos_neg_log_probs)
                
                pos_loss, count = 0, 0
                for pred_idx, gt_idx in assign_idx:
                    if pred_idx.item() == -1 or gt_idx.item() == -1:
                        continue
                    pos_loss += pos_neg_log_probs[pred_idx, gt_idx]
                    count += 1
                pos_loss /= count
                pos_loss = pos_loss * self.pos_loss_weight
                losses['pos_loss'].append(pos_loss)

        losses = {k: torch.stack(v).mean() for k, v in losses.items()}
        return losses

    def _ar_forward(self, datas, return_unscaled=False, **kwargs):
        losses = defaultdict(list)
        mocaps = [d[('mocap', 'mocap')] for d in datas]
        mocaps = mmcv.parallel.collate(mocaps)

        gt_positions = mocaps['gt_positions']
        gt_positions = gt_positions.transpose(0,1)

        gt_ids = mocaps['gt_ids']
        T, B, f = gt_ids.shape
        gt_ids = gt_ids.transpose(0,1)

        gt_grids = mocaps['gt_grids']
        T, B, N, Np, f = gt_grids.shape
        gt_grids = gt_grids.transpose(0,1)

        
       
        global_pos_embeds = self.global_pos_encoding.weight.unsqueeze(0) # 1 x 2 x 256
        global_pos_embeds = global_pos_embeds.expand(B, -1, -1) #B x 2 x 256
       
        
        all_outputs = []
        for j in range(T):
            global_pos_embeds = self._forward_single_ar(datas[j], global_pos_embeds)
            curr = self.ctn(global_pos_embeds)
            mean, cov, obj_logits = self.convert(curr)
            for i in range(B):
                dist = self.dist(mean[i], cov[i])
                grid = gt_grids[i][j]
                No, G, f = grid.shape
                grid = grid.reshape(No*G, 2)
                log_grid_pdf = dist.log_prob(grid.unsqueeze(1)) * 1.5
                log_grid_pdf = log_grid_pdf.reshape(No, G, len(mean[i]))
                logsum = torch.logsumexp(log_grid_pdf, dim=1).t()
                pos_neg_log_probs = -logsum
                if self.match_by_id:
                    assign_idx = torch.stack([gt_ids[i,j]]*2, dim=-1).cpu()
                else:
                    assign_idx = linear_assignment(pos_neg_log_probs)
                
                pos_loss, count = 0, 0
                for pred_idx, gt_idx in assign_idx:
                    if pred_idx.item() == -1 or gt_idx.item() == -1:
                        continue
                    pos_loss += pos_neg_log_probs[pred_idx, gt_idx]
                    count += 1
                pos_loss /= count
                pos_loss = pos_loss * self.pos_loss_weight
                losses['pos_loss'].append(pos_loss)

        losses = {k: torch.stack(v).mean() for k, v in losses.items()}
        return losses


    def _ar_forwardv1(self, datas, return_unscaled=False, **kwargs):
        losses = defaultdict(list)
        mocaps = [d[('mocap', 'mocap')] for d in datas]
        mocaps = mmcv.parallel.collate(mocaps)

        gt_positions = mocaps['gt_positions']
        gt_positions = gt_positions.transpose(0,1)
        # gt_positions = gt_positions.reshape(T*B, N, C)

        gt_ids = mocaps['gt_ids']
        T, B, f = gt_ids.shape
        gt_ids = gt_ids.transpose(0,1)
        # gt_ids = gt_ids.reshape(T*B, f)

        gt_grids = mocaps['gt_grids']
        T, B, N, Np, f = gt_grids.shape
        gt_grids = gt_grids.transpose(0,1)
        # gt_grids = gt_grids.reshape(T*B, N, Np, f)

        all_embeds = [self._forward_single(data) for data in datas]
        all_embeds = torch.stack(all_embeds, dim=0) 
        Nt, B, L, D = all_embeds.shape
        all_embeds = all_embeds.reshape(Nt*B, L, D)
        
        if self.num_queries is not None:
            #global_pos_embeds = all_embeds.new_zeros(1, self.num_queries, 256)
            global_pos_embeds = self.global_pos_encoding.weight.unsqueeze(0)
        else:
            global_pos_embeds = self.global_pos_encoding(None).unsqueeze(0)
        global_pos_embeds = global_pos_embeds.expand(B*Nt, -1, -1, -1)

        final_embeds = self.global_cross_attn(global_pos_embeds, all_embeds)
        final_embeds = final_embeds.reshape(B*Nt, -1, D)
        _, L, D = final_embeds.shape
        
        output_vals = self.ctn(final_embeds)
        
        # output_vals = self.output_head(final_embeds) #B L No
        output_vals = output_vals.reshape(Nt, B, L, -1)
        output_vals = output_vals.transpose(0,1) #B x Nt x L x No

        bs = len(output_vals)
        all_outputs = []
        for i in range(bs):
            T = len(output_vals[i])
            for j in range(T):
                if j == 0:
                    curr = output_vals[i][j]
                else:
                    for layer in self.ar_cross_attn:
                        curr = layer(curr, output_vals[i][j])
                mean, cov, obj_logits = self.convert(curr.unsqueeze(0))
                mean, cov, obj_logits = mean.squeeze(), cov.squeeze(), obj_logits.squeeze()
                dist = self.dist(mean, cov)
                grid = gt_grids[i][j]
                No, G, f = grid.shape
                grid = grid.reshape(No*G, 2)
                log_grid_pdf = dist.log_prob(grid.unsqueeze(1)) * 1.5
                log_grid_pdf = log_grid_pdf.reshape(No, G, len(mean))
                logsum = torch.logsumexp(log_grid_pdf, dim=1).t()
                pos_neg_log_probs = -logsum
                if self.match_by_id:
                    assign_idx = torch.stack([gt_ids[i]]*2, dim=-1).cpu()
                else:
                    assign_idx = linear_assignment(pos_neg_log_probs)
                
                pos_loss = 0
                for pred_idx, gt_idx in assign_idx:
                    pos_loss += pos_neg_log_probs[pred_idx, gt_idx]
                pos_loss /= len(assign_idx)
                pos_loss = pos_loss * self.pos_loss_weight
                losses['pos_loss'].append(pos_loss)

        losses = {k: torch.stack(v).mean() for k, v in losses.items()}
        return losses
    
    def convert(self, output_vals):
        output_vals = self.output_head(output_vals)
        if self.include_z:
            mean = output_vals[..., 0:3]
        else:
            mean = output_vals[..., 0:2]
        if self.add_grid_to_mean:
            mean[..., 0] += self.global_pos_encoding.unscaled_params_x.flatten()
            mean[..., 1] += self.global_pos_encoding.unscaled_params_y.flatten()
        mean = mean.sigmoid()
        mean = mean * self.mean_scale

        if self.predict_full_cov and self.include_z:
            cov = F.softplus(output_vals[..., 3:3+9])
            cov = cov.view(B*Nt, L, 3,3).tril()
        elif not self.predict_full_cov and self.include_z:
            cov = F.softplus(output_vals[..., 3:6])
        elif not self.predict_full_cov and not self.include_z:
            cov = F.softplus(output_vals[..., 2:4])
            cov = torch.diag_embed(cov)
        elif self.predict_full_cov and not self.include_z:
            cov = F.softplus(output_vals[..., 2:4])
            cov = torch.diag_embed(cov)
            cov[..., -1, 0] += output_vals[..., 4]
            B, N, _, _ = cov.shape
            cov = cov.reshape(B*N, 2, 2)
            cov = torch.bmm(cov, cov.transpose(-2,-1))
            cov = cov.reshape(B, N, 2, 2)
        obj_logits = output_vals[..., -1]
        return mean, cov, obj_logits


    def _forward(self, datas, return_unscaled=False, **kwargs):
        all_embeds, all_mocap = [], []
        for t, data in enumerate(datas):
            embeds = self._forward_single(data)
            all_mocap.append(data[('mocap', 'mocap')])
            all_embeds.append(embeds)
        
        all_embeds = torch.stack(all_embeds, dim=0) 
        Nt, B, L, D = all_embeds.shape
        all_embeds = all_embeds.reshape(Nt*B, L, D)
        
        if self.num_queries is not None:
            #global_pos_embeds = all_embeds.new_zeros(1, self.num_queries, 256)
            global_pos_embeds = self.global_pos_encoding.weight.unsqueeze(0)
        else:
            global_pos_embeds = self.global_pos_encoding(None).unsqueeze(0)
        global_pos_embeds = global_pos_embeds.expand(B*Nt, -1, -1, -1)

        final_embeds = self.global_cross_attn(global_pos_embeds, all_embeds)
        final_embeds = final_embeds.reshape(B*Nt, -1, D)
        _, L, D = final_embeds.shape
        
        final_embeds = self.ctn(final_embeds)
        
        output_vals = self.output_head(final_embeds) #B L 7
        output_vals = output_vals.reshape(Nt, B, L, self.num_outputs)
        
        if self.time_attn is not None:
            output_vals = output_vals.permute(1, 2, 0, 3) #B L Nt 7
            output_vals = output_vals.reshape(B*L, Nt, self.num_outputs)
            output_vals = self.time_attn(output_vals)
            output_vals = output_vals.reshape(B, L, Nt, self.num_outputs)
            output_vals = output_vals.permute(2, 0, 1, 3) #Nt B L 7

        output_vals = output_vals.reshape(Nt*B, L, self.num_outputs)
        output_vals = self.output_sa(output_vals)
        
        if self.include_z:
            mean = output_vals[..., 0:3]
        else:
            mean = output_vals[..., 0:2]
        if self.add_grid_to_mean:
            mean[..., 0] += self.global_pos_encoding.unscaled_params_x.flatten()
            mean[..., 1] += self.global_pos_encoding.unscaled_params_y.flatten()
        mean = mean.sigmoid()
        mean = mean * self.mean_scale

        if self.predict_full_cov and self.include_z:
            cov = F.softplus(output_vals[..., 3:3+9])
            cov = cov.view(B*Nt, L, 3,3).tril()
        elif not self.predict_full_cov and self.include_z:
            cov = F.softplus(output_vals[..., 3:6])
        elif not self.predict_full_cov and not self.include_z:
            cov = F.softplus(output_vals[..., 2:4])
            cov = torch.diag_embed(cov)
        elif self.predict_full_cov and not self.include_z:
            cov = F.softplus(output_vals[..., 2:4])
            cov = torch.diag_embed(cov)
            cov[..., -1, 0] += output_vals[..., 4]
            B, N, _, _ = cov.shape
            cov = cov.reshape(B*N, 2, 2)
            cov = torch.bmm(cov, cov.transpose(-2,-1))
            cov = cov.reshape(B, N, 2, 2)

        obj_logits = output_vals[..., -1]

        corner_vals = None
        if self.predict_corners:
            corner_vals = output_vals[..., 7:]
            B, L, _ = corner_vals.shape
            corner_vals = corner_vals.reshape(B, L, 4, 2)
            corner_vals = corner_vals.sigmoid()
            corner_vals = corner_vals * self.mean_scale[0:2]
        return mean, cov, None, obj_logits, corner_vals
    
    def _forward_single_ar(self, data, Q):
        # inter_embeds = []
        for key in data.keys():
            mod, node = key
            if mod == 'mocap':
                continue
            if mod not in self.backbones.keys():
                continue
            backbone = self.backbones[mod]
            model = self.models[mod + '_' + node]
            try:
                feats = backbone(data[key]['img'])
            except:
                feats = backbone([data[key]['img']])
            Q = model(feats, pos_embeds=Q)
            # inter_embeds.append(embeds)

        # if len(inter_embeds) == 0:
            # import ipdb; ipdb.set_trace() # noqa
        
        # inter_embeds = torch.cat(inter_embeds, dim=-2)
        return Q

    def _forward_single(self, data, return_unscaled=False, **kwargs):
        inter_embeds = []
        for key in data.keys():
            mod, node = key
            if mod == 'mocap':
                continue
            if mod not in self.backbones.keys():
                continue
            backbone = self.backbones[mod]
            model = self.models[mod + '_' + node]
            try:
                feats = backbone(data[key]['img'])
            except:
                feats = backbone([data[key]['img']])
            embeds = model(feats)
            inter_embeds.append(embeds)

        if len(inter_embeds) == 0:
            import ipdb; ipdb.set_trace() # noqa
        
        inter_embeds = torch.cat(inter_embeds, dim=-2)
        return inter_embeds
        
    def forward_track(self, data, **kwargs):
        if self.autoregressive:
            return self._ar_forward_test(data, **kwargs)
        mean, cov, cls_logits, obj_logits, _ = self._forward(data, **kwargs)
        mean = mean[-1] #get last time step, there should be no future
        cov = cov[-1]
        obj_logits = obj_logits[-1]
        result = {
            'det_means': mean.cpu().detach().cpu(),
            #'det_covs': torch.diag_embed(cov).detach().cpu(),
            'det_covs': cov.detach().cpu(),
            'det_obj_probs': obj_logits.sigmoid().detach().cpu(),
            # 'track_ids': torch.arange(len(mean))
        }
        result = self.tracker(result)
        return result
    
    def forward_train(self, data, **kwargs):
        losses = defaultdict(list)
        if self.autoregressive:
            return self._ar_forward(data, **kwargs)
        else:
            mean, cov, cls_logits, obj_logits, corners = self._forward(data, **kwargs)
        mocaps = [d[('mocap', 'mocap')] for d in data]
        mocaps = mmcv.parallel.collate(mocaps)

        gt_positions = mocaps['gt_positions']
        T, B, N, C = gt_positions.shape
        gt_positions = gt_positions.reshape(T*B, N, C)

        gt_ids = mocaps['gt_ids']
        T, B, f = gt_ids.shape
        gt_ids = gt_ids.reshape(T*B, f)

        if self.predict_corners:
            gt_corners = mocaps['gt_corners']
            T, B, N, Nc,  _ = gt_corners.shape
            gt_corners = gt_corners.reshape(T*B, N, Nc, -1)

        if self.grid_loss:
            gt_grids = mocaps['gt_grids']
            T, B, N, Np, f = gt_grids.shape
            gt_grids = gt_grids.reshape(T*B, N, Np, f)

        bs = len(mean)
        assert len(gt_positions) == bs
        for i in range(bs):
            obj_probs = F.sigmoid(obj_logits[i])
            dist = self.dist(mean[i], cov[i])
            gt_pos = gt_positions[i]
            mask = gt_pos[:, 0] != -1
            gt_pos = gt_pos[mask]
            
            obj_targets = mean.new_zeros(len(mean[i])) + (1.0 - self.bce_target) #Nq
            obj_targets = obj_targets.float()
            if len(gt_pos) > 0:
                if self.grid_loss:
                    grid = gt_grids[i][mask]
                    No, G, f = grid.shape
                    grid = grid.reshape(No*G, 2)
                    log_grid_pdf = dist.log_prob(grid.unsqueeze(1)) * 1.5
                    log_grid_pdf = log_grid_pdf.reshape(No, G, len(mean[i]))
                    logsum = torch.logsumexp(log_grid_pdf, dim=1).t()
                    pos_neg_log_probs = -logsum
                else:
                    pos_log_probs = [dist.log_prob(pos) for pos in gt_pos]
                    pos_neg_log_probs = -torch.stack(pos_log_probs, dim=-1) #Nq x num_objs


                if self.match_by_id:
                    assign_idx = torch.stack([gt_ids[i]]*2, dim=-1).cpu()
                else:
                    assign_idx = linear_assignment(pos_neg_log_probs)
                
                mse_loss, pos_loss = 0, 0
                for pred_idx, gt_idx in assign_idx:
                    pos_loss += pos_neg_log_probs[pred_idx, gt_idx]
                    mse_loss += self.mse_loss(mean[i][pred_idx], gt_pos[gt_idx]).mean()
                pos_loss /= len(assign_idx)
                pos_loss = pos_loss * self.pos_loss_weight
                losses['pos_loss'].append(pos_loss)
                
                mse_loss /= len(assign_idx)
                if self.mse_loss_weight != 0:
                    mse_loss *= self.mse_loss_weight
                    losses['mse_loss'].append(mse_loss)

                obj_targets[assign_idx[:, 0]] = self.bce_target
            
            # if pos_neg_log_probs.shape[0] > pos_neg_log_probs.shape[1]:
            # obj_loss_vals = self.bce_loss(obj_probs.squeeze(), obj_targets)
            obj_loss_vals = self.bce_loss(obj_probs, obj_targets)
            pos_obj_loss = obj_loss_vals[obj_targets == self.bce_target].mean()
            neg_obj_loss = obj_loss_vals[obj_targets == 1.0 - self.bce_target].mean() 
            if sum(obj_targets == self.bce_target).item() != 0:
                losses['pos_obj_loss'].append(pos_obj_loss)

            if sum(obj_targets == 1.0 - self.bce_target).item() != 0:
                losses['neg_obj_loss'].append(neg_obj_loss)

            is_obj = obj_probs >= 0.5
            percent = is_obj.float().sum()
            acc = torch.abs(len(gt_pos) - percent)
            losses['acc'].append(acc.detach())
            
            if self.predict_corners:
                corner_loss = 0
                for pred_idx, gt_idx in assign_idx:
                    corner_loss += self.mse_loss(gt_corners[i][gt_idx], corners[i][pred_idx]).mean()
                losses['corner_loss'].append(corner_loss / len(assign_idx))

        losses = {k: torch.stack(v).mean() for k, v in losses.items()}
        return losses
        

    def simple_test(self, img, img_metas, rescale=False):
        pass

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
            ``num_samples``.

            - ``loss`` is a tensor for back propagation, which can be a
            weighted sum of multiple losses.
            - ``log_vars`` contains all the variables to be sent to the
            logger.
            - ``num_samples`` indicates the batch size (when the model is
            DDP, it means the batch size on each GPU), which is used for
            averaging the logs.
        """
        losses = self(data)
        loss, log_vars = self._parse_losses(losses)
        
        num_samples = len(data[0][('mocap', 'mocap')]['gt_positions'])

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
            which may be a weighted sum of all losses, log_vars contains
            all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars



    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
