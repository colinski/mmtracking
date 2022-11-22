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
# from mmtrack.models.mot import BaseMultiObjectTracker
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
# from .hungarian import match
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

@MODELS.register_module()
class DecoderMocapModelV2(BaseMocapModel):
    def __init__(self,
                 backbone_cfgs=None,
                 model_cfgs=None,
                 decoder_cfg=dict(type='DETRDecoder',
                    num_layers=6,
                    self_attn_cfg=dict(type='ResSelfAttn', attn_cfg=dict(type='QKVAttention', qk_dim=256, num_heads=8)),
                    cross_attn_cfg=dict(type='ResCrossAttn', attn_cfg=dict(type='QKVAttention', qk_dim=256, num_heads=8)),
                    ffn_cfg=dict(type='SLP', in_channels=256),
                    return_all_layers=False,
                 ),
                 output_encoder_cfg=dict(type='DETREncoder',
                    num_layers=6,
                    self_attn_cfg=dict(type='ResSelfAttn', 
                        attn_cfg=dict(type='QKVAttention', qk_dim=7, num_heads=1, attn_drop=0.0,seq_drop=0.0)
                    ),
                    ffn_cfg=None,
                    out_norm_cfg=None
                 ),
                 bce_target=0.99,
                 output_sa_cfg=dict(type='QKVAttention',
                     qk_dim=7,
                     num_heads=1, 
                     in_proj=True,
                     out_proj=True,
                     attn_drop=0.0, 
                     seq_drop=0.0,
                     return_weights=False,
                     v_dim=None
                 ),
                 time_attn_cfg=dict(type='QKVAttention',
                     qk_dim=7,
                     num_heads=1, 
                     in_proj=True,
                     out_proj=True,
                     attn_drop=0.0, 
                     seq_drop=0.0,
                     return_weights=False,
                     v_dim=None
                 ),
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
                 num_output_sa_layers=6,
                 max_age=5,
                 min_hits=3,
                 track_eval=False,
                 mse_loss_weight=0.0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = 2
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.frame_count = 0 
        self.track_eval = track_eval
        self.mse_loss_weight = mse_loss_weight
        self.bce_target = bce_target
        
        self.decoder = build_from_cfg(decoder_cfg, FEEDFORWARD_NETWORK)
        # self.output_encoder = build_from_cfg(output_encoder_cfg, FEEDFORWARD_NETWORK)

        # self.output_sa = [ResSelfAttn(output_sa_cfg) for _ in range(num_output_sa_layers)]
        # self.output_sa = nn.Sequential(*self.output_sa)

        
        self.time_attn = None
        if time_attn_cfg is not None:
            self.time_attn = [ResSelfAttn(time_attn_cfg) for _ in range(6)]
            self.time_attn = nn.Sequential(*self.time_attn)

        # self.history = None
        # if history_size is not None:
            # self.time_attn = [ResCrossAttn(output_sa_cfg) for _ in range(num_output_sa_layers)]
            # self.time_attn = nn.Sequential(*self.time_attn)

            # self.history_size = history_size
            # self.history = []

        self.backbones = nn.ModuleDict()
        for key, cfg in backbone_cfgs.items():
            self.backbones[key] = build_backbone(cfg)
        
        self.models = nn.ModuleDict()

        for key, cfg in model_cfgs.items():
            mod, node = key
            self.models[mod + '_' + node] = build_model(cfg)
        
        # if img_model_cfg is not None:
            # self.img_model = build_model(img_model_cfg)
        
        # if range_model_cfg is not None:
            # self.range_model = build_model(range_model_cfg)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.global_anchor_encoder = AnchorEncoding(dim=256, learned=False, out_proj=False)
        
        self.global_cross_attn = ResCrossAttn(cross_attn_cfg)
        
        self.ctn = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )
        
        self.output_sa = [ResSelfAttn(output_sa_cfg) for _ in range(num_output_sa_layers)]
        self.output_sa = nn.Sequential(*self.output_sa)

        
        self.output_head = nn.Linear(256, 3+3+1)

        self.dist = D.Normal

        # focal_loss_cfg = dict(type='FocalLoss',
            # use_sigmoid=True, 
            # gamma=2.0, alpha=0.25, reduction='none',
            # loss_weight=1.0, activated=True)
        # self.focal_loss = build_loss(focal_loss_cfg)
        self.bce_loss = nn.BCELoss(reduction='none')

    
    #def forward(self, data, return_loss=True, **kwargs):
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

    def _forward(self, datas, return_unscaled=False, **kwargs):
        mod_embeds = []
        for t, data in enumerate(datas):
            embeds = self._forward_single(data)
            mod_embeds.append(embeds)
        
        mod_embeds = torch.stack(mod_embeds, dim=0) 
        Nt, B, L, D = mod_embeds.shape
        mod_embeds = mod_embeds.reshape(Nt*B, L, D)

        global_anchor_pos = self.global_anchor_encoder(None).unsqueeze(0)
        global_anchor_pos = global_anchor_pos.expand(B*Nt, -1, -1, -1)

        global_anchor_embeds = torch.zeros_like(global_anchor_pos)

        final_embeds = self.decoder(global_anchor_embeds, global_anchor_pos, mod_embeds, None)



        #final_embeds = self.global_cross_attn(global_pos_embeds, all_embeds)
        # final_embeds = self.global_cross_attn(global_anchor_pos, mod_embeds)
        final_embeds = final_embeds.reshape(B*Nt, -1, D)
        _, L, D = final_embeds.shape
        
        final_embeds = self.ctn(final_embeds)
        
        output_vals = self.output_head(final_embeds) #B L 7
        output_vals = output_vals.reshape(Nt, B, L, 7)
        
        if self.time_attn is not None:
            output_vals = output_vals.permute(1, 2, 0, 3) #B L Nt 7
            output_vals = output_vals.reshape(B*L, Nt, 7)
            output_vals = self.time_attn(output_vals)
            output_vals = output_vals.reshape(B, L, Nt, 7)
            output_vals = output_vals.permute(2, 0, 1, 3) #Nt B L 7

        output_vals = output_vals.reshape(Nt*B, L, 7)
        output_vals = self.output_sa(output_vals)
        
        # output_vals = self.output_encoder(output_vals, None)
        # output_vals = output_vals.reshape(Nt, B, L, 7)
        
        mean = output_vals[..., 0:3]
        mean[..., 0] += self.global_anchor_encoder.unscaled_params_x.flatten()
        mean[..., 1] += self.global_anchor_encoder.unscaled_params_y.flatten()
        mean = mean.sigmoid()
        mean = mean * torch.tensor([7,5,1]).to(mean.device)

        cov = F.softplus(output_vals[..., 3:6])
        obj_logits = output_vals[..., -1]
        return mean, cov, None, obj_logits

        # output_vals = output_vals[len(data['past_frames'])]
        # output_vals = self.output_sa(output_vals)

    def _forward_single(self, data, return_unscaled=False, **kwargs):
        inter_embeds = []
        for key in data.keys():
            mod, node = key
            if mod == 'mocap':
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
        
    def forward_test(self, data, **kwargs):
        mean, cov, cls_logits, obj_logits = self._forward(data, **kwargs)
        mean = mean[-1] #get last time step, there should be no future
        cov = cov[-1]
        obj_logits = obj_logits[-1]
        # assert len(mean) == 1 #assume batch size of 1
        # mean = mean[0] #Nq x 3 
        # cov = cov[0] #Nq x 3
        #obj_probs = F.sigmoid(obj_logits[0]).squeeze()
        obj_probs = F.sigmoid(obj_logits).squeeze()
        is_obj = obj_probs >= 0.5
        mean = mean[is_obj]
        cov = cov[is_obj]
        
        result = {
            'pred_position_mean': mean.cpu().detach().unsqueeze(0).numpy(),
            'pred_position_cov': cov.cpu().detach().unsqueeze(0).numpy(),
            'pred_obj_prob': obj_probs[is_obj].cpu().detach().unsqueeze(0).numpy(),
            'track_ids': np.zeros((1, len(mean)))
        }
        return result

    def forward_track(self, data, **kwargs):
        output_vals = self._forward(data, return_unscaled=True)
        means, covs, cls_logits, obj_logits = self._forward(data, **kwargs)
        means = means[-1] #get last time step, there should be no future
        covs = covs[-1]
        obj_logits = obj_logits[-1]
        # assert len(means) == 1 #assume batch size of 1
        # means = means[0] #Nq x 3 
        # covs = covs[0] #Nq x 3
        # obj_probs = F.sigmoid(obj_logits[0]).squeeze() #Nq,
        obj_probs = F.sigmoid(obj_logits).squeeze() #Nq,
        is_obj = obj_probs >= 0.5
        means = means[is_obj]
        covs = covs[is_obj]

        print(covs)

        self.frame_count += 1
        
        #get predictions from existing tracks
        for track in self.tracks:
            track.predict()
        
        # if len(self.tracks) != 0:
        #collect all the new bbox predictions
        # pred_means = torch.zeros(0, 3)
        # pred_cov = torch.zeros(0, 3)
        
        log_probs = torch.zeros(len(self.tracks), len(means))
        for i, track in enumerate(self.tracks):
            # track_dist = self.dist(track.mean, track.cov)
            # track_dist = D.Independent(pred_dist, 1)
            for j, mean in enumerate(means):
                m = PositionMeasurement(means[j].cpu(), torch.diag(covs[j]).cpu(), time=track.kf.time)
                log_prob = track.kf.log_likelihood_of_update(m)
                # pred_dist = self.dist(means[j], covs[j])
                # pred_dist = D.Independent(pred_dist, 1)
                # log_prob = pred_dist.log_prob(track.mean[...,0:3].cuda())
                log_probs[i, j] = log_prob
        
        if len(log_probs) == 0: #no tracks yet
            for j in range(len(means)):
                new_track = MocapTrack(means[j], covs[j])
                self.tracks.append(new_track)
        else:
            exp_probs = log_probs.exp()
            assign_idx = linear_assignment(-log_probs)
            unassigned = []
            for t, d in assign_idx:
                if exp_probs[t,d] >= 1e-16:
                    self.tracks[t].update(means[d], covs[d])
                else:
                    unassigned.append(d)
            for d in unassigned:
                new_track = MocapTrack(means[d], covs[d])
                self.tracks.append(new_track)



        # if len(self.tracks) > 0:
            # preds = [track.state for track in self.tracks]
            # preds = torch.stack(preds, dim=0)
        


        # bbox = dets[:, 0:4]
        # matches, unmatched_dets = match(bbox, preds, self.iou_thres)

        # for d, t in matches:
            # self.tracks[t].update(dets[d])

        # for d in unmatched_detd:
            # new_track = MocapTrack(dets[d])
            # self.tracks.append(new_track)
            
        # states, ids = [torch.empty(0,4).cuda()], []
        # labels, scores = [], []
        
        track_means, track_covs, track_ids = [means.new_empty(0,3).cpu()], [means.new_empty(0,3).cpu()], []
        for t, track in enumerate(self.tracks):
            onstreak = track.hit_streak >= self.min_hits
            warmingup = self.frame_count <= self.min_hits
            if track.wasupdated and (onstreak or warmingup):
                track_means.append(track.mean.unsqueeze(0))
                track_covs.append(track.cov.diag().unsqueeze(0))
                track_ids.append(track.id)
        
        track_means = torch.cat(track_means)
        track_covs = torch.cat(track_covs)
        track_ids = torch.tensor(track_ids)

        # print(track_means, track_ids)
        # states = torch.cat(states, dim=0)
        # ids = torch.tensor(ids).cuda()
        # labels = torch.tensor(labels).cuda()
        # scores = torch.tensor(scores).cuda()
        # ret = (states, labels, ids, scores)
        # keep_tracks = []
        # for track in self.tracks:
            # if track.time_since_update > self.max_age:
                # continue
            # keep_tracks.append(track)
        # self.tracks = keep_tracks
        self.tracks = [track for track in self.tracks\
                       if track.time_since_update < self.max_age]

        result = {
            'pred_position_mean': track_means.detach().unsqueeze(0).cpu().numpy(),
            'pred_position_cov': track_covs.detach().unsqueeze(0).cpu().numpy(),
            'pred_obj_prob': obj_probs[is_obj].cpu().detach().unsqueeze(0).numpy(),
            'track_ids': track_ids.unsqueeze(0).numpy()
        }
        return result

    def forward_train(self, data, **kwargs):
        losses = defaultdict(list)
        mean, cov, cls_logits, obj_logits = self._forward(data, **kwargs)
        mocaps = [d[('mocap', 'mocap')] for d in data]
        mocaps = mmcv.parallel.collate(mocaps)

        gt_positions = mocaps['gt_positions']
        T, B, N, C = gt_positions.shape
        gt_positions = gt_positions.reshape(T*B, N, C)

        gt_labels = mocaps['gt_labels']
        T, B, N = gt_labels.shape
        gt_labels = gt_labels.reshape(T*B, N)


        bs = len(mean)
        assert len(gt_positions) == bs
        for i in range(bs):
            # is_missing = data['missing']['zed_camera_left'][i]
            # if is_missing:
                # continue
            
            # is_missing = data['missing']['zed_camera_depth'][i]
            # if is_missing:
                # continue

            dist = self.dist(mean[i], cov[i])
            dist = D.Independent(dist, 1) #Nq independent Gaussians

            # gt_pos = data['mocap']['gt_positions'][i]#[-2].unsqueeze(0)
            # gt_labels = data['mocap']['gt_labels'][i]#[-2].unsqueeze(0)

            gt_pos = gt_positions[i]
            # gt_labels = gt_labels[i]

            # is_node = gt_labels[i] == 0
            # final_mask = ~is_node
            # if self.remove_zero_at_train:
                # z_is_zero = gt_pos[:, -1] == 0.0
                # final_mask = final_mask & ~z_is_zero
            # gt_pos = gt_pos[final_mask]
            # gt_labels = gt_labels[final_mask]

            # if len(gt_pos) == 0:
                # import ipdb; ipdb.set_trace() # noqa
                # continue

            pos_log_probs = [dist.log_prob(pos) for pos in gt_pos]
            pos_neg_log_probs = -torch.stack(pos_log_probs, dim=-1) #Nq x num_objs
            assign_idx = linear_assignment(pos_neg_log_probs)
            
            mse_loss, pos_loss = 0, 0
            for pred_idx, gt_idx in assign_idx:
                pos_loss += pos_neg_log_probs[pred_idx, gt_idx]
                mse_loss += self.mse_loss(mean[i][pred_idx], gt_pos[gt_idx]).mean()
            pos_loss /= len(assign_idx)
            pos_loss /= 10
            losses['pos_loss'].append(pos_loss)

            mse_loss /= len(assign_idx)
            if self.mse_loss_weight != 0:
                mse_loss *= self.mse_loss_weight
                losses['mse_loss'].append(mse_loss)

            obj_targets = pos_neg_log_probs.new_zeros(len(pos_neg_log_probs)) + (1.0 - self.bce_target)
            obj_targets = obj_targets.float()
            obj_targets[assign_idx[:, 0]] = self.bce_target
            
            obj_probs = F.sigmoid(obj_logits[i])
            # obj_probs = torch.softmax(obj_logits[i], dim=0)
            
            # obj_loss_vals = self.focal_loss(obj_probs, obj_targets.long())
            # losses['pos_obj_loss'] = obj_loss_vals[obj_targets == 1].mean()
            # losses['neg_obj_loss'] = obj_loss_vals[obj_targets == 0].mean()

            obj_loss_vals = self.bce_loss(obj_probs.squeeze(), obj_targets)
            pos_obj_loss = obj_loss_vals[obj_targets == self.bce_target].mean()
            neg_obj_loss = obj_loss_vals[obj_targets == 1.0 - self.bce_target].mean() 
            losses['pos_obj_loss'].append(pos_obj_loss)
            losses['neg_obj_loss'].append(neg_obj_loss)

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

