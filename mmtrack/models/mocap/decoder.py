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

def linear_assignment(cost_matrix):
    cost_matrix = cost_matrix.cpu().detach().numpy()
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    assign_idx = np.array([[y[i], i] for i in x if i >= 0])
    sort_idx = np.argsort(assign_idx[:, 1])
    assign_idx = assign_idx[sort_idx]
    assign_idx = torch.from_numpy(assign_idx)
    return assign_idx.long()

@MODELS.register_module()
class DecoderMocapModel(BaseMocapModel):
    def __init__(self,
                 img_model_cfg=None,
                 img_backbone_cfg=None,
                 img_neck_cfg=None,
                 depth_backbone_cfg=None,
                 depth_neck_cfg=None,
                 bce_target=0.99,
                 remove_zero_at_train=True,
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
                 num_sa_layers=6,
                 self_attn_cfg=dict(type='QKVAttention',
                     qk_dim=7,
                     num_heads=1, 
                     in_proj=True,
                     out_proj=True,
                     attn_drop=0.0, 
                     seq_drop=0.0,
                     return_weights=False,
                     v_dim=None
                 ),
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

        self.remove_zero_at_train = remove_zero_at_train
        self.bce_target = bce_target
        
        self.img_model = build_model(img_model_cfg)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.global_pos_encoding = AnchorEncoding(dim=256, learned=False, out_proj=False)
        self.global_cross_attn = ResCrossAttn(cross_attn_cfg)
        
        self.self_attn = [ResSelfAttn(self_attn_cfg) for _ in range(num_sa_layers)]
        self.self_attn = nn.Sequential(*self.self_attn)
        
        self.ctn = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )
        

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


    def _forward(self, data, return_unscaled=False, **kwargs):
        inter_embeds = []

        if 'zed_camera_left' in data.keys():
            img = data['zed_camera_left']['img']
            img_embeds = self.img_model(img)
            inter_embeds.append(img_embeds)

        if 'zed_camera_depth' in data.keys():
            dmap = data['zed_camera_depth']['img']
            depth_embeds = self.depth_model(dmap)
            inter_embeds.append(depth_embeds)

                    
        if len(inter_embeds) == 0:
            import ipdb; ipdb.set_trace() # noqa

        inter_embeds = torch.cat(inter_embeds, dim=1)
        B, L, D = inter_embeds.shape
        
        global_pos_embeds = self.global_pos_encoding(None).unsqueeze(0)
        global_pos_embeds = global_pos_embeds.expand(B, -1, -1, -1)

        final_embeds = self.global_cross_attn(global_pos_embeds, inter_embeds)
        final_embeds = final_embeds.reshape(B, -1, D)
        
        final_embeds = self.ctn(final_embeds)

        output_vals = self.output_head(final_embeds) #B L 7
        
        # for layer in self.self_attn:
            # output_vals[:, :, 0] += self.global_pos_encoding.unscaled_params_x.flatten()
            # output_vals[:, :, 1] += self.global_pos_encoding.unscaled_params_y.flatten()
            # output_vals = layer(output_vals)
            
        output_vals = self.self_attn(output_vals)

        if return_unscaled:
            return output_vals

        # mean = self.mean_head(final_embeds)
        mean = output_vals[..., 0:3]
        mean[:, :, 0] += self.global_pos_encoding.unscaled_params_x.flatten()
        mean[:, :, 1] += self.global_pos_encoding.unscaled_params_y.flatten()
        mean = mean.sigmoid()

        # cov = self.cov_head(final_embeds)
        cov = F.softplus(output_vals[..., 3:6])
        # cls_logits = self.cls_head(final_embeds)
        # obj_logits = self.obj_head(final_embeds)
        obj_logits = output_vals[..., -1]
        # obj_logits = self.self_attn(obj_logits)
        return mean, cov, None, obj_logits
 

    def forward_test(self, data, **kwargs):
        mean, cov, cls_logits, obj_logits = self._forward(data, **kwargs)
        assert len(mean) == 1 #assume batch size of 1
        mean = mean[0] #Nq x 3 
        cov = cov[0] #Nq x 3
        obj_probs = F.sigmoid(obj_logits[0]).squeeze()
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
        assert len(means) == 1 #assume batch size of 1
        means = means[0] #Nq x 3 
        covs = covs[0] #Nq x 3
        obj_probs = F.sigmoid(obj_logits[0]).squeeze()
        is_obj = obj_probs >= 0.5
        means = means[is_obj]
        covs = covs[is_obj]

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
                pred_dist = self.dist(means[j], covs[j])
                pred_dist = D.Independent(pred_dist, 1)
                log_prob = pred_dist.log_prob(track.mean[...,0:3].cuda())
                log_probs[i, j] = log_prob
        
        if len(log_probs) == 0: #no tracks yet
            for j in range(len(means)):
                new_track = MocapTrack(means[j], covs[j])
                self.tracks.append(new_track)
        else:
            print(log_probs.exp())
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
        
        track_means, track_covs = [torch.empty(0,3)], [torch.empty(0,3)]
        track_ids = []
        for t, track in enumerate(self.tracks):
            onstreak = track.hit_streak >= self.min_hits
            warmingup = self.frame_count <= self.min_hits
            if track.wasupdated and (onstreak or warmingup):
                track_means.append(track.mean[...,0:3].unsqueeze(0))
                track_covs.append(track.cov[...,0:3].diag().unsqueeze(0))
                track_ids.append(track.id)
        
        track_means = torch.cat(track_means)
        track_covs = torch.cat(track_covs)
        track_ids = torch.tensor(track_ids)

        print(track_means, track_ids)
        # states = torch.cat(states, dim=0)
        # ids = torch.tensor(ids).cuda()
        # labels = torch.tensor(labels).cuda()
        # scores = torch.tensor(scores).cuda()
        # ret = (states, labels, ids, scores)
        keep_tracks = []
        for track in self.tracks:
            if track.time_since_update > self.max_age:
                continue
            keep_tracks.append(track)
        self.tracks = keep_tracks
        # self.tracks = [track for track in self.tracks\
                       # if track.time_since_update < self.max_age]

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
        bs = len(mean)
        for i in range(bs):
            is_missing = data['missing']['zed_camera_left'][i]
            if is_missing:
                continue
            
            # is_missing = data['missing']['zed_camera_depth'][i]
            # if is_missing:
                # continue

            dist = self.dist(mean[i], cov[i])
            dist = D.Independent(dist, 1) #Nq independent Gaussians

            gt_pos = data['mocap']['gt_positions'][i]#[-2].unsqueeze(0)
            gt_labels = data['mocap']['gt_labels'][i]#[-2].unsqueeze(0)

            is_node = gt_labels == 0
            final_mask = ~is_node
            if self.remove_zero_at_train:
                z_is_zero = gt_pos[:, -1] == 0.0
                final_mask = final_mask & ~z_is_zero
            gt_pos = gt_pos[final_mask]
            gt_labels = gt_labels[final_mask]

            if len(gt_pos) == 0:
                continue

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
        
        num_samples = len(data['mocap']['gt_positions'])

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

