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

from mmcv.cnn.bricks.registry import FEEDFORWARD_NETWORK
from mmcv import build_from_cfg

from pyro.contrib.tracking.measurements import PositionMeasurement
from mmtrack.models.mocap.tracker import Tracker


def linear_assignment(cost_matrix):
    cost_matrix = cost_matrix.cpu().detach().numpy()
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    assign_idx = np.array([[y[i], i] for i in x if i >= 0])
    sort_idx = np.argsort(assign_idx[:, 1])
    assign_idx = assign_idx[sort_idx]
    assign_idx = torch.from_numpy(assign_idx)
    return assign_idx.long()

@MODELS.register_module()
class OracleModel(BaseMocapModel):
    def __init__(self,
                 cov=[0.001,0.001],
                 mean_cov=None,
                 max_age=5,
                 min_hits=3,
                 track_eval=False,
                 no_update=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.cov = torch.tensor(cov).unsqueeze(0).float()
        self.mean_cov = mean_cov
        if self.mean_cov is not None:
            self.mean_cov = torch.tensor(mean_cov).unsqueeze(0)
        self.dummy_loss = nn.Parameter(torch.zeros(1))
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.frame_count = 0 
        self.track_eval = track_eval
        self.no_update = no_update
        self.tracker = Tracker()

    
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


    def forward_train(self, data, **kwargs):
        return {'dummy_loss': self.dummy_loss.mean()}

    def _forward(self, data, **kwargs):
        # try:
            # gt_pos = data[0][('mocap', 'mocap')]['gt_positions'][0][-2].unsqueeze(0)
        # except:
        gt_pos = data[0][('mocap', 'mocap')]['gt_positions'][0][0].unsqueeze(0)
        if self.mean_cov is not None:
            dist = D.Normal(gt_pos, self.mean_cov.cuda())
            mean = dist.sample([1])[0]
        else:
            mean = gt_pos
        result = {
            'det_means': mean.cpu(),
           'det_covs': torch.diag_embed(self.cov).cpu(),
            # 'det_covs': self.cov.cpu(),
            'det_obj_probs': torch.ones(len(mean)).float()
        }
        return result


    def forward_track(self, data, **kwargs):
        # means, covs = self._forward(data)
        det = self._forward(data)
        return self.tracker(det)
        # gt_labels = data['mocap']['gt_labels'][0][-2].unsqueeze(0)
        # is_node = gt_labels == 0
        # final_mask = ~is_node
        # z_is_zero = gt_pos[:, -1] == 0.0
        # final_mask = final_mask & ~z_is_zero
        # gt_pos = gt_pos[final_mask]
        # gt_labels = gt_labels[final_mask]

        # means = gt_pos
        # covs = self.cov
        self.frame_count += 1
        
        #get predictions from existing tracks
        for track in self.tracks:
            track.predict()

        no_update = self.no_update and self.frame_count > 5
         
        log_probs = torch.zeros(len(self.tracks), len(means))
        for i, track in enumerate(self.tracks):
            for j, mean in enumerate(means):
                m = PositionMeasurement(means[j].cpu(), torch.diag(covs[j]).cpu(), time=track.kf.time)
                log_prob = track.kf.log_likelihood_of_update(m)
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
                    if not no_update: self.tracks[t].update(means[d], covs[d])
                else:
                    unassigned.append(d)
            for d in unassigned:
                new_track = MocapTrack(means[d], covs[d])
                self.tracks.append(new_track)


        track_means, track_covs, track_ids = [means.new_empty(0,3).cpu()], [means.new_empty(0,3).cpu()], []
        for t, track in enumerate(self.tracks):
            onstreak = track.hit_streak >= self.min_hits
            warmingup = self.frame_count <= self.min_hits
            cond = track.wasupdated and (onstreak or warmingup)
            if cond or no_update:
                track_means.append(track.mean.unsqueeze(0))
                track_covs.append(track.cov.diag().unsqueeze(0))
                track_ids.append(track.id)
        
        track_means = torch.cat(track_means)
        track_covs = torch.cat(track_covs)
        track_ids = torch.tensor(track_ids)

        self.tracks = [track for track in self.tracks\
                       if track.time_since_update < self.max_age]
        
        det_means = means.detach().unsqueeze(0).cpu()
        det_covs = covs.detach().unsqueeze(0).cpu()
        track_means = track_means.detach().unsqueeze(0).cpu()
        track_covs = track_covs.detach().unsqueeze(0).cpu()
        det_ids = torch.zeros(len(means)).unsqueeze(0)
        track_ids = track_ids.unsqueeze(0) + 1 
        result = {
            # 'pred_position_mean': torch.cat([det_means, track_means], dim=1).numpy(),
            # 'pred_position_cov': torch.cat([det_covs, track_covs], dim=1).numpy(),
            # 'track_ids': torch.cat([det_ids, track_ids], dim=1).numpy(),
            'pred_position_mean': track_means.numpy(),
            'pred_position_cov': track_covs.numpy(),
            'track_ids': track_ids.numpy(),
        }
        return result


    def forward_test(self, data, **kwargs):
        mean, cov = self._forward(data)
        # gt_pos = data['mocap']['gt_positions'][0]
        # gt_labels = data['mocap']['gt_labels'][0]

        # is_node = gt_labels == 0
        # final_mask = ~is_node
        # z_is_zero = gt_pos[:, -1] == 0.0
        # final_mask = final_mask & ~z_is_zero
        # gt_pos = gt_pos[final_mask]
        # gt_labels = gt_labels[final_mask]

        result = {
            'pred_position_mean': mean.cpu().detach().unsqueeze(0).numpy(),
            'pred_position_cov': cov.cpu().detach().unsqueeze(0).numpy(),
            # 'pred_obj_prob': obj_probs[is_obj].cpu().detach().unsqueeze(0).numpy(),
            'pred_obj_prob': np.ones((1, len(mean))),
            'track_ids': np.zeros((1, len(mean)))
        }
        return result


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
        
        num_samples = len(data[0][('mocap','mocap')]['gt_positions'])

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

