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
#from resource_constrained_tracking.tracker import TorchMultiObsKalmanFilter
from tracker import TorchMultiObsKalmanFilter
#from rct.tracker import MultiObsKalmanFilter

def linear_assignment(cost_matrix):
    cost_matrix = cost_matrix.cpu().detach().numpy()
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    assign_idx = np.array([[y[i], i] for i in x if i >= 0])
    sort_idx = np.argsort(assign_idx[:, 1])
    assign_idx = assign_idx[sort_idx]
    assign_idx = torch.from_numpy(assign_idx)
    return assign_idx.long()

class MultiTracker(nn.Module):
    def __init__(self, max_age=5, min_hits=1, mode='kf'):
        super().__init__()
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.frame_count = 0 
        self.obj_prob_thres = 0.5
        self.mode = mode
        self.track = TorchMultiObsKalmanFilter(dt=1, std_acc=1)

    def __call__(self, result):
        # obj_probs = result['det_obj_probs']
        # is_obj = obj_probs > self.obj_prob_thres
        means = result['det_means']
        covs = result['det_covs']

        # if self.frame_count == 0:
            # mean = means.mean(dim=1)
            # cov = torch.mean(torch.stack(covs), dim=0)
            # self.track = TorchMultiObsKalmanFilter(mean, cov, dt=1, std_acc=1)

                
        if self.mode == 'mean':
            track_mean = means.mean(axis=1)
            track_cov = np.stack(covs).mean(axis=0)
            track_mean = torch.from_numpy(track_mean)
            track_cov = torch.from_numpy(track_cov)
            result.update({
                'track_means': track_mean.unsqueeze(0),
                'track_covs': track_cov.unsqueeze(0),
                'track_ids': torch.zeros(1)
            })
            return result

        # slot_ids = torch.arange(len(obj_probs))[is_obj]
        # if len(means) == 0:
            # result.update({
                # 'track_means': torch.empty(0,2).cpu(),
                # 'track_covs': torch.empty(0,2,2).cpu(),
                # 'track_ids': torch.empty(0,1),
                # 'slot_ids': torch.empty(0,1)
            # })
            # return result

        self.frame_count += 1

        out = self.track(means, covs)
        import ipdb; ipdb.set_trace() # noqa

        # self.track.predict()
        # out = self.track.update(means, covs)

        track_mean = out[0][0:2].squeeze()
        track_cov = out[1][0:2, 0:2]

        # track_mean = track_mean.squeeze()
        # track_cov = torch.from_numpy(track_cov)

        result.update({
            'track_means': track_mean.unsqueeze(0).cpu(),
            'track_covs': track_cov.unsqueeze(0).cpu(),
            'track_ids': torch.zeros(1)
        })
        # result.update({
            # 'track_means': track_mean,
            # 'track_covs': track_cov,
            # 'track_ids': torch.zeros(1)
        # })

        return result
        
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
                if exp_probs[t,d] >= 0:
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

        if len(track_means) == 0:
            import ipdb; ipdb.set_trace() # noqa

        result.update({
            'track_means': track_means.detach().cpu(),
            'track_covs': track_covs.detach().cpu(),
            'track_ids': track_ids,
            'slot_ids': slot_ids
        })
        return result
class Tracker:
    def __init__(self, max_age=5, min_hits=1):
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
                'track_means': torch.empty(0,2).cpu(),
                'track_covs': torch.empty(0,2,2).cpu(),
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
                if exp_probs[t,d] >= 0:
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

        if len(track_means) == 0:
            import ipdb; ipdb.set_trace() # noqa

        result.update({
            'track_means': track_means.detach().cpu(),
            'track_covs': track_covs.detach().cpu(),
            'track_ids': track_ids,
            'slot_ids': slot_ids
        })
        return result
