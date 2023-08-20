import numpy as np
import torch
import torch.nn as nn
import cv2
import json
from tqdm import trange, tqdm
from collections import defaultdict
import torch.distributions as D
from trackeval.metrics import CLEAR, HOTA, Identity
from .viz import *
import torch.nn.functional as F
from mango import Tuner, scheduler
from scipy.stats import uniform
from mango.domain.distribution import loguniform
from functools import partial
from associator import matching_associator, map_associator
from initiator import distance_initiator
from deleter import staleness_deleter
from multi_object_tracker import MultiObjectKalmanTracker
from sleep_multi_object_tracker import SleepingMultiObjectKalmanTracker
from collections import defaultdict
from data_utils import point
from multiprocessing import Pool

from motion_models import steerable_model


class TrackingTuner(nn.Module):
    def __init__(self, param_space={
            'initiator_thres': uniform(0.1, 30),
            'associator_thres': uniform(0.1, 30),
            'dt': [1],
            'std_acc': uniform(0.01, 3),
            'weights_thres': uniform(0.1, 1 - 0.1),
            'weights_mode': ['softmax', 'binary'],
            'cov_scale': uniform(0.05, 10-0.05),
            'I_scale': uniform(0, 500),
            'update_count_thres': np.arange(3, 10),
            'staleness_thres': np.arange(1,10),
            'sleep_staleness_thres': np.arange(1,100),
            'sleep_associator_thres': uniform(0.1, 200),
            }):
        super().__init__()
        self.param_space = param_space

        self.init_params = {
            'I_scale': 60,
            'associator_thres': 25,
            'cov_scale': 5,
            'dt': 1,
            'initiator_thres': 80,
            'staleness_thres': 3,
            'std_acc': 1.2,
            'update_count_thres': 3,
            'weights_mode': 'binary',
            'weights_thres': 0.6,
            'sleep_associator_thres': 80,
            'sleep_staleness_thres': 100
        }

    def tune(self, preds, gt, num_iteration=500):
        obj_fn = partial(objective, preds=preds, gt=gt, num_samples=100)
        obj_fn = scheduler.serial(obj_fn)
        
        #parallel_dec = scheduler.parallel(8)
        #obj_fn = parallel_dec(obj_fn)
        
        conf_dict = dict(num_iteration=num_iteration)#@, initial_custom=self.init_params)
        tuner = Tuner(self.param_space, obj_fn, conf_dict)
        result = tuner.maximize()
        return result

    def evaluate(self, preds, gt, **params):
        tracker_outputs = run_tracker(preds, **params)
        eval_res = evaluate(tracker_outputs, gt)
        return tracker_outputs, eval_res

#preds, gt are lists, one element per valset
def objective(preds, gt, num_samples=1000, **params):
    hota = []
    for i in range(len(preds)):
        tracker_outputs = run_tracker(preds[i], **params)
        eval_res = evaluate(tracker_outputs, gt[i], num_samples=num_samples)
        hota.append(np.mean(eval_res['HOTA']))
    min_val = np.min(hota)
    return min_val

def evaluate(preds, gt, num_samples=1000):
    class_info = ClassInfo()
    all_gt_pos = gt['all_gt_pos']
    all_gt_ids = gt['all_gt_ids']
    all_gt_rot = gt['all_gt_rot']
    all_gt_labels = gt['all_gt_labels']

    res = {}
    res['num_gt_dets'] = all_gt_ids.shape[0] * all_gt_ids.shape[1]
    res['num_gt_ids'] = len(torch.unique(all_gt_ids))
    flat_ids = torch.cat([x.flatten() for x in preds['track_ids']])
    unique_ids = torch.unique(flat_ids).numpy()
    if len(unique_ids) == 0: #should only happen during tuning with really bad params
        return {'HOTA': 0}
    max_id = int(np.max(unique_ids))
    res['num_tracker_ids'] = max_id+1 #len(torch.unique(flat_ids))
    res['num_timesteps'] = len(all_gt_ids)
    res['tracker_ids'] = []
    res['gt_ids'] = all_gt_ids.numpy().astype(int)
    res['similarity_scores'] = []
    res['grid_scores'] = []
    res['nll'] = []
    res['num_tracker_dets'] = 0

    all_probs, all_dists = [], []
    for i in range(res['num_timesteps']):
        pred_means = preds['track_means'][i]
        pred_covs = preds['track_covs'][i]
        pred_ids = preds['track_ids'][i]
        res['num_tracker_dets'] += len(pred_ids)
        res['tracker_ids'].append(pred_ids.numpy().astype(int))
        gt_pos = all_gt_pos[i]
        gt_rot = all_gt_rot[i]
        gt_label = all_gt_labels[i]
        #gt_grid = all_gt_grids[i]
        
        dists, probs = [], []
        scores = []
        grid_scores = []
        nll = []
        for j in range(len(pred_means)):
            try:
                dist = D.MultivariateNormal(pred_means[j].cuda(), pred_covs[j].cuda())
            except ValueError:
                cov = pred_covs[j]
                cov[0,1] = cov[1,0]
                dist = D.MultivariateNormal(pred_means[j].cuda(), cov.cuda())

            num_gt = len(gt_pos)
            for k in range(num_gt):
                pos = gt_pos[k]
                label = gt_label[k].item()
                w = class_info.id2width(label)
                h = class_info.id2height(label)
                # if pos[0] == -1 or pos[1] == -1:
                    # continue
                
                nll.append(dist.log_prob(pos.cuda()))
                samples = dist.sample([num_samples]).cpu()
                angle = rot2angle(gt_rot[k], return_rads=False)
                rec, _ = gen_rectange(gt_pos[k], angle, w=w, h=h)
                mask = points_in_rec(samples, rec)
                scores.append(mask.mean())

        if len(scores) == 0:
            scores = torch.empty(len(gt_pos), 0).numpy()
        else:
            scores = torch.tensor(scores).reshape(len(pred_means), -1)
            grid_scores = torch.tensor(grid_scores).reshape(len(pred_means), -1)
            grid_scores = grid_scores.numpy().T
            scores = scores.numpy().T
            nll = torch.tensor(nll).reshape(len(pred_means), -1)

        res['similarity_scores'].append(scores)
        res['grid_scores'].append(grid_scores)
        res['nll'].append(nll)
    
    out = {}
    clear = CLEAR({'THRESHOLD': 0.5, 'PRINT_CONFIG': False})
    hota = HOTA()
    identity = Identity({'THRESHOLD': 0.5, 'PRINT_CONFIG': False})


    out = clear.eval_sequence(res)
    out = {k : float(v) for k,v in out.items()}

    
    hout = hota.eval_sequence(res)
    #means = {k + '_mean' : v.mean() for k, v in hout.items()}
    
    # means = {k: np.array(v).mean() for k, v in hout.items()}
    # out.update(means)
    
    hout = {k: v.tolist() for k,v in hout.items()}
    out.update(hout)

    iout = identity.eval_sequence(res)
    iout = {k : float(v) for k,v in iout.items()}
    out.update(iout)

    #out['nll_vals'] = nll.tolist()
    #out['grid_scores'] = scores.tolist()

    # with open(fname, 'w') as f:
        # json.dump(out, f)
    return out


def run_tracker(preds, **params):
    for key, val in preds.items():
        num_frames = len(val)

    initiator  = distance_initiator(dist_threshold=params['initiator_thres'], update_count_threshold=params['update_count_thres'])
    #associator = matching_associator(distance_threshold=params['associator_thres']) #2.5
    associator = map_associator(distance_threshold=params['associator_thres']) #2.5
    ssociator = map_associator(distance_threshold=params['sleep_associator_thres']) #2.5
    deleter = staleness_deleter(staleness_threshold=params['staleness_thres'])
    sleep_deleter = staleness_deleter(staleness_threshold=params['sleep_staleness_thres'])
    #tracker = MultiObjectKalmanTracker(std_acc=params['std_acc'],initiator=initiator,associator=associator, deleter=deleter)
    #tracker = SleepingMultiObjectKalmanTracker(std_acc=params['std_acc'],initiator=initiator,associator=associator, deleter=deleter, sleep_associator=sleep_associator, sleep_deleter=sleep_deleter)
    mm = steerable_model(std_accl=0.1, std_accy=0.1)
    tracker = MultiObjectKalmanTracker(mm=mm, initiator=initiator,associator=associator, sleep_threshold=30,kill_threshold=5,update_count_threshold=0)

    track_results, det_results = [], []
    for i in range(0, num_frames):
        t = params['dt'] * i
        dets = defaultdict(list)
        for key, val in preds.items():
            d = val[i]
            if params['weights_mode'] == 'softmax':
                weights = d['weights'].flatten()
                weights = torch.softmax(weights, dim=0)
            elif params['weights_mode'] == 'binary':
                logits = d['binary_logits']
                weights = logits.sigmoid()
            else:
                raise ValueError('weights_mode must be softmax or binary')
            mask = weights >= params['weights_thres']
            means = d['mean'].reshape(-1, 2)[mask] #/ 100
            covs = d['cov'].reshape(-1, 2, 2)[mask]
            covs = params['cov_scale'] * covs + params['I_scale'] * torch.eye(2,2)
            #covs = covs / 100
            for j in range(len(means)):
                covj = covs[j]
                #covj[0, 1] = covj[1, 0] = 0
                # covj[0,1] = 0
                # covj[1,0] = 0
                # covj = torch.eye(2) * 30
                p = point(t, pos=means[j].unsqueeze(1), cov=covj, source=key)
                dets[key].append(p)
        det_results.append(dets)
        out = tracker.update(t, dets)
        track_results.append(out)

    
    outputs = {}
    filtered_dets =  {k: [] for k in preds.keys()}
    for i, dr in enumerate(det_results):
        #for did, points in dr.items():
        for did in preds.keys():
            frame_means, frame_covs = [], []
            points = dr[did]
            for p in points:
                mean = p.pos[0:2].squeeze().detach() #* 100
                cov = p.cov[0:2, 0:2].squeeze().detach() #* 100
                frame_means.append(mean)
                frame_covs.append(cov)
            filtered_dets[did].append({'mean': frame_means, 'cov': frame_covs})
        #outputs['det_means'].append(frame_means)
        #outputs['det_covs'].append(frame_covs)
    outputs['filtered_dets'] = filtered_dets
  
    outputs['det_means'] = []
    outputs['det_covs'] = []
    for i, dr in enumerate(det_results):
        frame_means, frame_covs = [], []
        for did, points in dr.items():
            for p in points:
                mean = p.pos[0:2].squeeze().detach() #* 100
                cov = p.cov[0:2, 0:2].squeeze().detach() #* 100
                frame_means.append(mean)
                frame_covs.append(cov)
        outputs['det_means'].append(frame_means)
        outputs['det_covs'].append(frame_covs)
    
    outputs.update({'track_means': [], 'track_covs': [], 'track_ids': []})
    for i, tr in enumerate(track_results):
        frame_means, frame_covs, frame_ids = [], [], []
        for tid, tpoint in tr.items():
            mean = tpoint.x[0:2].squeeze().detach()
            cov = tpoint.cov[0:2, 0:2].squeeze().detach()
            frame_means.append(mean) #* 100)
            frame_covs.append(cov) #* 100)
            #frame_ids.append(int(tid[-1]))
            frame_ids.append(int(tid))
        outputs['track_means'].append(frame_means)
        outputs['track_covs'].append(frame_covs)
        outputs['track_ids'].append(torch.tensor(frame_ids))
    return outputs
