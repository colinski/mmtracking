import numpy as np
import torch
import cv2
import json
from tqdm import trange, tqdm
from collections import defaultdict
import torch.distributions as D
from trackeval.metrics import CLEAR, HOTA, Identity
from .viz import *
import torch.nn.functional as F

class TrackingEvaluator(Dataset):
    def __init__(self, num_samples=1000, clear_thres=0.5, identity_thres=0.5, **kwargs):
        self.num_samples = num_samples
        self.clear = CLEAR({'THRESHOLD': clear_thres, 'PRINT_CONFIG': False})
        self.hota = HOTA()
        self.identity = Identity({'THRESHOLD': identity_thres, 'PRINT_CONFIG': False})
        self.class_info = ClassInfo()
        
    def evaluate(self, preds, gt):
        all_gt_pos = gt['all_gt_pos']
        all_gt_ids = gt['all_gt_ids']
        all_gt_rot = gt['all_gt_rot']
        all_gt_labels = gt['all_gt_labels']

        res = {}
        res['num_gt_dets'] = all_gt_ids.shape[0] * all_gt_ids.shape[1]
        res['num_gt_ids'] = len(torch.unique(all_gt_ids))
        flat_ids = torch.cat([x.flatten() for x in preds['track_ids']])
        res['num_tracker_ids'] = len(torch.unique(flat_ids))
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
                    dist = D.MultivariateNormal(pred_means[j], pred_covs[j])
                except ValueError:
                    cov = pred_covs[j]
                    cov[0,1] = cov[1,0]

                num_gt = len(gt_pos)
                for k in range(num_gt):
                    pos = gt_pos[k]
                    label = gt_label[k].item()
                    w = self.class_info.id2width(label)
                    h = self.class_info.id2height(label)
                    # if pos[0] == -1 or pos[1] == -1:
                        # continue

                    nll.append(dist.log_prob(pos))
                    samples = dist.sample([self.num_samples])
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
        
        out = self.clear.eval_sequence(res)
        out = {k : float(v) for k,v in out.items()}

        
        hout = self.hota.eval_sequence(res)
        #means = {k + '_mean' : v.mean() for k, v in hout.items()}
        means = {k: v.mean() for k, v in hout.items()}
        #hout = {k: v.tolist() for k,v in hout.items()}
        #out.update(hout)
        out.update(means)

        iout = self.identity.eval_sequence(res)
        iout = {k : float(v) for k,v in iout.items()}
        out.update(iout)

        out['nll_vals'] = nll.tolist()
        out['grid_scores'] = scores.tolist()

        # with open(fname, 'w') as f:
            # json.dump(out, f)
        return out
