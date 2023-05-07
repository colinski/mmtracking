from mmtrack.datasets import build_dataset, build_dataloader
from mmtrack.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint, init_dist
import torch
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
import os
from mmtrack.apis import multi_gpu_test, single_gpu_test
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob
from associator import matching_associator
from initiator import distance_initiator
from multi_object_tracker import MultiObjectKalmanTracker
from collections import defaultdict
from data_utils import point
import cv2
import argparse
from mmtrack.datasets.mocap.eval import TrackingEvaluator
import json

def run_tracker(preds, initiator_thres=0.5, associator_thres=0.2, 
                dt=1, std_acc=0.1, weights_thres=0.6, weights_mode='softmax'):
    for key, val in preds.items():
        num_frames = len(val)

    initiator  = distance_initiator(dist_threshold=initiator_thres) # 1 5
    associator = matching_associator(distance_threshold=associator_thres) #2.5
    tracker = MultiObjectKalmanTracker(dt=dt,std_acc=std_acc,initiator=initiator,associator=associator)

    track_results, det_results = [], []
    for i in range(0, num_frames):
        dets = defaultdict(list)
        for key, val in preds.items():
            d = val[i]
            if weights_mode == 'softmax':
                weights = d['weights'].flatten()
                weights = torch.softmax(weights, dim=0)
            elif weights_mode == 'binary':
                logits = d['binary_logits']
                weights = logits.sigmoid()
            else:
                raise ValueError('weights_mode must be softmax or binary')
            #cat_dist = torch.distributions.Categorical(probs=weights)
            #if cat_dist.entropy() >= entropy_thres:
            #    continue
            #logits = d['binary_logits']
            #weights = logits.sigmoid()
            mask = weights >= weights_thres
            means = d['mean'].reshape(-1, 2)[mask] / 100
            covs = d['cov'].reshape(-1, 2, 2)[mask] / 100
            for j in range(len(means)):
                p = point(means[j].unsqueeze(1), covs[j], key)
                dets[key].append(p)
        det_results.append(dets)
        out = tracker.update(dt*i, dets)
        track_results.append(out)
        
    outputs =  {'det_means': [], 'det_covs': []}
    for i, dr in enumerate(det_results):
        frame_means, frame_covs = [], []
        for did, points in dr.items():
            for p in points:
                mean = p.pos[0:2].squeeze().detach() * 100
                cov = p.cov[0:2, 0:2].squeeze().detach() * 100
                frame_means.append(mean)
                frame_covs.append(cov)
        outputs['det_means'].append(frame_means)
        outputs['det_covs'].append(frame_covs)
        
    outputs.update({'track_means': [], 'track_covs': [], 'track_ids': []})
    for i, tr in enumerate(track_results):
        frame_means, frame_covs, frame_ids = [], [], []
        for tid, p in tr.items():
            mean = p.pos[0:2].squeeze().detach()
            cov = p.cov[0:2, 0:2].squeeze().detach()
            frame_means.append(mean * 100)
            frame_covs.append(cov * 100)
            frame_ids.append(int(tid[-1]))
        outputs['track_means'].append(frame_means)
        outputs['track_covs'].append(frame_covs)
        outputs['track_ids'].append(torch.tensor(frame_ids))
    return outputs


parser = argparse.ArgumentParser()
parser.add_argument('expdir', type=str)
parser.add_argument('--mode', type=str, default='test')
args = parser.parse_args()

test_dir = f'{args.expdir}/{args.mode}'
os.makedirs(test_dir, exist_ok=True)

cfg = Config.fromfile(f'{args.expdir}/config.py')
model = build_model(cfg.model)
model.init_weights()
checkpoint = load_checkpoint(model, f'{args.expdir}/log/latest.pth', map_location='cpu')
model = MMDataParallel(model, device_ids=[0])
evaluator = TrackingEvaluator()

if args.mode == 'val':
    test_cfgs = cfg.data.val
elif args.mode == 'test':
    test_cfgs = cfg.data.test
else:
    raise ValueError('mode must be val or test')
for i, test_cfg in enumerate(test_cfgs):
    testset = build_dataset(test_cfg)
    gt = testset.collect_gt()
    testloader = build_dataloader(testset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    outputs = single_gpu_test(model, testloader, show=False, out_dir=None)
    track_outputs = run_tracker(outputs, weights_mode='binary', weights_thres=0.3)
    res = evaluator.evaluate(track_outputs, gt)
    res['cfg'] = test_cfg
    with open(f'{test_dir}/results{i}.json', 'w') as f:
        json.dump(res, f)

    testset.write_video(track_outputs, fname=f'{test_dir}/video{i}.mp4', start_idx=0, end_idx=500)



# def run_search(preds, dataset, gt, weights_mode='softmax', num_trials=100,
               # weight_thres_dist=torch.distributions.Uniform(0.5,1),
               # initiator_thres_dist=torch.distributions.Uniform(0,1),
               # associator_thres_dist = torch.distributions.Uniform(0,1)):
    # all_results = []
    # for k in trange(num_trials):
        # params = {}
        # params['weights_thres'] = weight_thres_dist.sample([1]).item()
        # params['initiator_thres'] = initiator_thres_dist.sample([1]).item()
        # params['associator_thres'] = associator_thres_dist.sample([1]).item()
        # params['weights_mode'] = weights_mode

        # outputs = run_tracker(preds, **params)
        # try:
            # out = dataset.eval_mot(outputs, gt)
        # except:
            # continue
        # out.update(params)
        # all_results.append(out)
    # df = pd.DataFrame(all_results)
    # df = df.set_index(['weights_mode', 'weights_thres', 'initiator_thres', 'associator_thres'])
    # return df

