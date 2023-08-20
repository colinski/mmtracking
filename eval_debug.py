from mmtrack.datasets import build_dataset, build_dataloader
from mmtrack.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint, init_dist
import torch
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
import os
from mmtrack.apis import multi_gpu_test, single_gpu_test
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
from associator import matching_associator, map_associator
from initiator import distance_initiator
from multi_object_tracker import MultiObjectKalmanTracker
from deleter import staleness_deleter
from scheduler import MultiEntropyScheduler,RandomScheduler,MinDistanceScheduler
from sensors import Sensor
from collections import defaultdict
from data_utils import point
import cv2
import seaborn as sns
import scipy
from tqdm import tqdm, trange
import pandas as pd
from mmtrack.datasets.mocap.eval import TrackingTuner, evaluate, run_tracker


expdir = '/home/csamplawski/exps/2023-05-31/trucks12_realsense_yolotiny_480p/'


#train_outputs = torch.load(f'{expdir}/train/outputs.pth')
val_outputs = torch.load(f'{expdir}/val/outputs.pth')
#test_outputs = torch.load(f'{expdir}/test/outputs.pth')


cfg = Config.fromfile(f'{expdir}/config.py')

idx_set = np.arange(len(val_outputs))
preds, gt = [], []
for idx in idx_set:
    preds.append(val_outputs[idx]['detections'])
    testset = build_dataset(cfg.data.val[idx])
    gt.append(testset.collect_gt(valid_only=True))
    testset.write_video(None, fname=f'/tmp/debug_vid_{idx}.mp4', start_idx=0, end_idx=499)

to_save = {}
to_save['detections'] = preds
to_save['gt'] = gt
torch.save(to_save, '/tmp/val_outputs_and_gt.pth')
import ipdb; ipdb.set_trace() # noqa

outputs = {}
outputs['det_means'] = []
outputs['det_covs'] = []
for i in range(0, 500):
    frame_means = []
    frame_covs = []
    for key, val in preds[0].items():
        d = val[i]
        logits = d['binary_logits']
        weights = logits.sigmoid()
        mask = weights >= 0.5
        means = d['mean'].reshape(-1, 2)[mask].detach()
        covs = d['cov'].reshape(-1, 2, 2)[mask].detach()
        for j in range(len(means)):
            frame_means.append(means[j])
            frame_covs.append(covs[j])
    outputs['det_means'].append(frame_means)
    outputs['det_covs'].append(frame_covs)


testset.write_video(outputs, fname='/tmp/debug_vid.mp4', start_idx=0, end_idx=100, plot_detections=True)


# tuner = TrackingTuner()
# tune_res = tuner.tune(preds, gt, num_iteration=50)


valset_best_params = {'I_scale': 150.5778602333644,
 'associator_thres': 29.02589864772972,
 'cov_scale': 0.23698655225565973,
 'dt': 1,
 'initiator_thres': 6.497818508263305,
 'sleep_associator_thres': 79.90773522253225,
 'sleep_staleness_thres': 85,
 'staleness_thres': 4,
 'std_acc': 1.4195332561229417,
 'update_count_thres': 7,
 'weights_mode': 'softmax',
 'weights_thres': 0.5528922854738426}


# In[27]:


#tune_res['best_params']
valset_best_params = {'I_scale': 67.20327904075086,
 'associator_thres': 26.86536037418154,
 'cov_scale': 5.3921567229666625,
 'dt': 1,
 'initiator_thres': 8.103731829632142,
 'staleness_thres': 3,
 'std_acc': 1.198386012332232,
 'update_count_thres': 3,
 'weights_mode': 'binary',
 'weights_thres': 0.9107456313218705}
#valset_best_params['associator_thres'] /= 2
valset_best_params['sleep_associator_thres'] = 26.86536037418154*3
valset_best_params['sleep_staleness_thres'] = 50
valset_best_params['weights_thres'] = 0.6046872346346
valset_best_params['initiator_thres'] = 8.103731829632142*10


# valset_best_params = {'I_scale': 32.05146084404015,
 # 'associator_thres': 27.485945131849427,
 # 'cov_scale': 1.0655102364829308,
 # 'dt': 1,
 # 'initiator_thres': 28.767695481134734,
 # 'sleep_associator_thres': 2.2840447559906107,
 # 'sleep_staleness_thres': 55,
 # 'staleness_thres': 8,
 # 'std_acc': 1.5938317288427462,
 # 'update_count_thres': 4,
 # 'weights_mode': 'softmax',
 # 'weights_thres': 0.24687367857563275}


all_results = []
for i in range(len(preds)):
    tracker_outputs = run_tracker(preds[i], **valset_best_params)
    res = evaluate(tracker_outputs, gt[i])
    all_results.append(res)
    print(np.mean(res['HOTA']))

