from mmtrack.datasets import build_dataset, build_dataloader
from mmtrack.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint
import torch
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
import os
from mmtrack.apis import multi_gpu_test, single_gpu_test
import numpy as np
import glob
from collections import defaultdict
import cv2
import argparse
from mmtrack.datasets.mocap.eval import TrackingEvaluator
import json
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('expdir', type=str)
args = parser.parse_args()

val_dir = f'{args.expdir}/val'
os.makedirs(val_dir, exist_ok=True)

cfg = Config.fromfile(f'{args.expdir}/config.py')
model = build_model(cfg.model)
model.init_weights()
checkpoint = load_checkpoint(model, f'{args.expdir}/log/latest.pth', map_location='cpu')
model = MMDataParallel(model, device_ids=[0])
evaluator = TrackingEvaluator()

val_cfgs = cfg.data.val
det_outputs, gt = [], []
for val_cfg in val_cfgs:
    valset = build_dataset(val_cfg)
    valloader = build_dataloader(valset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    outputs = single_gpu_test(model, valloader, show=False, out_dir=None)
    det_outputs.append(outputs)
    gt.append(valset.collect_gt())

tune_result = evaluator.tune(det_outputs, gt)

with open(f'{val_dir}/tune_result.pickle', 'wb') as f:
    pickle.dump(tune_result, f)

# for test_cfg in test_cfgs:
    # testset = build_dataset(test_cfg)
    # gt = testset.collect_gt()
    # testloader = build_dataloader(testset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    # det_outputs = single_gpu_test(model, testloader, show=False, out_dir=None)
    # track_outputs, track_result = evaluator.evaluate(det_outputs, gt, **tune_result['best_params'])
    # import ipdb; ipdb.set_trace() # noqa


#track_outputs = run_tracker(outputs, weights_mode='binary', weights_thres=0.3)

# res = evaluator.evaluate(track_outputs, gt)

# res['cfg'] = val_cfg
# with open(f'{val_dir}/results{i}.json', 'w') as f:
    # json.dump(res, f)

#valset.write_video(track_outputs, fname=f'{val_dir}/video{i}.mp4', start_idx=0, end_idx=500)



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

