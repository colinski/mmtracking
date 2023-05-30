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
#from mmtrack.datasets.mocap.eval import TrackingEvaluator
import json
import pickle
from mmtrack.utils.heatmap import gen_heatmap

parser = argparse.ArgumentParser()
parser.add_argument('expdir', type=str)
args = parser.parse_args()

train_dir = f'{args.expdir}/train'
val_dir = f'{args.expdir}/val'
test_dir = f'{args.expdir}/test'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

cfg = Config.fromfile(f'{args.expdir}/config.py')
model = build_model(cfg.model)
model.init_weights()
checkpoint = load_checkpoint(model, f'{args.expdir}/log/latest.pth', map_location='cpu')
model = MMDataParallel(model, device_ids=[0])

all_outputs = []
for i, test_cfg in enumerate(cfg.data.train):
    testset = build_dataset(test_cfg)
    testloader = build_dataloader(testset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    det_outputs = single_gpu_test(model, testloader, show=False, out_dir=None)
    all_outputs.append({'detections': det_outputs, 'data_cfg': test_cfg, 'full_cfg': cfg, 'dataset':testset})
torch.save(all_outputs, f'{train_dir}/outputs.pth')


all_outputs = []
for i, test_cfg in enumerate(cfg.data.val):
    testset = build_dataset(test_cfg)
    testloader = build_dataloader(testset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    det_outputs = single_gpu_test(model, testloader, show=False, out_dir=None)
    all_outputs.append({'detections': det_outputs, 'data_cfg': test_cfg, 'full_cfg': cfg, 'dataset':testset})
torch.save(all_outputs, f'{val_dir}/outputs.pth')

all_outputs = []
for i, test_cfg in enumerate(cfg.data.test):
    testset = build_dataset(test_cfg)
    testloader = build_dataloader(testset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    det_outputs = single_gpu_test(model, testloader, show=False, out_dir=None)
    all_outputs.append({'detections': det_outputs, 'data_cfg': test_cfg, 'full_cfg': cfg, 'dataset':testset})
torch.save(all_outputs, f'{test_dir}/outputs.pth')




