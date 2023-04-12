import torch
from mmtrack.datasets import build_dataloader, build_dataset
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange
import numpy as nps
import cv2
from multi_object_tracker import MultiObjectKalmanTracker
from mmtrack.datasets.mocap.viz import init_fig, gen_rectange, gen_ellipse, rot2angle, points_in_rec
import json
from associator import matching_associator
from initiator import distance_initiator
from collections import defaultdict
import torch.nn.functional as F
font = {'size'   : 12}
matplotlib.rc('font', **font)


img_norm_cfg = dict(mean=[0,0,0], std=[255,255,255], to_rgb=True)
img_pipeline = [
    dict(type='DecodeJPEG'),
    dict(type='LoadFromNumpyArray'),
    dict(type='Resize', img_scale=(270, 480), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

pipelines = {
    'zed_camera_left': img_pipeline,
} 

valid_mods=['mocap', 'zed_camera_left']
valid_nodes=[1,2,3,4]

data_root1 = 'data/mmm/2022-09-01/trucks1_lightsT_obstaclesF/train'
data_root2 = 'data/mmm/2022-09-01/trucks2_lightsT_obstaclesF/train'
trainset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir= f'/dev/shm/cache_train/',
        hdf5_fnames=[
            f'{data_root1}/mocap.hdf5',
            f'{data_root1}/node_1/zed.hdf5',
            f'{data_root1}/node_2/zed.hdf5',
            f'{data_root1}/node_3/zed.hdf5',
            f'{data_root1}/node_4/zed.hdf5',
            f'{data_root2}/mocap.hdf5',
            f'{data_root2}/node_1/zed.hdf5',
            f'{data_root2}/node_2/zed.hdf5',
            f'{data_root2}/node_3/zed.hdf5',
            f'{data_root2}/node_4/zed.hdf5',
        ],
        valid_mods=valid_mods,
        valid_nodes=valid_nodes,
        include_z=False, #(x,y) position only
    ),
    num_future_frames=0,
    num_past_frames=0, #sequence of length 5
    pipelines=pipelines,
)
trainset = build_dataset(trainset)

data_root1 = 'data/mmm/2022-09-01/trucks1_lightsT_obstaclesF/val'
data_root2 = 'data/mmm/2022-09-01/trucks2_lightsT_obstaclesF/val'
valset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir= f'/dev/shm/cache_val/',            
        hdf5_fnames=[
            f'{data_root1}/mocap.hdf5',
            f'{data_root1}/node_1/zed.hdf5',
            f'{data_root1}/node_2/zed.hdf5',
            f'{data_root1}/node_3/zed.hdf5',
            f'{data_root1}/node_4/zed.hdf5',
            f'{data_root2}/mocap.hdf5',
            f'{data_root2}/node_1/zed.hdf5',
            f'{data_root2}/node_2/zed.hdf5',
            f'{data_root2}/node_3/zed.hdf5',
            f'{data_root2}/node_4/zed.hdf5',
        ],
        valid_mods=valid_mods,
        valid_nodes=valid_nodes,
        include_z=False, #(x,y) position only
    ),
    num_future_frames=0,
    num_past_frames=0, #sequence of length 5
    pipelines=pipelines,
)
valset = build_dataset(valset)
print(len(valset))

data_root1 = 'data/mmm/2022-09-01/trucks1_lightsT_obstaclesF/test'
data_root2 = 'data/mmm/2022-09-01/trucks2_lightsT_obstaclesF/test'
testset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir= f'/dev/shm/cache_test/',
        hdf5_fnames=[
            f'{data_root1}/mocap.hdf5',
            f'{data_root1}/node_1/zed.hdf5',
            f'{data_root1}/node_2/zed.hdf5',
            f'{data_root1}/node_3/zed.hdf5',
            f'{data_root1}/node_4/zed.hdf5',
            f'{data_root2}/mocap.hdf5',
            f'{data_root2}/node_1/zed.hdf5',
            f'{data_root2}/node_2/zed.hdf5',
            f'{data_root2}/node_3/zed.hdf5',
            f'{data_root2}/node_4/zed.hdf5',
        ],
        valid_mods=valid_mods,
        valid_nodes=valid_nodes,
        include_z=False, #(x,y) position only
    ),
    num_future_frames=0,
    num_past_frames=0, #sequence of length 5
    pipelines=pipelines,
)
testset = build_dataset(testset)

def get_weights(preds, node='node_3'):
    all_weights = []
    for key, val in preds.items():
        if node not in key:
            continue
        for x in val:
            weights = x['weights']
            all_weights.append(weights)
    all_weights = torch.stack(all_weights)
    return all_weights

def get_hist_vals(all_weights):
    hist_vals = []
    for weights in all_weights:
        wdist = torch.distributions.Categorical(logits=weights.flatten())
        hist_vals.append(wdist.entropy())
    return torch.tensor(hist_vals)


expdir = 'logs/trucks12_lightsT_obstaclesF_zed_nodes1234_yolo_tiny_28x20_binary_scale'
preds = torch.load(f'{expdir}/test/outputs.pt')
all_weights = get_weights(preds)
hist_vals = get_hist_vals(all_weights)


for key, val in preds.items():
    num_frames = len(val)

from data_utils import point

dt = 0.05
dt = 1
initiator  = distance_initiator(dist_threshold=0.5) # 1 0.2 0.5 1 5
associator = matching_associator(distance_threshold=0.2) #2.5
tracker = MultiObjectKalmanTracker(dt=dt,std_acc=.1,initiator=initiator,associator=associator)

track_results, det_results = [], []
for i in trange(num_frames):
    dets = defaultdict(list)
    for key, val in preds.items():
        d = val[i]
        weights = d['weights'].flatten()
        weights = torch.softmax(weights, dim=0)
        cat_dist = torch.distributions.Categorical(probs=weights)
        if cat_dist.entropy() > 2:
            continue
        logits = d['binary_probs']
        weights = logits.sigmoid()
        mask = weights >= 0.6
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
    #outputs['det_means'].append(frame_means)
    #outputs['det_covs'].append(frame_covs)


outputs.update({'track_means': [], 'track_covs': []})
for i, tr in enumerate(track_results):
    frame_means, frame_covs = [], []
    for tid, p in tr.items():
        mean = p.pos[0:2].squeeze().detach()
        cov = p.cov[0:2, 0:2].squeeze().detach()
        frame_means.append(mean * 100)
        frame_covs.append(cov * 100)
    #if len(frame_means) == 0:
    #    print(i, tr)
    #frame_means = torch.cat(frame_means)
    #frame_covs = torch.cat(frame_covs)
    outputs['track_means'].append(frame_means)
    outputs['track_covs'].append(frame_covs)


#valset.write_video(outputs, fname='vids/trucks12_track_sigmoid_val.mp4', start_idx=0, end_idx=1100)
#valset.write_video(outputs, fname='vids/trucks2_track_sigmoid_val.mp4', start_idx=1500, end_idx=2000)

#testset.write_video(outputs, fname='vids/debug_trucks1.mp4', start_idx=0, end_idx=500);
testset.write_video(outputs, fname='vids/debug_trucks2.mp4', start_idx=4000, end_idx=4500);
#testset.write_video(outputs, fname='vids/trucks2_track_sigmoid_test_crossval.mp4', start_idx=4000, end_idx=4500);
