import torch
from mmtrack.datasets import build_dataloader, build_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange
import numpy as np
import cv2
from tracker import TorchMultiObsKalmanFilter
from mmtrack.datasets.mocap.viz import init_fig, gen_rectange, gen_ellipse, rot2angle, points_in_rec
import json
import sys




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

root = sys.argv[1]
expdir = sys.argv[2]

#data_root = 'data/mmm/2022-09-01/trucks1_lightsT_obstaclesF/train'
data_root = f'{root}/train'
trainset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir= f'/dev/shm/cache_train/',
        hdf5_fnames=[
            f'{data_root}/mocap.hdf5',
            f'{data_root}/node_1/zed.hdf5',
            f'{data_root}/node_2/zed.hdf5',
            f'{data_root}/node_3/zed.hdf5',
            f'{data_root}/node_4/zed.hdf5',
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

data_root = f'{root}/val'
valset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir= f'/dev/shm/cache_val/',
        hdf5_fnames=[
            f'{data_root}/mocap.hdf5',
            f'{data_root}/node_1/zed.hdf5',
            f'{data_root}/node_2/zed.hdf5',
            f'{data_root}/node_3/zed.hdf5',
            f'{data_root}/node_4/zed.hdf5',
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

data_root = f'{root}/test'
testset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir= f'/dev/shm/cache_test/',
        hdf5_fnames=[
            f'{data_root}/mocap.hdf5',
            f'{data_root}/node_1/zed.hdf5',
            f'{data_root}/node_2/zed.hdf5',
            f'{data_root}/node_3/zed.hdf5',
            f'{data_root}/node_4/zed.hdf5',
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

def collect_outputs(preds):
    for key, val in preds.items():
        num_frames = len(val) #should be same for all keys
    mean_seq, cov_seq = [], []
    for t in range(num_frames):
        means = [preds[key][t]['mean'].cpu() for key in preds.keys()]
        covs = [preds[key][t]['cov'].cpu() for key in preds.keys()]
        
        means = torch.stack(means) 
        covs = torch.stack(covs)
        nV, H, W, _ = means.shape
        means = means.reshape(nV*H*W, 2).t()
        covs = covs.reshape(nV*H*W, 2,2)
        mean_seq.append(means)
        cov_seq.append(covs)
    outputs = {'det_means': mean_seq, 'det_covs': cov_seq}
    return outputs

grid_x, grid_y = torch.meshgrid([torch.arange(0,700), torch.arange(0,500)])
grid = torch.stack([grid_x, grid_y], dim=-1).cuda().float()


def write_vid(frames, vid_name='pdf_node_1.mp4', size=100, key='log_probs', norm=True):
    vid = cv2.VideoWriter(vid_name, cv2.VideoWriter_fourcc(*'mp4v'), 20, (700,500))
    for lp in frames:
        lp = lp.numpy()
        lp = np.fliplr(lp)
        lp = lp.T
        # lp = cv2.resize(lp, dsize=(700,500), interpolation=cv2.INTER_LINEAR)
        if norm:
            lp = cv2.normalize(lp, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        lp = lp * 255
        lp = lp.astype(np.uint8)
        lp = cv2.applyColorMap(lp, cv2.COLORMAP_JET)
        #lp = cv2.copyMakeBorder(lp, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
        #lp = cv2.resize(lp, dsize=(700,500), interpolation=cv2.INTER_LINEAR)
        vid.write(lp)
    vid.release()


def get_pdf_frames(preds, key='zed_camera_left_node_1', size=500):
    frames = []
    for vals in preds[key][0:size]:
        mean, cov, weights = vals['mean'].cuda(), vals['cov'].cuda(), vals['weights'].cuda()
        H, W, _ = mean.shape
        mean = mean.reshape(H*W,2)
        cov = cov.reshape(H*W,2,2)
        weights = weights.reshape(H*W)
        normal = torch.distributions.MultivariateNormal(mean, cov)
        mix = torch.distributions.Categorical(probs=weights)
        dist = torch.distributions.MixtureSameFamily(mix, normal)
        nll_vals = dist.log_prob(grid).cpu()
        nll_vals = nll_vals.exp()
        #nll_vals[nll_vals < -15] = -15
        frames.append(nll_vals)
    frames = torch.stack(frames)
    #frames = frames / frames.max()
    return frames

datasets = {'train': trainset, 'val': valset, 'test': testset}
for ds in ['train', 'val', 'test']:
    dataset = datasets[ds]
    preds = torch.load(f'{expdir}/{ds}/outputs.pt')
    #gt = dataset.collect_gt()
    #outputs = collect_outputs(preds)
    #res, outputs = dataset.track_eval(outputs, gt)
    dataset.write_video(None, logdir=f'{expdir}/{ds}/', video_length=500)

    frames = get_pdf_frames(preds, 'zed_camera_left_node_1', size=500)
    write_vid(frames, f'{expdir}/{ds}/pdf_node_1.mp4')

    frames = get_pdf_frames(preds, 'zed_camera_left_node_2', size=500)
    write_vid(frames, f'{expdir}/{ds}/pdf_node_2.mp4')

    frames = get_pdf_frames(preds, 'zed_camera_left_node_3', size=500)
    write_vid(frames, f'{expdir}/{ds}/pdf_node_3.mp4')

    frames = get_pdf_frames(preds, 'zed_camera_left_node_4', size=500)
    write_vid(frames, f'{expdir}/{ds}/pdf_node_4.mp4')
