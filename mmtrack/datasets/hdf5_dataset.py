# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import random
from abc import ABCMeta, abstractmethod

import numpy as np
from addict import Dict
from mmcv.utils import print_log
from mmdet.datasets.pipelines import Compose
from torch.utils.data import Dataset

from mmtrack.core.evaluation import eval_sot_ope
from mmtrack.datasets import DATASETS
import glob
import pickle
import cv2
import h5py
import torch
import json
import time
import torchaudio
from tqdm import trange
import matplotlib.pyplot as plt

def init_fig():
    fig = plt.figure(figsize=(16,9))
    axes = {}
    axes['zed_camera_left'] = plt.subplot2grid((3,4), (0,0)) #ax1
    axes['zed_camera_right'] = plt.subplot2grid((3,4), (0,1)) #ax2
    axes['zed_camera_depth'] = plt.subplot2grid((3,4), (0,2))
    axes['mocap'] = plt.subplot2grid((3,4), (0,3), rowspan=2)
    axes['realsense_camera_img'] = plt.subplot2grid((3,4), (1,0))
    axes['realsense_camera_depth'] = plt.subplot2grid((3,4), (1,1))
    axes['azimuth_static'] = plt.subplot2grid((3,4), (1,2))
    axes['range_doppler'] = plt.subplot2grid((3,4), (2,0))
    axes['detected_points'] = plt.subplot2grid((3,4), (2,1), projection='3d')
    axes['mic_waveform'] = plt.subplot2grid((3,4), (2,2))
    axes['mic_direction'] = plt.subplot2grid((3,4), (2,3), projection='polar')
    fig.suptitle('Title', fontsize=16)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    return fig, axes

def read_hdf5(f):
    data = {}
    for ms in f.keys():
        data[ms] = {}
        for k, v in f[ms].items():
            if k == 'mocap': #mocap or node_N
                data[ms]['mocap'] = v[()]
            else:
                data[ms][k] = {}
                for k2, v2 in f[ms][k].items():
                    if k2 == 'detected_points':
                        data[ms][k][k2] = v2[()]
                    else:
                        data[ms][k][k2] = v2[:]
    return data

@DATASETS.register_module()
class HDF5Dataset(Dataset, metaclass=ABCMeta):
    CLASSES = None
    def __init__(self,
                 hdf5_fname,
                 fps=20,
                 valid_keys=['mocap', 'zed_camera_left', 'zed_camera_depth'],
                 img_pipeline=[],
                 depth_pipeline=[],
                 azimuth_pipeline=[],
                 range_pipeline=[],
                 audio_pipeline=[],
                 test_mode=False,
                 is_random=False,
                 **kwargs):
        self.class2idx = {'truck': 1, 'node': 0}
        self.fname = hdf5_fname
        self.fps = fps
        self.is_random = is_random
        with h5py.File(self.fname, 'r') as f:
            self.data = read_hdf5(f)
        self.keys = list(self.data.keys())
        self.keys = np.array(self.keys)
        sort_idx = np.argsort(self.keys)
        self.keys = self.keys[sort_idx]
        self.timesteps = torch.from_numpy(self.keys.astype(int))
        self.img_pipeline = Compose(img_pipeline)
        self.depth_pipeline = Compose(depth_pipeline)
        self.range_pipeline = Compose(range_pipeline)
        self.azimuth_pipeline = Compose(azimuth_pipeline)
        self.audio_pipeline = Compose(audio_pipeline)
        self.test_mode = test_mode
        self.start_time = int(self.keys[0]) #+ (60*60*1000)
        self.end_time = int(self.keys[-1])
        elapsed_time = self.end_time - self.start_time
        self.num_frames = int(elapsed_time / 1000) * self.fps
        self.frame_len = int((1 / self.fps) * 1000)
        self.buffer = {}
        self.flag = np.zeros(len(self), dtype=np.uint8) #ones?
        self.valid_keys = valid_keys
    

    def __len__(self):
        return self.num_frames

    def parse_buffer(self):
        for key, val in self.buffer.items():
            if key == 'mocap':
                mocap_data = json.loads(val)
                positions = torch.tensor([d['normalized_position'] for d in mocap_data])
                labels = torch.tensor([self.class2idx[d['type']] for d in mocap_data])
                ids = torch.tensor([d['id'] for d in mocap_data])
                z_is_zero = positions[:, -1] == 0.0
                is_node = labels == self.class2idx['node']
                final_mask = ~z_is_zero #& ~is_node
                self.buffer[key] = {
                    'gt_positions': positions[final_mask],
                    'gt_labels': labels[final_mask].long(),
                    'gt_ids': ids[final_mask].long()
                }

            if key == 'zed_camera_left':
                img = cv2.imdecode(val, 1)
                self.buffer[key] = self.img_pipeline(img)
        
            if key == 'zed_camera_right':
                img = cv2.imdecode(val, 1)
                self.buffer[key] = self.img_pipeline(img)
        
            if key == 'zed_camera_depth':
                self.buffer[key] = self.depth_pipeline(val)
            
            if key == 'realsense_camera_img':
                img = cv2.imdecode(val, 1)
                self.buffer[key] = self.img_pipeline(img)

            if key == 'realsense_camera_depth':
                img = cv2.imdecode(val, 1)
                self.buffer[key] = self.img_pipeline(img)

            if key == 'azimuth_static':
                arr = np.nan_to_num(val)
                self.buffer[key] = self.azimuth_pipeline(arr)
                    
            if key == 'range_doppler':
                self.buffer[key] = self.range_pipeline(val.T)
            
            if key == 'mic_waveform':
                val = val.T
                val = val[1:5]
                self.buffer[key] = self.audio_pipeline(val)
                

    def fill_buffer(self, start_idx, end_time):
        start_idx = 0 
        for time in self.keys[start_idx:]:
            data = self.data[time]
            if 'mocap' in data.keys():
                self.buffer['mocap'] = data['mocap']
            if 'node_3' in data.keys():
                data = data['node_3']

            for key, val in data.items():
                if key in self.valid_keys:
                    self.buffer[key] = val
            
            if int(time) >= end_time:
                return
        
    def __getitem__(self, ind):
        if ind == 0 or self.is_random:
            self.buffer = {}
        start = int(self.start_time + ind * self.frame_len) #start is N frames after start_time
        diffs = torch.abs(self.timesteps - start) #find closest time as it isnt frame perfect
        min_idx = torch.argmin(diffs).item()
        end = int(self.timesteps[min_idx] + self.frame_len) #one frame worth of data
        self.fill_buffer(min_idx, end) #run through data and fill buffer
        self.parse_buffer() #convert to arrays
        return self.buffer

    def evaluate(self, outputs, **eval_kwargs):
        # pred_pos = np.array(outputs['pred_position'])
        # pred_pos = pred_pos.squeeze()
        # gt_pos = np.array(outputs['gt_position'])
        # sq_err = (pred_pos - gt_pos)**2
        mse = 0
        
        size = (1600, 900)
        fname = f'/tmp/latest_vid.mp4'
        vid = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, size)
        fig, axes = init_fig()

        for i in trange(len(self)):
            data = self[i]

            save_frame = False
            if 'mocap' in data.keys():
                save_frame = True
                axes['mocap'].clear()
                
                
                for pos in outputs['pred_position'][i]:
                    # if pos[-1] == 0.0: #z == 0, ignore
                        # continue
                    alpha = 0.5 if len(pos) > 1 else 1
                    axes['mocap'].scatter(pos[:, 1], pos[:, 0], alpha=alpha) # to rotate, longer side to be y axis

                for pos in data['mocap']['gt_positions']:
                    if pos[-1] == 0.0: #z == 0, ignore
                        continue
                    axes['mocap'].scatter(pos[1], pos[0], marker=',', color='k') # to rotate, longer side to be y axis


            if 'zed_camera_left' in data.keys():
                axes['zed_camera_left'].clear()
                axes['zed_camera_left'].axis('off')
                axes['zed_camera_left'].set_title("ZED Left Image") # code = data['zed_camera_left'][:]
                img = data['zed_camera_left']['img'].data.cpu().squeeze()
                mean = data['zed_camera_left']['img_metas'].data['img_norm_cfg']['mean']
                std = data['zed_camera_left']['img_metas'].data['img_norm_cfg']['std']
                img = img.permute(1, 2, 0).numpy()
                img = (img * std) - mean
                img = img.astype(np.uint8)
                axes['zed_camera_left'].imshow(img)
            
            if 'zed_camera_depth' in data.keys():
                axes['zed_camera_depth'].clear()
                axes['zed_camera_depth'].axis('off')
                axes['zed_camera_depth'].set_title("ZED Depth Map")
                dmap = data['zed_camera_depth']['img'].data[0].cpu().squeeze()
                axes['zed_camera_depth'].imshow(dmap, cmap='turbo')#vmin=0, vmax=10000)

            if 'realsense_camera_img' in data.keys():
                axes['realsense_camera_img'].clear()
                axes['realsense_camera_img'].axis('off')
                axes['realsense_camera_img'].set_title("Realsense Camera Image") # code = data['zed_camera_left'][:]
                img = data['realsense_camera_img']['img'].data[0].cpu().squeeze()
                mean = data['realsense_camera_img']['img_metas'].data[0][0]['img_norm_cfg']['mean']
                std = data['realsense_camera_img']['img_metas'].data[0][0]['img_norm_cfg']['std']
                img = img.permute(1, 2, 0).numpy()
                img = (img * std) - mean
                img = img.astype(np.uint8)
                axes['realsense_camera_img'].imshow(img)

            if 'realsense_camera_depth' in data.keys():
                axes['realsense_camera_depth'].clear()
                axes['realsense_camera_depth'].axis('off')
                axes['realsense_camera_depth'].set_title("Realsense Camera Image") # code = data['zed_camera_left'][:]
                depth = data['realsense_camera_depth']['img'].data[0].cpu().squeeze()
                mean = data['realsense_camera_depth']['img_metas'].data[0][0]['img_norm_cfg']['mean']
                std = data['realsense_camera_depth']['img_metas'].data[0][0]['img_norm_cfg']['std']
                depth = depth.permute(1, 2, 0).numpy()
                depth = (depth * std) - mean
                depth = depth.astype(np.uint8)
                axes['realsense_camera_depth'].imshow(depth)
            
            if 'range_doppler' in data.keys():
                axes['range_doppler'].clear()
                axes['range_doppler'].axis('off')
                axes['range_doppler'].set_title("Range Doppler")
                img = data['range_doppler']['img'].data[0].cpu().squeeze().numpy()
                axes['range_doppler'].imshow(img, cmap='turbo', aspect='auto')

            if 'azimuth_static' in data.keys():
                axes['azimuth_static'].clear()
                axes['azimuth_static'].axis('off')
                axes['azimuth_static'].set_title("Azimuth Static")
                img = data['azimuth_static']['img'].data[0].cpu().squeeze().numpy()
                axes['azimuth_static'].imshow(img, cmap='turbo', aspect='auto')

            if 'mic_waveform' in data.keys():
                axes['mic_waveform'].clear()
                axes['mic_waveform'].axis('off')
                axes['mic_waveform'].set_title("Audio Spectrogram")
                img = data['mic_waveform']['img'].data[0].cpu().squeeze().numpy()
                C, H, W = img.shape
                img = img.reshape(C*H, W)
                axes['mic_waveform'].imshow(img, cmap='turbo', aspect='auto')

            if save_frame:
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                data = cv2.resize(data, dsize=size)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                vid.write(data) 

        vid.release()
        return {'mse': mse}
        
