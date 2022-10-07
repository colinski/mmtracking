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
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import copy
from mmcv.runner import get_dist_info
from matplotlib.patches import Ellipse

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
    fig.suptitle('Title', fontsize=11)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    return fig, axes

def read_hdf5(f, keys):
    data = {}
    #for ms in tqdm(f.keys()):
    for ms in tqdm(keys):
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
                 start_times=[1656096536271, 1656096636977],
                 end_times=[1656096626261, 1656096726967],
                 img_pipeline=[],
                 depth_pipeline=[],
                 azimuth_pipeline=[],
                 range_pipeline=[],
                 audio_pipeline=[],
                 num_past_frames=0,
                 num_future_frames=0,
                 test_mode=False,
                 remove_first_frame=False,
                 max_len=None,
                 limit_axis=True,
                 draw_cov=True,
                 vid_path='/tmp',
                 **kwargs):

        self.MODALITIES = ['zed_camera_left', 'zed_camera_right', 'zed_camera_depth',
                           'realsense_camera_img', 'realsense_camera_depth',
                           'range_doppler', 'azimuth_static', 'mic_waveform']
        self.valid_keys = valid_keys
        self.class2idx = {'truck': 1, 'node': 0}
        
        rank, ws = get_dist_info()

        self.fname = hdf5_fname
        self.fps = fps
        self.vid_path = vid_path
        self.limit_axis = limit_axis
        self.draw_cov = draw_cov
        self.num_future_frames = num_future_frames
        self.num_past_frames = num_past_frames

        with h5py.File(self.fname, 'r') as f:
            keys = list(f.keys())
            keys = np.array(keys).astype(int)

            start_time = start_times[rank]
            end_time = end_times[rank]
            
            #find closest keys to start and end times
            diffs = (keys - start_time)**2
            start_idx = np.argmin(diffs)
            diffs = (keys - end_time)**2
            end_idx = np.argmin(diffs)

            keys = keys[start_idx:end_idx]
            keys = list(keys.astype(str))
            data = read_hdf5(f, keys)
            self.buffers = self.fill_buffers(data)
            if remove_first_frame:
                self.buffers = self.buffers[1:]
            if max_len is not None:
                self.buffers = self.buffers[0:max_len]

        self.img_pipeline = Compose(img_pipeline)
        self.depth_pipeline = Compose(depth_pipeline)
        self.range_pipeline = Compose(range_pipeline)
        self.azimuth_pipeline = Compose(azimuth_pipeline)
        self.audio_pipeline = Compose(audio_pipeline)
        self.test_mode = test_mode
        self.flag = np.zeros(len(self), dtype=np.uint8) #ones?
    

    def __len__(self):
        return len(self.buffers)
    
    def init_buffer(self):
        buff = {}
        buff['zed_camera_left'] = np.zeros((10, 10, 3)).astype(np.uint8) #will be resized
        buff['zed_camera_left'] = cv2.imencode('.jpg', buff['zed_camera_left'])[1] #save compressed
        buff['zed_camera_depth'] = np.zeros((10, 10, 1)).astype(np.int16) #will be resized
        buff['range_doppler'] = np.zeros((10, 10))
        buff = {k: v for k, v in buff.items() if k in self.valid_keys}
        buff['missing'] = {}
        for mod in self.MODALITIES:
            buff['missing'][mod] = True
        return buff

    def fill_buffers(self, all_data):
        buffers = []
        buff = self.init_buffer()
        factor = 100 // self.fps
        num_frames = 0
        for time in tqdm(all_data.keys()):
            save_frame = False
            data = all_data[time]
            if 'mocap' in data.keys():
                mocap_data = json.loads(data['mocap'])
                gt_pos = torch.tensor([d['normalized_position'] for d in mocap_data])
                gt_labels = torch.tensor([self.class2idx[d['type']] for d in mocap_data])

                is_node = gt_labels == 0
                final_mask = ~is_node
                z_is_zero = gt_pos[:, -1] == 0.0
                final_mask = final_mask & ~z_is_zero
                gt_pos = gt_pos[final_mask]
                if len(gt_pos) == 0:
                    continue

                buff['mocap'] = data['mocap']
                num_frames += 1
                save_frame = True
            if 'node_1' in data.keys():
                data = data['node_1']

            for key, val in data.items():
                if key in self.valid_keys:
                    buff[key] = val
                    buff['missing'][key] = False
            
            if save_frame and num_frames % factor == 0:
                new_buff = copy.deepcopy(buff)
                buffers.append(new_buff)
                # buff = self.init_buffer()

        return buffers

    
    def parse_buffer(self, buff):
        new_buff = {'missing': buff['missing']}
        for key, val in buff.items():
            if key == 'mocap':
                mocap_data = json.loads(val)
                positions = torch.tensor([d['normalized_position'] for d in mocap_data])
                labels = torch.tensor([self.class2idx[d['type']] for d in mocap_data])
                ids = torch.tensor([d['id'] for d in mocap_data])
                z_is_zero = positions[:, -1] == 0.0
                is_node = labels == self.class2idx['node']
                final_mask = ~z_is_zero | is_node
                final_mask = positions[:, -1] >= 0.0
                new_buff[key] = {
                    'gt_positions': positions[final_mask],
                    'gt_labels': labels[final_mask].long(),
                    'gt_ids': ids[final_mask].long()
                }

            if key == 'zed_camera_left': #and not buff['missing'][key]:
                img = cv2.imdecode(val, 1)
                new_buff[key] = self.img_pipeline(img)

            if key == 'zed_camera_depth':
                new_buff[key] = self.depth_pipeline(val)

            if key == 'range_doppler':
                new_buff[key] = self.range_pipeline(val.T)
        
        return new_buff
    
    def __getitem__(self, ind):
        buff = self.buffers[ind]
        new_buff = self.parse_buffer(buff)
        new_buff['ind'] = ind
        

        idx_set = torch.arange(len(self))
        start_idx = max(0, ind - self.num_past_frames)
        past_idx = idx_set[start_idx:ind]

        if len(past_idx) < self.num_past_frames:
            zeros = torch.zeros(self.num_past_frames - len(past_idx)).long()
            past_idx = torch.cat([zeros, past_idx])

        end_idx = min(ind + self.num_past_frames + 1, len(self))
        future_idx = idx_set[ind + 1:end_idx]

        if len(future_idx) < self.num_future_frames:
            zeros = torch.zeros(self.num_future_frames- len(future_idx)).long()
            future_idx = torch.cat([future_idx, zeros + len(self) - 1])
        
        new_buff['past_frames'] = []
        for idx in past_idx:
            buff = self.buffers[idx]
            buff = self.parse_buffer(buff)
            buff['ind'] = idx
            new_buff['past_frames'].append(buff)

        new_buff['future_frames'] = []
        for idx in future_idx:
            buff = self.buffers[idx]
            buff = self.parse_buffer(buff)
            buff['ind'] = idx
            new_buff['future_frames'].append(buff)

        return new_buff

    def evaluate(self, outputs, **eval_kwargs):
        mse = 0
        size = (1600, 900)
        fname = f'{self.vid_path}/latest_vid.mp4'
        vid = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, size)
        fig, axes = init_fig()

        colors = ['red', 'blue', 'green', 'yellow', 'black']
        markers = []
        colors = []
        for i in range(100):
            markers.append(',')
            markers.append('o')
            #colors.extend(['red', 'blue', 'green', 'yellow', 'black'])
            colors.extend(['green', 'red', 'black', 'yellow'])
        # markers = [',', 'o', ',', 'o', ',', 'o', ',', 'o']

        for i in trange(len(self)):
            data = self[i]

            save_frame = False
            if 'mocap' in data.keys():
                save_frame = True
                axes['mocap'].clear()
                if self.limit_axis:
                    axes['mocap'].set_xlim(0,1)
                    axes['mocap'].set_ylim(0,1)
                 
                means = outputs['pred_position_mean'][i][0]
                covs = outputs['pred_position_cov'][i][0]
                ids = outputs['track_ids'][i][0].astype(int)
                print(means, covs, ids)
                for j in range(len(means)):
                    mean = means[j]
                    cov = covs[j]
                    ID = ids[j]
                    if len(mean) != 3 or len(cov) != 3:
                        import ipdb; ipdb.set_trace() # noqa
                    axes['mocap'].scatter(mean[1], mean[0], color='blue', marker=f'${ID}$', lw=1, s=20*4**1)
                    # axes['mocap'].annotate(str(ID), (mean[1], mean[0]))
                    if self.draw_cov:
                        ellipse = Ellipse(xy=(mean[1], mean[0]), width=cov[1]*1, height=cov[0]*1, 
                                                    edgecolor='blue', fc='None', lw=1, linestyle='--')
                        axes['mocap'].add_patch(ellipse)
                
                # for pos in outputs['pred_position_mean'][i]:
                    # if pos[-1] == 0.0: #z == 0, ignore
                        # continue
                    # alpha = 0.5 if len(pos) > 1 else 1
                    # axes['mocap'].scatter(pos[:, 1], pos[:, 0], alpha=alpha) # to rotate, longer side to be y axis
                gt_pos = data['mocap']['gt_positions']
                gt_ids = data['mocap']['gt_ids']
                gt_labels = data['mocap']['gt_labels']
                # for pos in data['mocap']['gt_positions']:
                for j in range(len(gt_pos)):
                    pos = gt_pos[j]
                    ID = gt_ids[j]
                    if gt_labels[j] == 0:
                        marker = ','
                        color = 'black'
                    else:
                        marker = markers[ID]
                        color = colors[ID]
                    # if pos[-1] == 0.0: #z == 0, ignore
                        # continue
                    axes['mocap'].scatter(pos[1], pos[0], marker=marker, color=color) # to rotate, longer side to be y axis


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
        
