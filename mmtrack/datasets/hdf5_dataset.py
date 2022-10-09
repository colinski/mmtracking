from abc import ABCMeta, abstractmethod
import numpy as np
from mmdet.datasets.pipelines import Compose
from torch.utils.data import Dataset
from mmtrack.datasets import DATASETS
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

def init_fig(valid_keys, num_rows=4, colspan=2):
    assert ('mocap', 'mocap') in valid_keys
    num_plots = len(valid_keys)
    num_mods = num_plots - 1
    num_cols = int(np.ceil(num_mods / num_rows)) + colspan
    fig = plt.figure(figsize=(num_cols*5, num_rows*5))
    # fig = plt.figure(figsize=(16, 9))
    axes = {}
    axes[('mocap', 'mocap')] = plt.subplot2grid((num_rows, num_cols), (0, 0), rowspan=num_rows, colspan=colspan)
    
    valid_keys = [vk for vk in valid_keys if vk != ('mocap', 'mocap')]
    row, col = 0, colspan
    for i, key in enumerate(valid_keys):
        axes[key] = plt.subplot2grid((num_rows, num_cols), (row, col))
        row += 1
        if row  == num_rows:
            row = 0
            col += 1

    # fig.suptitle('Title', fontsize=11)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    return fig, axes

def convert2dict(f, keys):
    data = {}
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


def load_chunk(fname, start_time, end_time):
    with h5py.File(fname, 'r') as f:
        keys = list(f.keys())
        keys = np.array(keys).astype(int)
        diffs = (keys - start_time)**2
        start_idx = np.argmin(diffs)
        diffs = (keys - end_time)**2
        end_idx = np.argmin(diffs)
        keys = keys[start_idx:end_idx]
        keys = list(keys.astype(str))
        data = convert2dict(f, keys)
    return data


@DATASETS.register_module()
class HDF5Dataset(Dataset, metaclass=ABCMeta):
    CLASSES = None
    def __init__(self,
                 hdf5_fnames=[],
                 fps=20,
                 valid_keys=['mocap', 'zed_camera_left', 'zed_camera_depth'],
                 start_time=1656096536271,
                 end_time=1656096626261,
                 pipelines={},
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
        
        # rank, ws = get_dist_info()
        self.fnames = hdf5_fnames
        self.fps = fps
        self.vid_path = vid_path
        self.limit_axis = limit_axis
        self.draw_cov = draw_cov
        self.num_future_frames = num_future_frames
        self.num_past_frames = num_past_frames
        
        data = {}
        for fname in hdf5_fnames:
            chunk = load_chunk(fname, start_time, end_time)
            for ms, val in chunk.items():
                if ms in data.keys():
                    data[ms].update(val)
                else:
                    data[ms] = val


        self.buffers = self.fill_buffers(data)
        if remove_first_frame:
            self.buffers = self.buffers[1:]
        if max_len is not None:
            self.buffers = self.buffers[0:max_len]
        
        self.active_keys = sorted(self.buffers[-1].keys())

        self.pipelines = {}
        for mod, cfg in pipelines.items():
            self.pipelines[mod] = Compose(cfg)

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
        # buff = self.init_buffer()
        buff = {}
        factor = 100 // self.fps
        num_frames = 0
        keys = sorted(list(all_data.keys()))
        for time in tqdm(keys):
            save_frame = False
            data = all_data[time]
            for key in data.keys():
                if key == 'mocap':
                    mocap_data = json.loads(data['mocap'])
                    gt_pos = torch.tensor([d['normalized_position'] for d in mocap_data])
                    gt_labels = torch.tensor([self.class2idx[d['type']] for d in mocap_data])
                    gt_ids = torch.tensor([d['id'] for d in mocap_data])
                    is_node = gt_labels == 0
                    final_mask = ~is_node
                    z_is_zero = gt_pos[:, -1] == 0.0
                    final_mask = final_mask & ~z_is_zero
                    if len(gt_pos[final_mask]) == 0:
                        continue
                    buff[('mocap', 'mocap')] = {
                        'gt_positions': gt_pos,
                        'gt_labels': gt_labels.long(),
                        'gt_ids': gt_ids.long()
                    }
                    num_frames += 1
                    save_frame = True

                if 'node' in key:
                    for k, v in data[key].items():
                        if k in self.valid_keys:
                            buff[(k, key)] = v
            
            if save_frame and num_frames % factor == 0:
                new_buff = copy.deepcopy(buff)
                buffers.append(new_buff)
        return buffers

    
    def apply_pipelines(self, buff):
        #new_buff = {'missing': buff['missing']}
        new_buff = {}
        for key, val in buff.items():
            mod, node = key
            if mod == 'mocap':
                new_buff[key] = val
            else:
                new_buff[key] = self.pipelines[mod](val)
        return new_buff
    
    def __getitem__(self, ind):
        buff = self.buffers[ind]
        new_buff = self.apply_pipelines(buff)
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
            buff = self.apply_pipelines(buff)
            buff['ind'] = idx
            new_buff['past_frames'].append(buff)

        new_buff['future_frames'] = []
        for idx in future_idx:
            buff = self.buffers[idx]
            buff = self.apply_pipelines(buff)
            buff['ind'] = idx
            new_buff['future_frames'].append(buff)
        return new_buff

    def evaluate(self, outputs, **eval_kwargs):
        mse = 0
        fname = f'{self.vid_path}/latest_vid.mp4'
        fig, axes = init_fig(self.active_keys, len(self.fnames))
        size = (fig.get_figwidth()*100, fig.get_figheight()*100)
        # size = (fig.get_figheight()*100, fig.get_figwidth()*100)
        size = tuple([int(s) for s in size])
        vid = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, size)

        colors = ['red', 'blue', 'green', 'yellow', 'black']
        markers = []
        colors = []
        for i in range(100):
            markers.append(',')
            markers.append('o')
            colors.extend(['green', 'red', 'black', 'yellow'])

        for i in trange(len(self)):
            data = self[i]
            save_frame = False
            for key, val in data.items():
                try:
                    mod, node = key
                except:
                    continue
                if mod == 'mocap':
                    save_frame = True
                    axes[key].clear()
                    if self.limit_axis:
                        axes[key].set_xlim(0,1)
                        axes[key].set_ylim(0,1)
                     
                    means = outputs['pred_position_mean'][i][0]
                    covs = outputs['pred_position_cov'][i][0]
                    ids = outputs['track_ids'][i][0].astype(int)
                    for j in range(len(means)):
                        mean = means[j]
                        cov = covs[j]
                        ID = ids[j]
                        axes[key].scatter(mean[1], mean[0], color='blue', marker=f'${ID}$', lw=1, s=20*4**1)
                        if self.draw_cov:
                            ellipse = Ellipse(xy=(mean[1], mean[0]), width=cov[1]*1, height=cov[0]*1, 
                                                        edgecolor='blue', fc='None', lw=1, linestyle='--')
                            axes[key].add_patch(ellipse)
                    
                    gt_pos = val['gt_positions']
                    gt_ids = val['gt_ids']
                    gt_labels = val['gt_labels']
                    for j in range(len(gt_pos)):
                        pos = gt_pos[j]
                        ID = gt_ids[j]
                        if gt_labels[j] == 0:
                            marker = f'${ID + 1}$'
                            color = 'black'
                        else:
                            marker = markers[ID]
                            color = colors[ID]
                        # if pos[-1] == 0.0: #z == 0, ignore
                            # continue
                        axes[key].scatter(pos[1], pos[0], marker=marker, color=color) # to rotate, longer side to be y axis


                if mod in ['zed_camera_left', 'realsense_camera_img', 'realsense_camera_depth']:
                    axes[key].clear()
                    axes[key].axis('off')
                    axes[key].set_title(key) # code = data['zed_camera_left'][:]
                    img = data[key]['img'].data.cpu().squeeze()
                    mean = data[key]['img_metas'].data['img_norm_cfg']['mean']
                    std = data[key]['img_metas'].data['img_norm_cfg']['std']
                    img = img.permute(1, 2, 0).numpy()
                    img = (img * std) - mean
                    img = img.astype(np.uint8)
                    axes[key].imshow(img)
                 
                if mod == 'zed_camera_depth':
                    axes[key].clear()
                    axes[key].axis('off')
                    axes[key].set_title(key)
                    dmap = data[key]['img'].data[0].cpu().squeeze()
                    axes[key].imshow(dmap, cmap='turbo')#vmin=0, vmax=10000)

                # if 'realsense_camera_img' in data.keys():
                    # axes['realsense_camera_img'].clear()
                    # axes['realsense_camera_img'].axis('off')
                    # axes['realsense_camera_img'].set_title("Realsense Camera Image") # code = data['zed_camera_left'][:]
                    # img = data['realsense_camera_img']['img'].data[0].cpu().squeeze()
                    # mean = data['realsense_camera_img']['img_metas'].data[0][0]['img_norm_cfg']['mean']
                    # std = data['realsense_camera_img']['img_metas'].data[0][0]['img_norm_cfg']['std']
                    # img = img.permute(1, 2, 0).numpy()
                    # img = (img * std) - mean
                    # img = img.astype(np.uint8)
                    # axes['realsense_camera_img'].imshow(img)

                # if 'realsense_camera_depth' in data.keys():
                    # axes['realsense_camera_depth'].clear()
                    # axes['realsense_camera_depth'].axis('off')
                    # axes['realsense_camera_depth'].set_title("Realsense Camera Image") # code = data['zed_camera_left'][:]
                    # depth = data['realsense_camera_depth']['img'].data[0].cpu().squeeze()
                    # mean = data['realsense_camera_depth']['img_metas'].data[0][0]['img_norm_cfg']['mean'] 
                    # std = data['realsense_camera_depth']['img_metas'].data[0][0]['img_norm_cfg']['std']
                    # depth = depth.permute(1, 2, 0).numpy()
                    # depth = (depth * std) - mean
                    # depth = depth.astype(np.uint8)
                    # axes['realsense_camera_depth'].imshow(depth)
                
                if mod == 'range_doppler':
                    axes[key].clear()
                    axes[key].axis('off')
                    axes[key].set_title(key)
                    img = data[key]['img'].data[0].cpu().squeeze().numpy()
                    axes[key].imshow(img, cmap='turbo', aspect='auto')

                if mod == 'azimuth_static':
                    axes[key].clear()
                    axes[key].axis('off')
                    axes[key].set_title(key)
                    img = data[key]['img'].data[0].cpu().squeeze().numpy()
                    axes[key].imshow(img, cmap='turbo', aspect='auto')

                if mod == 'mic_waveform':
                    axes[key].clear()
                    axes[key].set_title(key)
                    axes[key].set_ylim(-1,1)
                    img = data[key]['img'].data[0].cpu().squeeze().numpy()
                    max_val = img[0].max()
                    min_val = img[0].min()
                    axes[key].plot(img[0], color='black')
                    # axes[key].axhline(max_val, color='black')
                    # axes[key].axhline(min_val, color='black')

            if save_frame:
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                data = cv2.resize(data, dsize=size)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                vid.write(data) 

        vid.release()
        return {'mse': mse}
        
