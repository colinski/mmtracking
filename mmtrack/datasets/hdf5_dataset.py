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
import mmcv
from mmcv.runner import get_dist_info
from matplotlib.patches import Ellipse
from collections import defaultdict
import torch.distributions as D
from scipy.spatial import distance
from trackeval.metrics import CLEAR

def init_fig(valid_keys, num_cols=4, colspan=1):
    assert ('mocap', 'mocap') in valid_keys

    mods = [vk[0] for vk in valid_keys if vk != ('mocap', 'mocap')]
    num_mods = len(set(mods))
    num_cols = num_mods + 2
    num_rows = num_mods + 2
    
    # num_plots = len(valid_keys)
    # num_mods = num_plots - 1
    #num_cols = int(np.ceil(num_mods / num_rows)) + colspan
    # num_rows = num_mods + 1

    # num_rows, num_cols = 2 + 5, 7 + 2
    # num_cols = 4
    fig = plt.figure(figsize=(num_cols*5, num_rows*5))
    # fig = plt.figure(figsize=(16, 9))
    axes = {}
    #axes[('mocap', 'mocap')] = plt.subplot2grid((num_rows, num_cols), (0, 0), rowspan=num_rows, colspan=colspan)
    axes[('mocap', 'mocap')] = plt.subplot2grid((num_rows, num_cols), (1, 1), rowspan=num_mods, colspan=num_mods)

    #row, col = 0, colspan
    node2row = {'node_2': num_rows-1, 'node_4': 0}
    node2col = {'node_3': 0, 'node_1': num_cols-1}
   
    valid_keys = [vk for vk in valid_keys if vk != ('mocap', 'mocap')]
    for node_num, col_num in node2col.items():
        count = 1
        for i, key in enumerate(valid_keys):
            if key[1] != node_num:
                continue
            axes[key] = plt.subplot2grid((num_rows, num_cols), (count, col_num))
            count += 1

    
    for node_num, row_num in node2row.items():
        count = 1
        for i, key in enumerate(valid_keys):
            if key[1] != node_num:
                continue
            axes[key] = plt.subplot2grid((num_rows, num_cols), (row_num, count))
            count += 1
             
            

    # for i, key in enumerate(valid_keys):
        # mod, node = key
        # if node == 'node_2':
            # col = num_cols
            # axes[key] = plt.subplot2grid((num_rows, num_cols), (row, col))

    # row, col = 1, 0
    # for i, key in enumerate(valid_keys):
        # axes[key] = plt.subplot2grid((num_rows, num_cols), (row, col))
        # row += 1
        # if row  == num_rows:
            # row = 0
            # col += 1

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
                 min_x=-2162.78244, max_x=4157.92774,
                 min_y=-1637.84491, max_y=2930.06133,
                 min_z=0.000000000, max_z=903.616290,
                 normalized_position=False,
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
        self.min_x = min_x
        self.max_x = max_x
        self.len_x = 7000
        self.min_y = min_y
        self.max_y = max_y
        self.len_y = 5000
        self.min_z = min_z
        self.max_z = max_z
        self.len_z = 1000
        self.normalized_position = normalized_position

        #self.max_len = np.sqrt(7**2 + 5**2)
        self.max_len = 1
        
        # rank, ws = get_dist_info()
        self.fnames = hdf5_fnames
        self.fps = fps
        self.vid_path = vid_path
        self.limit_axis = limit_axis
        self.draw_cov = draw_cov
        self.num_future_frames = num_future_frames
        self.num_past_frames = num_past_frames

        self.node_pos = None
        self.node_ids = None
        
        data = {}
        for fname in hdf5_fnames:
            chunk = load_chunk(fname, start_time, end_time)
            for ms, val in chunk.items():
                if ms in data.keys():
                    data[ms].update(val)
                else:
                    data[ms] = val


        self.buffers = self.fill_buffers(data)

        

        
        self.active_keys = sorted(self.buffers[-1].keys())
        
        #ensure that we always have all the modalities
        #some might be missing at the first start
        count = 0
        for i in range(len(self.buffers)):
            missing = False
            for key in self.active_keys:
                if key not in self.buffers[i].keys():
                    missing = True
            if missing:
                count += 1
                continue
            else:
                break
        self.buffers = self.buffers[count:]

        if remove_first_frame:
            self.buffers = self.buffers[1:]
        if max_len is not None:
            self.buffers = self.buffers[0:max_len]


        self.nodes = set([key[1] for key in self.active_keys if 'node' in key[1]])

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
        prev_num_objs = None
        for time in tqdm(keys):
            save_frame = False
            data = all_data[time]
            for key in data.keys():
                if key == 'mocap':
                    mocap_data = json.loads(data['mocap'])
                    if self.normalized_position:
                        gt_pos = torch.tensor([d['normalized_position'] for d in mocap_data])
                    else:
                        gt_pos = torch.tensor([d['position'] for d in mocap_data])
                        gt_pos[..., 0] += np.abs(self.min_x)
                        gt_pos[..., 1] += np.abs(self.min_y)
                        gt_pos[..., 2] += np.abs(self.min_z)
                        gt_pos /= 1000
                    gt_labels = torch.tensor([self.class2idx[d['type']] for d in mocap_data])
                    gt_ids = torch.tensor([d['id'] for d in mocap_data])
                    gt_rot = torch.tensor([d['rotation'] for d in mocap_data])
                    is_node = gt_labels == 0
                    if self.node_pos is None:
                        self.node_pos = gt_pos[is_node]
                        self.node_ids = gt_ids[is_node]
                    
                    gt_pos = gt_pos[~is_node]
                    z_is_zero = gt_pos[:, -1] == 0.0
                    # if torch.any(z_is_zero):
                        # continue
                    final_mask = ~is_node 
                    buff[('mocap', 'mocap')] = {
                        'gt_positions': gt_pos,
                        'gt_labels': gt_labels[final_mask].long(),
                        'gt_ids': gt_ids[final_mask].long() - 4,
                        'gt_rot': gt_rot[final_mask]
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
        # new_buff['ind'] = ind
        
        idx_set = torch.arange(len(self))
        start_idx = max(0, ind - self.num_past_frames)
        past_idx = idx_set[start_idx:ind]

        if len(past_idx) < self.num_past_frames:
            zeros = torch.zeros(self.num_past_frames - len(past_idx)).long()
            past_idx = torch.cat([zeros, past_idx])

        end_idx = min(ind + self.num_future_frames + 1, len(self))
        future_idx = idx_set[ind + 1:end_idx]

        if len(future_idx) < self.num_future_frames:
            zeros = torch.zeros(self.num_future_frames- len(future_idx)).long()
            future_idx = torch.cat([future_idx, zeros + len(self) - 1])
        
        buffs = []
        # new_buff['past_frames'] = []
        for idx in past_idx:
            buff = self.buffers[idx]
            buff = self.apply_pipelines(buff)
            # buff['ind'] = idx
            buffs.append(buff)
            # new_buff['past_frames'].append(buff)
        
        buffs.append(new_buff)

        # new_buff['future_frames'] = []
        for idx in future_idx:
            buff = self.buffers[idx]
            buff = self.apply_pipelines(buff)
            # buff['ind'] = idx
            buffs.append(buff)
            # new_buff['future_frames'].append(buff)
        
        #merge time series into a batch
        # res = mmcv.parallel.collate(buffs)
        return buffs

    def evaluate(self, outputs, **eval_kwargs):
        all_gt_pos, all_gt_labels, all_gt_ids = [], [], []
        for i in trange(len(self)):
            data = self[i][-1] #get last frame, eval shouldnt have future
            for key, val in data.items():
                mod, node = key
                if mod == 'mocap':
                    all_gt_pos.append(val['gt_positions'])
                    all_gt_ids.append(val['gt_ids'])
                    all_gt_labels.append(val['gt_labels'])

        all_gt_pos = torch.stack(all_gt_pos) #num_frames x num_objs x 3
        all_gt_labels = torch.stack(all_gt_labels)
        all_gt_ids = torch.stack(all_gt_ids)

        res = {}
        res['num_gt_dets'] = all_gt_ids.shape[0] * all_gt_ids.shape[1]
        res['num_gt_ids'] = len(torch.unique(all_gt_ids))
        res['num_timesteps'] = len(all_gt_ids)
        res['tracker_ids'] = []
        res['gt_ids'] = all_gt_ids.numpy()
        res['similarity_scores'] = []
        res['num_tracker_dets'] = 0
        
        all_probs, all_dists = [], []
        for i in range(len(all_gt_ids)):
            pred_mean = torch.from_numpy(outputs['pred_position_mean'][i][0])
            pred_cov = torch.from_numpy(outputs['pred_position_cov'][i][0])
            pred_ids = torch.from_numpy(outputs['track_ids'][i][0])
            res['num_tracker_dets'] += len(pred_mean)
            res['tracker_ids'].append(pred_ids.numpy())
            gt_pos = all_gt_pos[i]
            
            dists, probs = [], []
            for j in range(len(pred_mean)):
                dist = torch.norm(pred_mean[j][0:2] - gt_pos[:,0:2], dim=1)
                dists.append(dist)
                # dist = D.Normal(pred_mean[j], pred_cov[j])
                # dist = D.Independent(dist, 1) #Nq independent Gaussians
                # probs.append(dist.log_prob(gt_pos))
                # icov = torch.diag(1/pred_cov[j])
                # mdist = distance.mahalanobis(gt_pos[0], pred_mean[j], icov)
                # mdists.append(mdist)
            # probs = torch.stack(probs) #num_preds x num_gt_tracks
            if len(dists) != 0:
                dists = torch.stack(dists) #num_preds x num_gt_tracks
                dists = dists.numpy().T
                dists[dists > self.max_len] = self.max_len
                dists = 1 - (dists / self.max_len)
            else:
                dists = torch.empty(len(gt_pos), 0).numpy()
            res['similarity_scores'].append(dists)
            all_dists.append(dists)
            # all_probs.append(probs)
        fname = f'{self.vid_path}/res.json'
        met=CLEAR({'THRESHOLD': 1-(0.3/self.max_len)}) 
        out = met.eval_sequence(res)
        out = {k:float(v) for k,v in out.items()}
        with open(fname, 'w') as f:
            json.dump(out, f)
        print(out)

        mse = 0
        fname = f'{self.vid_path}/latest_vid.mp4'
        #fig, axes = init_fig(self.active_keys, len(self.nodes))
        fig, axes = init_fig(self.active_keys)
        size = (fig.get_figwidth()*100, fig.get_figheight()*100)
        # size = (fig.get_figheight()*100, fig.get_figwidth()*100)
        size = tuple([int(s) for s in size])
        vid = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, size)

        markers, colors = [], []
        for i in range(100):
            markers.append(',')
            markers.append('o')
            colors.extend(['green', 'red', 'black', 'yellow'])
        
        id2dist = defaultdict(list)
        for i in trange(len(self)):
            data = self[i][-1] #get last frame, eval shouldnt have future
            save_frame = False
            for key, val in data.items():
                mod, node = key
                if mod == 'mocap':
                    save_frame = True
                    axes[key].clear()
                    axes[key].grid('on')
                    if self.limit_axis:
                        axes[key].set_xlim(0,7)
                        axes[key].set_ylim(0,5)
                        axes[key].set_aspect('equal')
                    
                    gt_pos = val['gt_positions']
                    gt_ids = val['gt_ids']
                    gt_labels = val['gt_labels']
                    gt_rot = val['gt_rot']

                    for j in range(len(self.node_pos)):
                        pos = self.node_pos[j]
                        ID = self.node_ids[j] + 1
                        axes[key].scatter(pos[0], pos[1], marker=f'${ID}$', color='black')

                    for j in range(len(gt_pos)):
                        # plot_gt(gt_pos[j])
                        pos = gt_pos[j]
                        ID = gt_ids[j]
                        rot = gt_rot[j]
                        marker = markers[ID]
                        color = colors[ID]
                        # if pos[-1] == 0.0: #z == 0, ignore
                            # continue

                        axes[key].scatter(pos[0], pos[1], marker=marker, color=color) # to rotate, longer side to be y axis
                        
                        if rot[4] <= 0:
                            rads = np.arcsin(rot[3]) / (2*np.pi)
                        else:
                            rads = np.arcsin(rot[1]) / (2*np.pi)

                        angle = rads * 360
                        
                        # w = 15/100
                        # h = 30/100
                        w = 30/100
                        h = 15/100
                        ellipse = Ellipse(xy=(pos[0], pos[1]), width=w, height=h, angle=angle,
                                                        edgecolor=color, fc='None', lw=1, linestyle='--')
                        
                        axes[key].add_patch(ellipse)

                        r=0.15
                        axes[key].arrow(pos[0], pos[1], r*rot[0], r*rot[1], head_width=0.05, head_length=0.05, fc=color, ec=color)
                            

                    means = outputs['pred_position_mean'][i][0]
                    covs = outputs['pred_position_cov'][i][0]
                    ids = outputs['track_ids'][i][0].astype(int)
                    for j in range(len(means)):
                        mean = means[j]
                        # dists = [np.linalg.norm(mean - pos.numpy()) for pos in gt_pos]
                        # dist = dists[-2]
                        cov = covs[j]
                        ID = ids[j]
                        # id2dist[ID].append(dist)
                        # rot = gt_rot[-2]
                        axes[key].scatter(mean[0], mean[1], color='blue', marker=f'${ID}$', lw=1, s=20*4**1)
                        #axes[key].text(mean[1], mean[0], '%0.2f' % dist)
                        if self.draw_cov:
                            ellipse = Ellipse(xy=(mean[0], mean[1]), width=cov[0]*1, height=cov[1]*1, 
                                                        edgecolor='blue', fc='None', lw=1, linestyle='--')
                            axes[key].add_patch(ellipse)
                    
                    

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

            if save_frame:
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                data = cv2.resize(data, dsize=size)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                vid.write(data) 

        vid.release()
        # dist_mean = {k: np.mean(v) for k, v in id2dist.items()}
        # dist_std = {k: np.std(v) for k, v in id2dist.items()}
        # print(dist_mean, dist_std)
        return {'mse': mse}
