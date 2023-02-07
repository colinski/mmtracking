from abc import ABCMeta, abstractmethod
import os
import glob
import pickle
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
from matplotlib.patches import Ellipse, Rectangle
from collections import defaultdict
import torch.distributions as D
from scipy.spatial import distance
from trackeval.metrics import CLEAR, HOTA, Identity
import matplotlib
from .viz import init_fig, gen_rectange, gen_ellipse, rot2angle, points_in_rec
from mmtrack.datasets import build_dataset
import torch.nn.functional as F

font = {#'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

@DATASETS.register_module()
class HDF5Dataset(Dataset, metaclass=ABCMeta):
    CLASSES = None
    def __init__(self,
                 cacher_cfg=None,
                 pipelines={},
                 num_past_frames=0,
                 num_future_frames=0,
                 test_mode=False,
                 limit_axis=True,
                 draw_cov=True,
                 truck_w=30,
                 truck_h=15,
                 include_z=False,
                 **kwargs):
        self.truck_w = truck_w
        self.truck_h = truck_h
        self.cacher = build_dataset(cacher_cfg)
        self.fnames, self.active_keys = self.cacher.cache()
        self.max_len = 1
        self.fps = self.cacher.fps
        self.limit_axis = limit_axis
        self.draw_cov = draw_cov
        self.num_future_frames = num_future_frames
        self.num_past_frames = num_past_frames
        self.node_pos = None
        self.node_ids = None
        self.colors = ['red', 'green', 'orange', 'black', 'yellow', 'blue']
        
        self.pipelines = {}
        for mod, cfg in pipelines.items():
            self.pipelines[mod] = Compose(cfg)

        self.test_mode = test_mode
        self.flag = np.zeros(len(self), dtype=np.uint8) #ones?
    
    def __len__(self):
        return len(self.fnames)
    
    def apply_pipelines(self, buff):
        new_buff = {}
        for key, val in buff.items():
            mod, node = key
            if mod == 'mocap':
                new_buff[key] = val
            else:
                new_buff[key] = self.pipelines[mod](val)
        return new_buff

    def read_buff(self, ind):
        with open(self.fnames[ind], 'rb') as f:
            buff = pickle.load(f)
        return buff
    
    def __getitem__(self, ind, apply_pipelines=True):
        new_buff = self.read_buff(ind)
        if apply_pipelines:
            new_buff = self.apply_pipelines(new_buff)
        
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
        for idx in past_idx:
            buff = self.read_buff(idx)
            buff = self.apply_pipelines(buff)
            buffs.append(buff)
        buffs.append(new_buff)

        for idx in future_idx:
            buff = self.read_buff(idx)
            buff = self.apply_pipelines(buff)
            buffs.append(buff)
        return buffs
    
    def eval_mot(self, outputs, **eval_kwargs):
        all_gt_pos, all_gt_labels, all_gt_ids, all_gt_rot, all_gt_grids = [], [], [], [], []
        for i in trange(len(self)):
            data = self[i][-1] #get last frame, eval shouldnt have future
            for key, val in data.items():
                mod, node = key
                if mod == 'mocap':
                    all_gt_pos.append(val['gt_positions'])
                    all_gt_ids.append(val['gt_ids'])
                    # all_gt_labels.append(val['gt_labels'])
                    all_gt_rot.append(val['gt_rot'])
                    all_gt_grids.append(val['gt_grids'])

        all_gt_pos = torch.stack(all_gt_pos) #num_frames x num_objs x 3
        # all_gt_labels = torch.stack(all_gt_labels)
        all_gt_ids = torch.stack(all_gt_ids)
        all_gt_rot = torch.stack(all_gt_rot)
        all_gt_grids = torch.stack(all_gt_grids)

        res = {}
        res['num_gt_dets'] = all_gt_ids.shape[0] * all_gt_ids.shape[1]
        res['num_gt_ids'] = len(torch.unique(all_gt_ids))
        res['num_tracker_ids'] = len(torch.unique(all_gt_ids))
        res['num_timesteps'] = len(all_gt_ids)
        res['tracker_ids'] = []
        res['gt_ids'] = all_gt_ids.numpy().astype(int)
        res['similarity_scores'] = []
        res['grid_scores'] = []
        res['num_tracker_dets'] = 0

        from mmtrack.models.mocap.decoderv3 import calc_grid_loss 
        all_probs, all_dists = [], []
        for i in range(res['num_timesteps']):
            pred_means = outputs['track_means'][i]
            pred_covs = outputs['track_covs'][i]
            pred_ids = outputs['track_ids'][i]
            res['num_tracker_dets'] += len(pred_ids)
            res['tracker_ids'].append(pred_ids.numpy().astype(int))
            gt_pos = all_gt_pos[i]
            gt_rot = all_gt_rot[i]
            gt_grid = all_gt_grids[i]
            
            dist = D.MultivariateNormal(pred_means.unsqueeze(0), pred_covs.unsqueeze(0))
            loss_vals = calc_grid_loss(dist, gt_grid)
            
            dists, probs = [], []
            scores = []
            grid_scores = []
            for j in range(len(pred_means)):
                # dist = torch.norm(pred_mean[j][0:2] - gt_pos[:,0:2], dim=1)
                # dists.append(dist)

                dist = D.MultivariateNormal(pred_means[j], pred_covs[j])
                # dist = D.Independent(dist, 1) #Nq independent Gaussians
                # samples = dist.sample([10000])
                
                num_gt = len(gt_pos)
                for k in range(num_gt):
                    grid = gt_grid[k]
                    log_probs = dist.log_prob(grid) #*1.5
                    logsum = torch.logsumexp(log_probs.flatten(), dim=0)
                    grid_scores.append(logsum)
                    # angle = rot2angle(gt_rot[k], return_rads=False)
                    # rec, _ = gen_rectange(gt_pos[k], angle, w=self.truck_w, h=self.truck_h)
                    # mask = points_in_rec(samples, rec)
                    #scores.append(np.mean(mask))
                    scores.append(logsum.exp())

            if len(scores) == 0:
                scores = torch.empty(len(gt_pos), 0).numpy()
            else:
                scores = torch.tensor(scores).reshape(len(pred_means), -1)
                grid_scores = torch.tensor(grid_scores).reshape(len(pred_means), -1)
                grid_scores = grid_scores.numpy().T
                scores = scores.numpy().T

            # if len(dists) != 0:
                # dists = torch.stack(dists) #num_preds x num_gt_tracks
                # dists = dists.numpy().T
                # dists[dists > self.max_len] = self.max_len
                # dists = 1 - (dists / self.max_len)
            # else:
                # dists = torch.empty(len(gt_pos), 0).numpy()
            res['similarity_scores'].append(scores)
            res['grid_scores'].append(grid_scores)
            # all_dists.append(dists)
            # all_probs.append(probs)
        
        scores = np.stack(res['similarity_scores'])
        grid_scores = np.stack(res['grid_scores'])
        logdir = eval_kwargs['logdir']
        fname = f'{logdir}/res.json'
        #met=CLEAR({'THRESHOLD': 1-(0.3/self.max_len)}) 
        met = CLEAR({'THRESHOLD': 0.5})
        out = met.eval_sequence(res)
        out = {k : float(v) for k,v in out.items()}
        
        hmet = HOTA()
        hout = hmet.eval_sequence(res)
        means = {k + '_mean' : v.mean() for k, v in hout.items()}
        hout = {k: v.tolist() for k,v in hout.items()}
        out.update(hout)
        out.update(means)

        imet = Identity({'THRESHOLD': 0.5})
        iout = imet.eval_sequence(res)
        iout = {k : float(v) for k,v in iout.items()}
        out.update(iout)

        print(out)
        with open(fname, 'w') as f:
            json.dump(out, f)
        return out

    def evaluate(self, outputs, **eval_kwargs):
        metrics = eval_kwargs['metric']
        res = {}
        if 'track' in metrics:
            res = self.eval_mot(outputs, **eval_kwargs)
            print(res)
        if 'vid' in metrics:
            self.write_video(outputs, **eval_kwargs)
        return res

    def write_video(self, outputs, **eval_kwargs): 
        logdir = eval_kwargs['logdir']
        video_length = len(self)
        if 'video_length' in eval_kwargs.keys():
            video_length = eval_kwargs['video_length']
        fname = f'{logdir}/latest_vid.mp4'
        fig, axes = init_fig(self.active_keys)
        size = (fig.get_figwidth()*50, fig.get_figheight()*50)
        size = tuple([int(s) for s in size])
        vid = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, size)

        markers, colors = [], []
        for i in range(100):
            markers.append(',')
            markers.append('o')
            colors.extend(['green', 'red', 'black', 'yellow'])

        frame_count = 0
        
        id2dist = defaultdict(list)
        for i in trange(video_length):
            data = self[i][-1] #get last frame, eval shouldnt have future
            data = self.__getitem__(i, apply_pipelines=False)[-1]
            save_frame = False
            for key, val in data.items():
                mod, node = key
                if mod == 'mocap':
                    save_frame = True
                    axes[key].clear()
                    axes[key].grid('on', linewidth=3)
                    # axes[key].set_facecolor('gray')
                    if self.limit_axis:
                        axes[key].set_xlim(0,700)
                        axes[key].set_ylim(0,500)
                        axes[key].set_aspect('equal')
                    
                    num_gt = len(val['gt_positions'])
                    for j in range(num_gt):
                        pos = val['gt_positions'][j]
                        if pos[0] == -1:
                            continue
                        rot = val['gt_rot'][j]
                        ID = val['gt_ids'][j]
                        grid = val['gt_grids'][j]
                        marker = markers[ID]
                        color = colors[ID]
                        
                        axes[key].scatter(pos[0], pos[1], marker=markers[ID], color=color) 
                        
                        angle = rot2angle(rot, return_rads=False)
                        rec, _ = gen_rectange(pos, angle, w=self.truck_w, h=self.truck_h, color=color)
                        axes[key].add_patch(rec)

                        r=self.truck_w/2
                        axes[key].arrow(pos[0], pos[1], r*rot[0], r*rot[1], head_width=0.05*100, head_length=0.05*100, fc=color, ec=color)
                            
                    
                    if len(outputs['det_means']) > 0:
                        pred_means = outputs['det_means'][i]
                        for j in range(len(pred_means)):
                            mean = pred_means[j]
                            axes[key].scatter(mean[0], mean[1], color='black', marker=f'+', lw=1, s=20*4**2)
                    
                    # if 'track_means' in outputs.keys() and len(outputs['track_means'][i]) > 0:
                    pred_means = outputs['track_means'][i] 
                    pred_covs = outputs['track_covs'][i]
                    #pred_rots = outputs['track_rot'][i]
                    ids = outputs['track_ids'][i].to(int)
                    slot_ids = outputs['slot_ids'][i].to(int)
                    print(pred_means, pred_covs)
                    for j in range(len(pred_means)):
                        #rot = pred_rots[j]
                        #angle = torch.arctan(rot[0]/rot[1]) * 360
                        mean = pred_means[j]
                        color = self.colors[j % len(self.colors)]
                        
                        #rec, _ = gen_rectange(mean, angle, w=self.truck_w, h=self.truck_h, color=color)
                        #axes[key].add_patch(rec)


                        axes[key].scatter(mean[0], mean[1], color=color, marker=f'+', lw=1, s=20*4**2)
                        cov = pred_covs[j]
                        ID = ids[j]
                        sID = slot_ids[j]
                        #axes[key].text(mean[0], mean[1], s=f'T${ID}$S{sID}', fontdict={'color': color})
                        axes[key].text(mean[0], mean[1], s=f'{ID}', fontdict={'color': color})
                        if self.draw_cov:
                            ellipse = gen_ellipse(mean, cov, edgecolor=color, fc='None', lw=2, linestyle='--')
                            axes[key].add_patch(ellipse)
                    
                    

                if mod in ['zed_camera_left', 'realsense_camera_img', 'realsense_camera_depth']:
                    # node_num = int(node[-1])
                    # A = outputs['attn_weights'][i]
                    # A = A.permute(1,0,2) 
                    # nO, nH, L = A.shape
                    # A = A.reshape(nO, nH, 4, 35)
                    # head_dists = A.sum(dim=-1)[..., node_num-1]
                    # head_dists = F.interpolate(head_dists.unsqueeze(0).unsqueeze(0), scale_factor=60)[0][0]
                    
                    # z = torch.zeros_like(head_dists)
                    # head_dists = torch.stack([head_dists,z,z], dim=-1)

                    # head_dists = (head_dists * 255).numpy()
                    # head_dists = (head_dists - 255) * -1
                    # head_dists = head_dists.astype(np.uint8)

                    axes[key].clear()
                    axes[key].axis('off')
                    axes[key].set_title(key) # code = data['zed_camera_left'][:]
                    code = data[key]
                    img = cv2.imdecode(code, 1)
                    # img = data[key]['img'].data.cpu().squeeze()
                    # mean = data[key]['img_metas'].data['img_norm_cfg']['mean']
                    # std = data[key]['img_metas'].data['img_norm_cfg']['std']
                    # img = img.permute(1, 2, 0).numpy()
                    # img = (img * std) - mean
                    # img = img.astype(np.uint8)
                    #img = np.concatenate([img, head_dists], axis=0)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[key].imshow(img)

                if 'r50' in mod:
                    axes[key].clear()
                    axes[key].axis('off')
                    axes[key].set_title(key) # code = data['zed_camera_left'][:]
                    feat = data[key]['img'].data#[0].cpu().squeeze()
                    feat = feat.mean(dim=0).cpu()
                    feat[feat > 1] = 1
                    feat = (feat * 255).numpy().astype(np.uint8)
                    feat = np.stack([feat]*3, axis=-1)
                    #axes[key].imshow(feat, cmap='turbo')
                    axes[key].imshow(feat)

                 
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
                    # img = data[key]['img'].data[0].cpu().squeeze().numpy()
                    img = data[key]
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
                # fname = f'{logdir}/frame_{frame_count}.png'
                # cv2.imwrite(fname, data)
                frame_count += 1
                vid.write(data) 

        vid.release()
