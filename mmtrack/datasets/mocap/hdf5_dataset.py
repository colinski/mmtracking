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
from trackeval.metrics import CLEAR
import matplotlib
from .viz import init_fig, gen_rectange, gen_ellipse, rot2angle, points_in_rec
from mmtrack.datasets import build_dataset
import torch.nn.functional as F

font = {#'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

def convert2dict(f, keys, fname, valid_mods, valid_nodes):
    data = {}
    for ms in tqdm(keys, desc='loading %s' % fname):
        data[ms] = {}
        for k, v in f[ms].items():
            if k == 'mocap': #mocap or node_N
                data[ms]['mocap'] = v[()]
            else: #is node_N
                if k not in valid_nodes:
                    continue
                data[ms][k] = {}
                for k2, v2 in f[ms][k].items():
                    if k2 not in valid_mods:
                        continue
                    if k2 == 'detected_points':
                        data[ms][k][k2] = v2[()]
                    else:
                        data[ms][k][k2] = v2[:]
    return data


def load_chunk(fname, valid_mods, valid_nodes):
    with h5py.File(fname, 'r') as f:
        keys = list(f.keys())
        keys = np.array(keys).astype(int)
        # diffs = (keys - start_time)**2
        # start_idx = np.argmin(diffs)
        # diffs = (keys - end_time)**2
        # end_idx = np.argmin(diffs)
        # keys = keys[start_idx:end_idx]
        keys = list(keys.astype(str))
        data = convert2dict(f, keys, fname, valid_mods, valid_nodes)
    return data


@DATASETS.register_module()
class HDF5Dataset(Dataset, metaclass=ABCMeta):
    CLASSES = None
    def __init__(self,
                 # hdf5_fnames=[],
                 cacher_cfg=None,
                 # fps=20,
                 # valid_mods=['mocap', 'zed_camera_left', 'zed_camera_depth'],
                 # valid_nodes=[1,2,3,4],
                 # start_time=1656096536271,
                 # end_time=1656096626261,
                 # min_x=-2162.78244, max_x=4157.92774,
                 # min_y=-1637.84491, max_y=2930.06133,
                 # min_z=0.000000000, max_z=903.616290,
                 # name='train',
                 # uid=0,
                 # normalized_position=False,
                 pipelines={},
                 num_past_frames=0,
                 num_future_frames=0,
                 test_mode=False,
                 # remove_first_frame=False,
                 # max_len=None,
                 limit_axis=True,
                 draw_cov=True,
                 truck_w=30,
                 truck_h=15,
                 include_z=False,
                 **kwargs):
        # self.valid_mods = valid_mods
        # self.valid_nodes = ['node_%d' % n for n in valid_nodes]
        # self.class2idx = {'truck': 1, 'node': 0}
        # self.min_x = min_x
        # self.max_x = max_x
        # self.len_x = 7000
        # self.min_y = min_y
        # self.max_y = max_y
        # self.len_y = 5000
        # self.min_z = min_z
        # self.max_z = max_z
        # self.len_z = 1000
        # self.normalized_position = normalized_position
        self.truck_w = truck_w
        self.truck_h = truck_h
        # self.include_z = include_z
        # self.name = name
        # self.uid = uid
        self.cacher = build_dataset(cacher_cfg)
        self.fnames, self.active_keys = self.cacher.cache()

        #self.max_len = np.sqrt(7**2 + 5**2)
        self.max_len = 1
        
        # rank, ws = get_dist_info()
        # self.fnames = hdf5_fnames
        self.fps = self.cacher.fps
        self.limit_axis = limit_axis
        self.draw_cov = draw_cov
        self.num_future_frames = num_future_frames
        self.num_past_frames = num_past_frames

        self.node_pos = None
        self.node_ids = None

        self.colors = ['red', 'green', 'orange', 'black', 'yellow', 'blue']
        
        # cache_dir = f'/dev/shm/cache_{self.uid}/'
        # if not os.path.exists(cache_dir):
            # os.mkdir(cache_dir)
            # data = {}
            # for fname in hdf5_fnames:
                # chunk = load_chunk(fname, self.valid_mods, self.valid_nodes)
                # for ms, val in chunk.items():
                    # if ms in data.keys():
                        # for k, v in val.items():
                            # if k in data[ms].keys():
                                # data[ms][k].update(v)
                            # else:
                                # data[ms][k] = v
                    # else:
                        # data[ms] = val


            # buffers = self.fill_buffers(data)
            # self.active_keys = sorted(buffers[-1].keys())
            
            # count = 0
            # for i in range(len(buffers)):
                # missing = False
                # for key in self.active_keys:
                    # if key not in buffers[i].keys():
                        # missing = True
                # if missing:
                    # count += 1
                    # continue
                # else:
                    # break
            # buffers = buffers[count:]

            # if remove_first_frame:
                # buffers = buffers[1:]
            # if max_len is not None:
                # buffers = buffers[0:max_len]
            
            
            # final_dir = f'{cache_dir}/{self.name}'
            # if not os.path.exists(final_dir):
                # os.mkdir(final_dir)
            # self.fnames = []
            # for i in trange(len(buffers)):
                # buff = buffers[i]
                # fname = '%s/tmp_%09d.pickle' % (final_dir, i)
                # with open(fname, 'wb') as f:
                    # pickle.dump(buff, f)
                # self.fnames.append(fname)
    
        # self.fnames = glob.glob(f'{cache_dir}/{self.name}/*.pickle')
        # print(self.fnames)
        # with open(self.fnames[-1], 'rb') as f:
            # buff = pickle.load(f)
            # self.active_keys = sorted(buff.keys())
        # self.fnames = sorted(self.fnames)[0:max_len]
        # self.nodes = set([key[1] for key in self.active_keys if 'node' in key[1]])

        self.pipelines = {}
        for mod, cfg in pipelines.items():
            self.pipelines[mod] = Compose(cfg)

        self.test_mode = test_mode
        self.flag = np.zeros(len(self), dtype=np.uint8) #ones?
    
    def __len__(self):
        #return len(self.buffers)
        return len(self.fnames)
    
    # def init_buffer(self):
        # buff = {}
        # buff['zed_camera_left'] = np.zeros((10, 10, 3)).astype(np.uint8) #will be resized
        # buff['zed_camera_left'] = cv2.imencode('.jpg', buff['zed_camera_left'])[1] #save compressed
        # buff['zed_camera_depth'] = np.zeros((10, 10, 1)).astype(np.int16) #will be resized
        # buff['range_doppler'] = np.zeros((10, 10))
        # buff = {k: v for k, v in buff.items() if k in self.valid_mods}
        # buff['missing'] = {}
        # for mod in self.MODALITIES:
            # buff['missing'][mod] = True
        # return buff

    def fill_buffers(self, all_data):
        assert 1==2
        buffers = []
        # buff = self.init_buffer()
        buff = {}
        factor = 100 // self.fps
        num_frames = 0
        keys = sorted(list(all_data.keys()))
        prev_num_objs = None
        for time in tqdm(keys, desc='filling buffers'):
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
                        gt_pos /= 10
                    gt_rot = torch.tensor([d['rotation'] for d in mocap_data])
                    # gt_rot = torch.cat([gt_rot[..., 0:2], gt_rot[..., 3:5]], dim=-1)
                    
                    corners, grids = [], []
                    for k in range(len(gt_rot)):
                        angle = rot2angle(gt_rot[k], return_rads=False)
                        rec, grid = gen_rectange(gt_pos[k], angle, w=self.truck_w, h=self.truck_h)
                        corners.append(rec.get_corners())
                        if self.include_z:
                            z_val = gt_pos[k][-1]
                            z_vals = torch.ones(len(grid), 1) * z_val
                            grid = torch.cat([grid, z_vals], dim=-1)
                        grids.append(grid)

                    grids = torch.stack(grids)

                    corners = np.stack(corners)
                    corners = torch.tensor(corners).float()

                    gt_labels = torch.tensor([self.class2idx[d['type']] for d in mocap_data])
                    gt_ids = torch.tensor([d['id'] for d in mocap_data])
                    is_node = gt_labels == 0
                    if self.node_pos is None: 
                        self.node_pos = gt_pos[is_node]
                        self.node_ids = gt_ids[is_node]
                    
                    # gt_pos = gt_pos[~is_node]
                    # z_is_zero = gt_pos[:, -1] == 0.0
                    # if torch.any(z_is_zero):
                        # continue
                    final_mask = ~is_node 
                    if not self.include_z:
                        gt_pos = gt_pos[..., 0:2]
                    gt_pos = gt_pos[final_mask]
                    gt_grid = grids[final_mask] 
                    gt_rot = gt_rot[final_mask]
                    gt_ids = gt_ids[final_mask] - 4
                    if len(gt_pos) < 2:
                        zeros = torch.zeros(2 - len(gt_pos), gt_pos.shape[-1])
                        gt_pos = torch.cat([gt_pos, zeros - 1])
                        
                        zeros = torch.zeros(2 - len(gt_grid), 450, 2)
                        gt_grid = torch.cat([gt_grid, zeros - 1])

                        zeros = torch.zeros(2 - len(gt_rot), 9)
                        gt_rot = torch.cat([gt_rot, zeros - 1])
                        
                        zeros = torch.zeros(2 - len(gt_ids))
                        gt_ids = torch.cat([gt_ids, zeros - 1])
                        
                    buff[('mocap', 'mocap')] = {
                        'gt_positions': gt_pos,
                        #'gt_labels': gt_labels[final_mask].long(),
                        'gt_ids': gt_ids.long(),
                        'gt_rot': gt_rot,
                        #'gt_corners': corners[final_mask],
                        'gt_grids': gt_grid
                    }
                    num_frames += 1
                    save_frame = True

                if 'node' in key:
                    for k, v in data[key].items():
                        if k in self.valid_mods:
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

    def read_buff(self, ind):
        with open(self.fnames[ind], 'rb') as f:
            buff = pickle.load(f)
        return buff
    
    def __getitem__(self, ind):
        #buff = self.buffers[ind]
        buff = self.read_buff(ind)
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
           # buff = self.buffers[idx]
            buff = self.read_buff(idx)
            buff = self.apply_pipelines(buff)
            # buff['ind'] = idx
            buffs.append(buff)
            # new_buff['past_frames'].append(buff)
        
        buffs.append(new_buff)

        # new_buff['future_frames'] = []
        for idx in future_idx:
            # buff = self.buffers[idx]
            buff = self.read_buff(idx)
            buff = self.apply_pipelines(buff)
            # buff['ind'] = idx
            buffs.append(buff)
            # new_buff['future_frames'].append(buff)
        
        #merge time series into a batch
        # res = mmcv.parallel.collate(buffs)
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
        res['num_timesteps'] = len(all_gt_ids)
        res['tracker_ids'] = []
        res['gt_ids'] = all_gt_ids.numpy()
        res['similarity_scores'] = []
        res['grid_scores'] = []
        res['num_tracker_dets'] = 0

        from mmtrack.models.mocap.decoderv4 import calc_grid_loss 
        all_probs, all_dists = [], []
        for i in range(res['num_timesteps']):
            pred_means = outputs['track_means'][i]
            pred_covs = outputs['track_covs'][i]
            pred_ids = outputs['track_ids'][i]
            res['num_tracker_dets'] += len(pred_ids)
            res['tracker_ids'].append(pred_ids.numpy())
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
                samples = dist.sample([10000])
                
                num_gt = len(gt_pos)
                for k in range(num_gt):
                    grid = gt_grid[k]
                    log_probs = dist.log_prob(grid) #*1.5
                    logsum = torch.logsumexp(log_probs.flatten(), dim=0)
                    grid_scores.append(logsum)
                    angle = rot2angle(gt_rot[k], return_rads=False)
                    rec, _ = gen_rectange(gt_pos[k], angle, w=self.truck_w, h=self.truck_h)
                    mask = points_in_rec(samples, rec)
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
        met=CLEAR({'THRESHOLD': 0.5})
        out = met.eval_sequence(res)
        out = {k:float(v) for k,v in out.items()}
        with open(fname, 'w') as f:
            json.dump(out, f)
        return out

    def evaluate(self, outputs, **eval_kwargs):
        metrics = eval_kwargs['metric']
        res = {}
        if 'track' in metrics:
            #res = self.eval_mot(outputs, **eval_kwargs)
            print(res)
        if 'vid' in metrics:
            self.write_video(outputs, **eval_kwargs)
        return res

    def write_video(self, outputs, **eval_kwargs): 
        logdir = eval_kwargs['logdir']
        fname = f'{logdir}/latest_vid.mp4'
        #fig, axes = init_fig(self.active_keys, len(self.nodes))
        fig, axes = init_fig(self.active_keys)
        size = (fig.get_figwidth()*50, fig.get_figheight()*50)
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
                    axes[key].grid('on', linewidth=3)
                    # axes[key].set_facecolor('gray')
                    if self.limit_axis:
                        axes[key].set_xlim(0,700)
                        axes[key].set_ylim(0,500)
                        axes[key].set_aspect('equal')
                    
                    # for j in range(len(self.node_pos)):
                        # pos = self.node_pos[j]
                        # ID = self.node_ids[j] + 1
                        # axes[key].scatter(pos[0], pos[1], marker=f'${ID}$', color='black', lw=1, s=20*4**2)
                    
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
                        axes[key].arrow(pos[0], pos[1], r*rot[0], r*rot[1], head_width=0.05, head_length=0.05, fc=color, ec=color)

                        # axes[key].scatter(grid[...,0], grid[...,1], s=0.5, color='red')

                        # axes[key].text(pos[0], pos[1], '%0.2f %0.2f\n%0.2f %0.2f' % (rot[0],rot[1], rot[3], rot[4]))
                            
                    
                    if len(outputs['det_means']) > 0:
                        pred_means = outputs['det_means'][i]
                        for j in range(len(pred_means)):
                            mean = pred_means[j]
                            axes[key].scatter(mean[0], mean[1], color='black', marker=f'+', lw=1, s=20*4**2)
                    
                    # if 'track_means' in outputs.keys() and len(outputs['track_means'][i]) > 0:
                    pred_means = outputs['track_means'][i] 
                    pred_covs = outputs['track_covs'][i]
                    pred_rots = outputs['track_rot'][i]
                    ids = outputs['track_ids'][i].to(int)
                    slot_ids = outputs['slot_ids'][i].to(int)
                    print(pred_means, pred_covs)
                    for j in range(len(pred_means)):
                        rot = pred_rots[j]
                        mean = pred_means[j]
                        try:
                            angle = torch.arctan(rot[0]/rot[1]) * 360
                        except:
                            import ipdb; ipdb.set_trace() # noqa
                        color = self.colors[j % len(self.colors)]
                        
                        rec, _ = gen_rectange(mean, angle, w=self.truck_w, h=self.truck_h, color=color)
                        axes[key].add_patch(rec)


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
                    img = data[key]['img'].data.cpu().squeeze()
                    mean = data[key]['img_metas'].data['img_norm_cfg']['mean']
                    std = data[key]['img_metas'].data['img_norm_cfg']['std']
                    img = img.permute(1, 2, 0).numpy()
                    img = (img * std) - mean
                    img = img.astype(np.uint8)
                    #img = np.concatenate([img, head_dists], axis=0)
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
