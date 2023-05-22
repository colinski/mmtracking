import os
import glob
import pickle
import numpy as np
from mmtrack.datasets import DATASETS
import h5py
import torch
import json
import time
import torchaudio
from tqdm import trange, tqdm
import copy
from .viz import *
from coordinate_transform.Utils import FieldOfViewCheck
from coordinate_transform.Utils import CoordinateTransform

def get_calib():
    calib = {'node_1': np.array([[ 0.9999583 ,  0.00693817,  0.00579411, -0.00704172,  0.99981569, 0.01773981, -0.0056658 , -0.01777992,  0.99982172]]), 
     'node_2': np.array([[ 0.99726239,  0.0655922 ,  0.03401795, -0.06646787,  0.99747923, 0.0250374 , -0.03229125, -0.02723256,  0.99910498]]), 
     'node_3': np.array([[ 0.99926651, -0.03825218,  0.00313152,  0.03821827,  0.99922463, 0.00969683, -0.0034986 , -0.00957323,  0.99994614]]), 
     'node_4': np.array([[ 9.98273147e-01,  3.84921217e-02,  4.42961899e-02, -3.84972003e-02,  9.99256841e-01, -6.86471766e-04, -4.42930857e-02, -1.02671252e-03,  9.99016248e-01]])}
    return calib

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
        keys = list(keys.astype(str))
        data = convert2dict(f, keys, fname, valid_mods, valid_nodes)
    return data

def interp_seq(seq):
    if np.isnan(seq).all():
        return np.zeros_like(seq)
    nan_indices = np.isnan(seq)
    x = np.arange(len(seq))
    seq[nan_indices] = np.interp(x[nan_indices], x[~nan_indices], seq[~nan_indices])
    return seq

@DATASETS.register_module()
class DataCacher(object):
    CLASSES = None
    def __init__(self,
                 hdf5_fnames=[],
                 cache_dir= f'/dev/shm/cache_train/',
                 fps=20,
                 valid_mods=['mocap', 'zed_camera_left', 'zed_camera_depth'],
                 valid_nodes=[1,2,3,4],
                 min_x=-2162.78244, max_x=4157.92774,
                 min_y=-1637.84491, max_y=2930.06133,
                 min_z=0.000000000, max_z=903.616290,
                 normalized_position=False,
                 # max_len=None,
                 truck_w=30/100,
                 truck_h=15/100,
                 include_z=True,
                 **kwargs):
        self.valid_mods = valid_mods
        self.valid_nodes = ['node_%d' % n for n in valid_nodes]
        self.cache_dir = cache_dir
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
        self.truck_w = truck_w
        self.truck_h = truck_h
        self.include_z = include_z
        self.hdf5_fnames = hdf5_fnames
        self.fps = fps
        self.class2idx = {'tunnel': 5, 'drone':4, 'car': 3, 'bus': 2, 'truck': 1, 'node': 0}
        self.fov = FieldOfViewCheck()
        self.class_info = ClassInfo()
        # self.max_len = max_len

    def cache(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=False)
            data = {}
            for fname in self.hdf5_fnames:
                chunk = load_chunk(fname, self.valid_mods, self.valid_nodes)
                for ms, val in chunk.items():
                    if ms in data.keys():
                        for k, v in val.items():
                            if k in data[ms].keys():
                                data[ms][k].update(v)
                            else:
                                data[ms][k] = v
                    else:
                        data[ms] = val

            buffers = self.fill_buffers(data)
            self.active_keys = sorted(buffers[-1].keys())
            
            count = 0
            for i in range(len(buffers)):
                missing = False
                for key in self.active_keys:
                    if key not in buffers[i].keys():
                        missing = True
                if missing:
                    count += 1
                    continue
                else:
                    break
            buffers = buffers[count:]

            # if self.max_len is not None:
                # buffers = buffers[0:self.max_len]
        
            for i in trange(len(buffers)):
                buff = buffers[i]
                fname = '%s/frame_%09d.pickle' % (self.cache_dir, i)
                with open(fname, 'wb') as f:
                    pickle.dump(buff, f)
        
        self.fnames = sorted(glob.glob(f'{self.cache_dir}/*.pickle'))
        with open(self.fnames[-1], 'rb') as f:
            buff = pickle.load(f)
            self.active_keys = sorted(buff.keys())
        # self.fnames = self.fnames[0:self.max_len]
        return self.fnames, self.active_keys

    def interp_mocap(self, all_data):
        keys = sorted(list(all_data.keys()))
        times, all_gt_pos, all_gt_rot, all_gt_labels, all_gt_ids = [], [], [], [], []
        all_widths, all_heights = [], []
        for time in keys:
            data = all_data[time]
            for key in data.keys():
                if key == 'mocap':
                    mocap_data = json.loads(data['mocap'])
                    types = [d['type'] for d in mocap_data]
                    widths = torch.tensor([self.class_info.name2width(t) for t in types])
                    heights = torch.tensor([self.class_info.name2height(t) for t in types])
                    gt_pos = torch.tensor([d['position'] for d in mocap_data])
                    gt_rot = torch.tensor([d['rotation'] for d in mocap_data])
                    gt_labels = torch.tensor([self.class_info.name2id(d['type']) for d in mocap_data])
                    gt_ids = torch.tensor([d['id'] for d in mocap_data])
                    all_gt_pos.append(gt_pos)
                    all_gt_rot.append(gt_rot)
                    all_gt_labels.append(gt_labels)
                    all_gt_ids.append(gt_ids)
                    all_widths.append(widths)
                    all_heights.append(heights)
                    times.append(time)
        all_gt_pos = torch.stack(all_gt_pos, dim=0).numpy()
        sums = all_gt_pos.sum(axis=-1)
        missing_mask = sums == 0
        all_gt_pos[missing_mask] = np.nan
        seq_len, num_objs, num_coor = all_gt_pos.shape
        for i in range(num_objs):
            for j in range(num_coor):
                seq = all_gt_pos[:, i, j]
                seq = interp_seq(seq)
                all_gt_pos[:, i, j] = seq

        all_gt_rot = torch.stack(all_gt_rot, dim=0).numpy()
        seq_len, num_objs, num_coor = all_gt_rot.shape
        for i in range(num_objs):
            for j in range(num_coor):
                seq = all_gt_rot[:, i, j]
                seq = interp_seq(seq)
                all_gt_rot[:, i, j] = seq
        
        mocap_data = {}
        for i, t in enumerate(times):
            mocap_data[t] = {}
            mocap_data[t]['gt_pos'] = all_gt_pos[i]
            mocap_data[t]['gt_rot'] = all_gt_rot[i]
            mocap_data[t]['gt_labels'] = all_gt_labels[i]
            mocap_data[t]['gt_ids'] = all_gt_ids[i]
            mocap_data[t]['widths'] = all_widths[i]
            mocap_data[t]['heights'] = all_heights[i]

        return mocap_data

    def fill_buffers(self, all_data):
        interp_mocap_data = self.interp_mocap(all_data)
        buffers = []
        buff = {}
        factor = 100 // self.fps
        num_frames = 0
        keys = sorted(list(all_data.keys()))
        prev_num_objs = None
        calib = get_calib()
        for time in tqdm(keys, desc='filling buffers'):
            save_frame = False
            data = all_data[time]
            for key in data.keys():
                if key == 'mocap':
                    #mocap_data = json.loads(data['mocap'])
                    mocap_data = interp_mocap_data[time]
                    
                    # types = [d['type'] for d in mocap_data]
                    # widths = torch.tensor([self.class_info.name2width(t) for t in types])
                    # heights = torch.tensor([self.class_info.name2height(t) for t in types])
                    # gt_pos = torch.tensor([d['position'] for d in mocap_data])
                    # gt_rot = torch.tensor([d['rotation'] for d in mocap_data])
                    # gt_labels = torch.tensor([self.class_info.name2id(d['type']) for d in mocap_data])
                    # gt_ids = torch.tensor([d['id'] for d in mocap_data])
                    widths = mocap_data['widths']
                    heights = mocap_data['heights']
                    gt_pos = torch.from_numpy(mocap_data['gt_pos'])
                    gt_rot = torch.from_numpy(mocap_data['gt_rot'])
                    gt_labels = mocap_data['gt_labels']
                    gt_ids = mocap_data['gt_ids']
                    
                    is_node = gt_labels == 0

                    missing_mask = gt_pos[:, -1] == 0

                    valid_mask = ~is_node & ~missing_mask

                    node_pos = gt_pos[is_node]
                    node_rot = gt_rot[is_node]
                    num_nodes = len(node_pos)
                    num_objs = len(gt_pos)
                    visible = torch.zeros(num_objs, num_nodes)
                    pixels = torch.zeros(num_objs, num_nodes, 2)
                    for i in range(num_nodes):
                        npos = node_pos[i]
                        nrot = node_rot[i]
                        node_name = 'node_%d' % (i+1)
                        calib_rot = calib[node_name]
                        calib_nrot = CoordinateTransform.local_to_global_rotation(
                                nrot, calib_rot)

                        for j in range(num_objs):
                            opos = gt_pos[j]
                            orot = gt_rot[j]
                            try: #error if npos is all zeros
                                lpos, lrot = CoordinateTransform.global_to_local(
                                     npos, calib_nrot, opos, orot)
                            except:
                                continue
                            u, v = self.fov.local_to_pixel(lpos, node_name, 'zed')
                            pixels[j, i] = torch.tensor([u,v])
                            try:
                                check = self.fov.validate_field_of_view_raw(
                                    npos, nrot, opos, orot, calib_rot, 'zed')
                            except:
                                continue
                            visible[j, i] = check

                    if valid_mask.sum() == 0:
                        buff[('mocap', 'mocap')] = buffers[-1][('mocap', 'mocap')]
                        print('all objects missing')
                    else:
                        buff[('mocap', 'mocap')] = {
                            'gt_positions': gt_pos,
                            'gt_labels': gt_labels.long(),
                            'gt_ids': gt_ids.long(),
                            'gt_rot': gt_rot.reshape(-1, 3, 3),
                            'widths': widths,
                            'heights': heights,
                            'visible': visible,
                            'pixels': pixels,
                            'valid_mask': valid_mask,
                            'missing_mask': missing_mask
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
