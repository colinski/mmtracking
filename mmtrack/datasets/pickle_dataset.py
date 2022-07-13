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

def parse(data, subkey):
    pass

@DATASETS.register_module()
class PickleDataset(Dataset, metaclass=ABCMeta):
    CLASSES = None

    def __init__(self,
                 file_prefix,
                 img_pipeline=None,
                 depth_pipeline=None,
                 azimuth_pipeline=None,
                 range_pipeline=None,
                 audio_pipeline=None,
                 test_mode=False,
                 **kwargs):
        self.file_prefix = file_prefix
        self.img_pipeline = Compose(img_pipeline)
        self.depth_pipeline = Compose(depth_pipeline)
        self.azimuth_pipeline = Compose(azimuth_pipeline)
        self.range_pipeline = Compose(range_pipeline)
        self.test_mode = test_mode

        self.fnames = glob.glob(self.file_prefix + '/*.pickle')
        self.fnames = sorted(self.fnames)
        self.flag = np.zeros(len(self), dtype=np.uint8) #ones?


    def __len__(self):
        return len(self.fnames)
        
    def __getitem__(self, ind):
        with open(self.fnames[ind], 'rb') as f:
            data = pickle.load(f)
        
        res = {}
        objs = []
        mocap = data['mocap_data']
        obj_names = ['NOD_1', 'NOD_2', 'NOD_3', 'NOD_4',
                'Truck_1', 'Truck_2']
    
        for on in obj_names:
            obj_type, obj_id = on.split('_')
            obj_type = 'node' if obj_type == 'NOD' else obj_type
            obj = {
                'id': len(objs),
                'type': obj_type.lower(),
                'position': np.array([mocap[f'{on} X'], mocap[f'{on} Y'], mocap[f'{on} Z']]),
                'roll': mocap[f'{on} Roll'],
                'yaw': mocap[f'{on} Yaw'],
                'pitch': mocap[f'{on} Pitch'],
                'residual': mocap[f'{on} Residual'],
                'rotation': np.array([mocap[f'{on} Rot[{i}]'] for i in range(0, 9)]),
                'timestamp': mocap['Time']
            }
            objs.append(obj)
        res['objects'] = objs
        res['capture_object_id'] = 0

        res['camera'] = data['zed']

        res['radar'] = data['mmwave']
        res['radar']['azimuth_static'] = np.nan_to_num(res['radar']['azimuth_static']).astype(np.float32)
        res['radar']['range_doppler'] = res['radar']['range_doppler'].astype(np.int16)
        res['radar']['range_profile'] = np.array(res['radar']['range_profile'],  dtype=np.float32)
        res['radar']['noise_profile'] = np.array(res['radar']['noise_profile'],  dtype=np.float32)

        res['mic'] = {}
        waveform = data['respeaker']['flac'].get_array_of_samples()
        waveform = np.array(waveform)
        waveform = waveform.reshape(-1, 6)
        waveform = waveform[:, 1:5]
        res['mic']['waveform'] = waveform
        res['mic']['direction'] = data['respeaker']['direction']
        res['mic']['direction_timestamp'] = data['respeaker']['direction_time']
        res['mic']['waveform_timestamp'] = data['respeaker']['flac_time']

        res['camera']['left'] = cv2.resize(res['camera']['left'], dsize=(480, 270))
        res['camera']['right'] = cv2.resize(res['camera']['right'], dsize=(480, 270))
        res['camera']['depth'] = cv2.resize(res['camera']['depth'], dsize=(480, 270))
        
        import ipdb; ipdb.set_trace() # noqa
        # waveform = waveform.reshape(1, -1, 4)
        
        img = self.img_pipeline(data['zed']['left'])
        depth = self.depth_pipeline(data['zed']['depth'])
        azimuth = self.azimuth_pipeline(data['mmwave']['azimuth_static'])
        drange = self.range_pipeline(data['mmwave']['range_doppler'])
