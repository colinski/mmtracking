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
        
        mocap = data['mocap_data']
        truck = {
            'id': 1,
            'type': 'truck',
            'position': (mocap['Truck_1 X'], mocap['Truck_1 Y'], mocap['Truck_1 Z']),
            'yaw': mocap['Truck_1 Yaw'],
            'pitch': mocap['Truck_1 Pitch'],
        }
        import ipdb; ipdb.set_trace() # noqa
        
        img = self.img_pipeline(data['zed']['left'])
        depth = self.depth_pipeline(data['zed']['depth'])
        azimuth = self.azimuth_pipeline(data['mmwave']['azimuth_static'])
        drange = self.range_pipeline(data['mmwave']['range_doppler'])
