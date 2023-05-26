# Copyright (c) OpenMMLab. All rights reserved.
import math

import cv2
import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets.builder import PIPELINES
#from mmdet.datasets.pipelines import Normalize, Pad, RandomFlip, Resize
import torch

@PIPELINES.register_module()
class PadObjects(object):
    def __init__(self, pad_size=10, pad_value=-1.0):
        self.pad_size = pad_size
        self.pad_value = pad_value
    
    def _pad(self, d):
        for key, val in d.items():
            if type(val) == dict:
                d[key] = self._pad(val)
                return d
            pad = torch.zeros_like(val) + self.pad_value
            val = torch.cat([val, pad], dim=0)
            val = val[0:self.pad_size]
            d[key] = val
        return d

    def __call__(self, results):
        return self._pad(results)
        # new_results = {}
        # for key, val in results.items():
            # pad = torch.zeros_like(val) + self.pad_value
            # val = torch.cat([val, pad], dim=0)
            # val = val[0:self.pad_size]
            # new_results[key] = val
        # return new_results

@PIPELINES.register_module()
class PruneObjects(object):
    def __init__(self):
        pass

    def __call__(self, results):
        gt_pos = results['gt_positions']
        mask = gt_pos[:, -1] != 0.0
        new_results = {}
        for key, val in results.items():
            # if key == 'pixels':
                # continue
            new_results[key] = val[mask]
        return new_results

@PIPELINES.register_module()
class ScaleMocap(object):
    def __init__(self, x_min=-1, y_min=-1, z_min=0):
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min

    def __call__(self, results):
        gt_pos = results['gt_positions']
        mask = gt_pos[:, -1] != 0.0
        results['gt_positions_raw'] = gt_pos
        # gt_pos[mask][:, 0] += np.abs(self.x_min)
        # gt_pos[mask][:, 1] += np.abs(self.y_min)
        # gt_pos[mask][:, 2] += np.abs(self.z_min)
        
        gt_pos[:, 0] += np.abs(self.x_min)
        gt_pos[:, 1] += np.abs(self.y_min)
        gt_pos[:, 2] += np.abs(self.z_min)

        gt_pos[~mask] = gt_pos[~mask]*0 - 1

        results['gt_positions'] = gt_pos
        return results

@PIPELINES.register_module()
class mm2cm(object):
    def __init__(self):
        pass

    def __call__(self, results):
        gt_pos = results['gt_positions']
        mask = gt_pos[:, -1] != -1
        gt_pos[mask] = gt_pos[mask] / 10
        results['gt_positions'] = gt_pos
        return results

@PIPELINES.register_module()
class DropZ(object):
    def __init__(self):
        pass

    def __call__(self, results):
        gt_pos = results['gt_positions']
        gt_pos = gt_pos[:, 0:2]
        results['gt_positions'] = gt_pos
        return results
