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

font = {#'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

#https://gamedev.stackexchange.com/questions/86755/how-to-calculate-corner-positions-marks-of-a-rotated-tilted-rectangle
def is_on_right_side(points, v1, v2):
    x0, y0 = v1
    x1, y1 = v2
    a = y1 - y0
    b = x0 - x1
    c = - a*x0 - b*y0
    return a*points[:,0] + b*points[:,1] + c >= 0

def points_in_rec(points, rec):
    corners = rec.get_corners()
    num_corners = len(corners)
    is_right = [is_on_right_side(points, corners[i], corners[(i + 1) % num_corners]) for i in range(num_corners)]
    is_right = np.stack(is_right, axis=1)
    all_left = ~np.any(is_right, axis=1)
    all_right = np.all(is_right, axis=1)
    final = all_left | all_right
    return final

def rot_matrix(angle):
    rad = 2*np.pi * (angle/360)
    R = [np.cos(rad), np.sin(rad),-np.sin(rad), np.cos(rad)]
    R = np.array(R).reshape(2,2)
    R = torch.from_numpy(R).float()
    return R


#https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals/12321306#12321306
def gen_ellipse(pos, cov, nstd=2, **kwargs):
    if len(pos) > 2:
        pos = pos[0:2]
        cov = cov[0:2, 0:2]
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    return ellip

def gen_rectange(pos, rot, w, h, color='black'):
    if rot[4] <= 0:
        rads = np.arcsin(rot[3]) / (2*np.pi)
    else:
        rads = np.arcsin(rot[1]) / (2*np.pi)

    angle = rads * 360
    rec = Rectangle(xy=([pos[0]-w/2, pos[1]-h/2]), width=w, height=h, angle=angle, rotation_point='center',
                        edgecolor=color, fc='None', lw=1, linestyle='--')
    corners = rec.get_corners()

    x = np.arange(0.5,30,1) / 100.0
    y = np.arange(0.5,15,1) / 100.0
    X, Y = np.meshgrid(x,y)
    grid = np.stack([X,Y])
    grid = torch.from_numpy(grid).float()
    grid = grid.permute(1,2,0)
    grid = grid.reshape(-1,2)
    R = rot_matrix(angle)
    grid = torch.mm(grid, R)
    grid[:,0] += corners[0][0]
    grid[:,1] += corners[0][1]
    return rec, grid


def init_fig(valid_mods, num_cols=4, colspan=1):
    assert ('mocap', 'mocap') in valid_mods

    mods = [vk[0] for vk in valid_mods if vk != ('mocap', 'mocap')]
    num_mods = len(set(mods))
    num_cols = num_mods + 2 + 1
    num_rows = num_mods + 2 + 1
    
    # num_plots = len(valid_mods)
    # num_mods = num_plots - 1
    #num_cols = int(np.ceil(num_mods / num_rows)) + colspan
    # num_rows = num_mods + 1

    # num_rows, num_cols = 2 + 5, 7 + 2
    # num_cols = 4
    fig = plt.figure(figsize=(num_cols*16, num_rows*9))
    # fig = plt.figure(figsize=(16, 9))
    axes = {}
    #axes[('mocap', 'mocap')] = plt.subplot2grid((num_rows, num_cols), (0, 0), rowspan=num_rows, colspan=colspan)
    axes[('mocap', 'mocap')] = plt.subplot2grid((num_rows, num_cols), (1, 1), rowspan=num_mods + 1, colspan=num_mods+1)

    #row, col = 0, colspan
    node2row = {'node_2': num_rows-1, 'node_4': 0}
    node2col = {'node_3': 0, 'node_1': num_cols-1}
   
    valid_mods = [vk for vk in valid_mods if vk != ('mocap', 'mocap')]
    for node_num, col_num in node2col.items():
        count = 1
        for i, key in enumerate(valid_mods):
            if key[1] != node_num:
                continue
            axes[key] = plt.subplot2grid((num_rows, num_cols), (count, col_num))
            count += 1

    
    for node_num, row_num in node2row.items():
        count = 1
        for i, key in enumerate(valid_mods):
            if key[1] != node_num:
                continue
            axes[key] = plt.subplot2grid((num_rows, num_cols), (row_num, count))
            count += 1
             
    # for i, key in enumerate(valid_mods):
        # mod, node = key
        # if node == 'node_2':
            # col = num_cols
            # axes[key] = plt.subplot2grid((num_rows, num_cols), (row, col))

    # row, col = 1, 0
    # for i, key in enumerate(valid_mods):
        # axes[key] = plt.subplot2grid((num_rows, num_cols), (row, col))
        # row += 1
        # if row  == num_rows:
            # row = 0
            # col += 1

    # fig.suptitle('Title', fontsize=11)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    return fig, axes
