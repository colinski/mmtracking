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
import matplotlib.patches as patches
import pandas as pd


font = {#'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

#obj_pos is num_object x 3
#node_pos is (3,)
#node_rot is (3,3)
def local2global(obj_pos, node_pos, node_rot):
    obj_pos = obj_pos.t()
    obj_pos = (node_rot.t() @ obj_pos) 
    obj_pos = obj_pos + node_pos.unsqueeze(1)
    obj_pos = obj_pos.t()
    return obj_pos

def global2local(obj_pos, node_pos, node_rot):
    obj_pos = obj_pos.t()
    obj_pos = obj_pos - node_pos.unsqueeze(1)
    obj_pos = node_rot @ obj_pos
    obj_pos = obj_pos.t()
    return obj_pos

class ClassInfo:
    def __init__(self, info={
        'node': {'size': [15, 30, 0], 'id': 0, 'color': 'black'},
        'truck': {'size': [30, 15, 0], 'id': 1, 'color': 'orange'},
        'bus': {'size': [31, 8, 0], 'id': 2, 'color': 'red'},
        'car': {'size': [29, 13, 0], 'id': 3, 'color': 'blue'},
        'drone': {'size': [32, 27, 8], 'id': 4, 'color': 'gray'},
        'tunnel': {'size': [30, 30, 0], 'id': 5, 'color': 'brown'}
    }):
        self.info = pd.DataFrame.from_dict(info).T

    def name2id(self, name):
        return self.info.loc[name]['id']

    def id2name(self, id):
        try:
            return self.info[self.info['id'] == id].index[0]
        except:
            import ipdb; ipdb.set_trace() # noqa

    def id2color(self, id):
        return self.info.loc[self.id2name(id)]['color']

    def id2width(self, id):
        return self.info.loc[self.id2name(id)]['size'][0]

    def id2height(self, id):
        return self.info.loc[self.id2name(id)]['size'][1]
    
    def name2width(self, name):
        return self.info.loc[name]['size'][0]

    def name2height(self, name):
        return self.info.loc[name]['size'][1]

def get_meta_():
    meta = {
        'node': {'size': [15, 30, 0], 'id': 0},
        'truck': {'size': [30, 15, 0], 'id': 1},
        'bus': {'size': [31, 8, 0], 'id': 2},
        'car': {'size': [29, 13, 0], 'id': 3},
        'drone': {'size': [32, 27, 8], 'id': 4},
        'tunnel': {'size': [30, 30, 0], 'id': 5},
    }
    return meta

def get_node_info_(fill=False, alpha=1): 
    node_pos = torch.tensor([
        [ 3919.8430,   337.2432,   369.0333],
        [  155.9429, -1517.3743,   767.7822],
        [-2040.2272,  -464.5895,   901.6157],
        [ 2031.9064,  2279.1736,   724.6130],
    ])

    node_rot = torch.tensor([
        [-9.9990e-01,  1.4300e-02, -6.0000e-04, -1.4300e-02, -9.9990e-01,
          1.3000e-03, -5.8000e-04,  1.3100e-03,  1.0000e+00],
        [ 1.6939e-01,  9.7809e-01, -1.2103e-01, -9.8551e-01,  1.6917e-01,
         -1.2150e-02,  8.6000e-03,  1.2134e-01,  9.9257e-01],
        [ 9.6419e-01,  2.4893e-01, -9.1470e-02, -2.4892e-01,  9.6845e-01,
          1.1750e-02,  9.1500e-02,  1.1440e-02,  9.9574e-01],
        [-6.3167e-01, -7.6565e-01, -1.2154e-01,  7.6975e-01, -6.3807e-01,
          1.8980e-02, -9.2080e-02, -8.1570e-02,  9.9240e-01]
    ])

    nodes = {}
    for j in range(len(node_pos)):
        pos = node_pos[j]
        name = 'node_{}'.format(j+1)
        nodes[name] = {'pos': node_pos[j], 'rot': node_rot[j]}
    return nodes

def get_node_info(fill=False, alpha=1): 
    node_pos = torch.tensor([
        [608.2496, 197.5388],
        [231.8911,  12.0564],
        [ 12.2432, 117.5110],
        [419.3237, 391.6695]
    ])

    nodes = {}
    for j in range(len(node_pos)):
        pos = node_pos[j]
        name = 'node_{}'.format(j+1)
        if j == 0:
            xy = [(pos[0] - 30, pos[1]), (500,0), (0,0), (0,500), (350,500)]
            poly = patches.Polygon(xy=xy, fill=fill, color='red', alpha=alpha)
            #nodes[name] = {'poly': poly, 'pos': pos, 'id': j+1}
            nodes[name] = {'points': xy, 'pos': pos, 'id': j+1, 'color': 'red'}
        if j == 1:
            xy = [(pos[0], pos[1] + 60), (0,250), (0,500), (700,500)]
            poly = patches.Polygon(xy=xy, fill=fill, color='blue', alpha=alpha)
            #nodes[name] = {'poly': poly, 'pos': pos, 'id': j+1}
            nodes[name] = {'points': xy, 'pos': pos, 'id': j+1, 'color': 'blue'}
        if j == 2:
            xy = [(50, pos[1]), (150,500), (700,500), (700,0), (250,0)]
            poly = patches.Polygon(xy=xy, fill=fill, color='green', alpha=alpha)
            #nodes[name] = {'poly': poly, 'pos': pos, 'id': j+1}
            nodes[name] = {'points': xy, 'pos': pos, 'id': j+1, 'color': 'green'}
        if j == 3:
            xy = [(pos[0]-25, pos[1]-25), (0,300), (0,0),(500,0)]
            #poly = patches.Polygon(xy=xy, fill=fill, color='black', alpha=alpha)
            nodes[name] = {'points': xy, 'pos': pos, 'id': j+1, 'color': 'black'}
            #nodes[name] = {'poly': poly, 'pos': pos, 'id': j+1}
    return nodes


#chatGPT4 
def points_in_polygon(polygon, point):
    """
    Determines if a point (x, y) falls inside a given polygon.

    Args:
        polygon (matplotlib.patches.Polygon): A Polygon object from matplotlib.
        point (tuple): A tuple containing the (x, y) coordinates of the point to be checked.

    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    # Get the polygon vertices as a NumPy array
    polygon_vertices = np.array(polygon.get_xy())

    # Create a Path object from the polygon vertices
    path = matplotlib.path.Path(polygon_vertices)

    # Check if the point is inside the path
    return path.contains_point(point)

# Example usage:
# polygon = matplotlib.patches.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
# point = (0.5, 0.5)
# print(is_point_inside_polygon(polygon, point))  # Should print True

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
def gen_ellipse(pos, cov, nstd=np.sqrt(5.991), **kwargs):
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

def rot2angle(rot, return_rads=True):
    rot = rot.flatten()
    if rot[4] <= 0:
        rads = np.arcsin(rot[3]) / (2*np.pi)
    else:
        rads = np.arcsin(rot[1]) / (2*np.pi)
    if not return_rads:
        rads *= 360
    return rads


def gen_rectange(pos, angle, w, h, color='black'):
    # angle = rot2angle(rot, return_rads=False)
    rec = Rectangle(xy=([pos[0]-w/2, pos[1]-h/2]), width=w, height=h, angle=angle, rotation_point='center',
                        edgecolor=color, fc='None', lw=5)
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

def init_fig_(valid_mods, num_cols=4, colspan=1):
    assert ('mocap', 'mocap') in valid_mods

    mods = [vk[0] for vk in valid_mods if vk != ('mocap', 'mocap')]
    mods = sorted(list(set(mods)))
    num_mods = len(mods)
    num_cols = num_mods + 1
    num_rows = 4
    
    fig = plt.figure(figsize=(num_cols*10, num_rows*10))
    axes = {}
    axes[('mocap', 'mocap')] = plt.subplot2grid((num_rows, num_cols), (1, 0), 
            rowspan=1, colspan=1)

    axes[('mocap', 'mocap')].linewidth = 5
    axes[('mocap', 'mocap')].node_size = 20*4**2

    valid_mods = [vk for vk in valid_mods if vk != ('mocap', 'mocap')]
    for i, key in enumerate(valid_mods):
        col = mods.index(key[0])
        row = int(key[1].split('_')[-1]) - 1
        # row += 2
        col += 1

        # x, y = i % num_mods, num + num_mods
        print(row, col, key)
        axes[key] = plt.subplot2grid((num_rows, num_cols), (row, col))
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    return fig, axes

def init_fig_vert(valid_mods, num_cols=4, colspan=1):
    assert ('mocap', 'mocap') in valid_mods

    mods = [vk[0] for vk in valid_mods if vk != ('mocap', 'mocap')]
    mods = sorted(list(set(mods)))
    num_mods = len(mods)
    num_cols = 2 + 4
    num_rows = num_mods
    
    fig = plt.figure(figsize=(num_cols*10, num_rows*10))
    axes = {}
    axes[('mocap', 'mocap')] = plt.subplot2grid((num_rows, num_cols), (0, 0), 
            rowspan=2, colspan=2)

    valid_mods = [vk for vk in valid_mods if vk != ('mocap', 'mocap')]
    for i, key in enumerate(valid_mods):
        row = mods.index(key[0])
        col = int(key[1].split('_')[-1]) - 1
        col += 2
        # x, y = i % num_mods, num + num_mods
        print(row, col, key)
        axes[key] = plt.subplot2grid((num_rows, num_cols), (row, col))
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    return fig, axes

def init_fig(valid_mods, num_cols=4, colspan=1):
    assert ('mocap', 'mocap') in valid_mods

    mods = [vk[0] for vk in valid_mods if vk != ('mocap', 'mocap')]
    num_mods = len(set(mods))
    num_cols = num_mods + 2 + 1
    num_rows = num_mods + 2 + 1
    
    fig = plt.figure(figsize=(num_cols*16, num_rows*9))
    axes = {}
    axes[('mocap', 'mocap')] = plt.subplot2grid((num_rows, num_cols), (1, 1), rowspan=num_mods + 1, colspan=num_mods+1)

    axes[('mocap', 'mocap')].linewidth = 20
    axes[('mocap', 'mocap')].node_size = 20*16**2

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
             
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    return fig, axes
