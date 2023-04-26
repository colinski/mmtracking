# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import set_random_seed

from mmtrack.core import setup_multi_processes
from mmtrack.datasets import build_dataset
from mmtrack.datasets.mocap.cacher import DataCacher
import glob
import json
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='mmtrack test model')
    parser.add_argument('hdf5_path', help='path to input hdf5 files')
    parser.add_argument('pickle_path', help='path to save pickle files')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--valid_nodes', type=int, nargs='+', default=[1,2,3,4])
    parser.add_argument('--valid_mods', nargs='+', default=['mocap'])
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--len', type=int, default=-1)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    if os.path.exists(args.pickle_path) and args.overwrite:
        shutil.rmtree(args.pickle_path)

    hdf5_fnames = [f'{args.hdf5_path}/mocap.hdf5']
    for node_idx in args.valid_nodes:
        fnames = glob.glob(f'{args.hdf5_path}/node_{node_idx}/*.hdf5')
        hdf5_fnames.extend(fnames)

    cacher = DataCacher(
        cache_dir=args.pickle_path,
        hdf5_fnames=hdf5_fnames,
        valid_nodes=args.valid_nodes,
        valid_mods=args.valid_mods,
        fps=args.fps,
    )
    fnames, active_keys = cacher.cache()
    if args.len != -1:
        fnames = fnames[0:args.len]
    meta = {'fps': args.fps, 'active_keys': active_keys, 'fnames': fnames,
            'valid_nodes': args.valid_nodes, 'valid_mods': args.valid_mods}
    with open(f'{args.pickle_path}/meta.json', 'w') as f:
        json.dump(meta, f)

if __name__ == '__main__':
    main()
