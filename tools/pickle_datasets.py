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
    parser.add_argument('--train_region', nargs='+', default=[0.0,0.5])
    parser.add_argument('--val_region', nargs='+', default=[0.5,0.6])
    parser.add_argument('--test_region', nargs='+', default=[0.6,1.0])
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
    
    num_fnames = len(fnames)
    #train_fnames = fnames[0:int(args.train_region[1]*num_fnames)]
    train_fnames = fnames[int(args.train_region[0]*num_fnames):int(args.train_region[1]*num_fnames)]
    os.makedirs(f'{args.pickle_path}/train', exist_ok=True)
    for fname in train_fnames:
        shutil.move(fname, f'{args.pickle_path}/train')
    train_fnames = sorted(glob.glob(f'{args.pickle_path}/train/*.pickle'))
    meta = {'fps': args.fps, 'active_keys': active_keys, 'fnames': train_fnames,
            'valid_nodes': args.valid_nodes, 'valid_mods': args.valid_mods}
    with open(f'{args.pickle_path}/train/meta.json', 'w') as f:
        json.dump(meta, f)

    #val_fnames = fnames[int(args.train_region[1]*num_fnames):int(args.val_region[1]*num_fnames)]
    val_fnames = fnames[int(args.val_region[0]*num_fnames):int(args.val_region[1]*num_fnames)]
    os.makedirs(f'{args.pickle_path}/val', exist_ok=True)
    for fname in val_fnames:
        shutil.move(fname, f'{args.pickle_path}/val')
    val_fnames = sorted(glob.glob(f'{args.pickle_path}/val/*.pickle'))
    meta = {'fps': args.fps, 'active_keys': active_keys, 'fnames': val_fnames,
            'valid_nodes': args.valid_nodes, 'valid_mods': args.valid_mods}
    with open(f'{args.pickle_path}/val/meta.json', 'w') as f:
        json.dump(meta, f)

    #test_fnames = fnames[int(args.val_region[1]*num_fnames):]
    test_fnames = fnames[int(args.test_region[0]*num_fnames):int(args.test_region[1]*num_fnames)]
    os.makedirs(f'{args.pickle_path}/test', exist_ok=True)
    for fname in test_fnames:
        shutil.move(fname, f'{args.pickle_path}/test')
    test_fnames = sorted(glob.glob(f'{args.pickle_path}/test/*.pickle'))
    meta = {'fps': args.fps, 'active_keys': active_keys, 'fnames': test_fnames,
            'valid_nodes': args.valid_nodes, 'valid_mods': args.valid_mods}
    with open(f'{args.pickle_path}/test/meta.json', 'w') as f:
        json.dump(meta, f)


if __name__ == '__main__':
    main()
