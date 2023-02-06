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


def parse_args():
    parser = argparse.ArgumentParser(description='mmtrack test model')
    parser.add_argument('config', help='test config file path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # assert args.out or args.eval or args.format_only or args.show \
        # or args.show_dir, \
        # ('Please specify at least one operation (save/eval/format/show the '
         # 'results / save the results) with the argument "--out", "--eval"'
         # ', "--format-only", "--show" or "--show-dir"')

    # if args.eval and args.format_only:
        # raise ValueError('--eval and --format_only cannot be both specified')

    # if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        # raise ValueError('The output file must be a pkl file.')
        
    #building the datasets runs the cacher
    #future calls (ie during training) will skip the caching step
    cfg = Config.fromfile(args.config)
    build_dataset(cfg.trainset)
    build_dataset(cfg.valset)
    build_dataset(cfg.testset)
    #build_dataset(cfg.testset)
    

if __name__ == '__main__':
    main()
