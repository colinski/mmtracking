_base_ = [
    '../_base_/datasets/mmm/2022-09-01/trucks2_lightsT_obstaclesF.py'
]

img_norm_cfg = dict(mean=[0,0,0], std=[255,255,255], to_rgb=True)
img_pipeline = [
    dict(type='DecodeJPEG'),
    dict(type='LoadFromNumpyArray'),
    #dict(type='Resize', img_scale=(32*60, 32*34), keep_ratio=False),
    dict(type='Resize', img_scale=(480, 288), keep_ratio=False),
    #dict(type='Resize', img_scale=(35*32, 25*32), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

pipelines = {
    'zed_camera_left': img_pipeline,
}

#data_root = 'data/mmm/2022-09-01_1080p/trucks1_lightsT_obstaclesF/train'
data_root1 = 'data/mmm/2022-09-01/trucks1_lightsT_obstaclesF/train'
data_root2 = 'data/mmm/2022-09-01/trucks2_lightsT_obstaclesF/train'
trainset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir='/dev/shm/cache_train/',
        hdf5_fnames=[
            f'{data_root1}/mocap.hdf5',
            f'{data_root1}/node_1/zed.hdf5',
            f'{data_root1}/node_2/zed.hdf5',
            f'{data_root1}/node_3/zed.hdf5',
            f'{data_root1}/node_4/zed.hdf5',
            f'{data_root2}/mocap.hdf5',
            f'{data_root2}/node_1/zed.hdf5',
            f'{data_root2}/node_2/zed.hdf5',
            f'{data_root2}/node_3/zed.hdf5',
            f'{data_root2}/node_4/zed.hdf5',
        ],
        valid_nodes=[1,2,3,4],
        valid_mods=['mocap', 'zed_camera_left'],
        include_z=False,
    ),
    num_future_frames=0,
    num_past_frames=0,
    pipelines=pipelines
)

data_root1 = 'data/mmm/2022-09-01/trucks1_lightsT_obstaclesF/val'
#data_root2 = 'data/mmm/2022-09-01/trucks2_lightsT_obstaclesF/val'
valset1=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir='/dev/shm/cache_val1/',
        hdf5_fnames=[
            f'{data_root1}/mocap.hdf5',
            f'{data_root1}/node_1/zed.hdf5',
            f'{data_root1}/node_2/zed.hdf5',
            f'{data_root1}/node_3/zed.hdf5',
            f'{data_root1}/node_4/zed.hdf5',
            # f'{data_root2}/mocap.hdf5',
            # f'{data_root2}/node_1/zed.hdf5',
            # f'{data_root2}/node_2/zed.hdf5',
            # f'{data_root2}/node_3/zed.hdf5',
            # f'{data_root2}/node_4/zed.hdf5',
        ],
        valid_nodes=[1,2,3,4],
        valid_mods=['mocap', 'zed_camera_left'],
        include_z=False,
    ),
    num_future_frames=0,
    num_past_frames=0,
    pipelines=pipelines
)

data_root2 = 'data/mmm/2022-09-01/trucks2_lightsT_obstaclesF/val'
valset2=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir='/dev/shm/cache_val2/',
        hdf5_fnames=[
            f'{data_root2}/mocap.hdf5',
            f'{data_root2}/node_1/zed.hdf5',
            f'{data_root2}/node_2/zed.hdf5',
            f'{data_root2}/node_3/zed.hdf5',
            f'{data_root2}/node_4/zed.hdf5',
        ],
        valid_nodes=[1,2,3,4],
        valid_mods=['mocap', 'zed_camera_left'],
        include_z=False,
    ),
    num_future_frames=0,
    num_past_frames=0,
    pipelines=pipelines
)


data_root1 = 'data/mmm/2022-09-01/trucks1_lightsT_obstaclesF/test'
data_root2 = 'data/mmm/2022-09-01/trucks2_lightsT_obstaclesF/test'
testset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir='/dev/shm/cache_test/',
        hdf5_fnames=[
            f'{data_root1}/mocap.hdf5',
            f'{data_root1}/node_1/zed.hdf5',
            f'{data_root1}/node_2/zed.hdf5',
            f'{data_root1}/node_3/zed.hdf5',
            f'{data_root1}/node_4/zed.hdf5',
            f'{data_root2}/mocap.hdf5',
            f'{data_root2}/node_1/zed.hdf5',
            f'{data_root2}/node_2/zed.hdf5',
            f'{data_root2}/node_3/zed.hdf5',
            f'{data_root2}/node_4/zed.hdf5',
        ],
        valid_nodes=[1,2,3,4],
        valid_mods=['mocap', 'zed_camera_left'],
        include_z=False,
    ),
    num_future_frames=0,
    num_past_frames=0,
    pipelines=pipelines
)

# backbone_cfg=[
    # dict(type='YOLOv7', weights='src/mmtracking/yolov7-tiny.pt'),
    # dict(type='ChannelMapper',
        # in_channels=[512],
        # kernel_size=1,
        # out_channels=256,
        # act_cfg=None,
        # norm_cfg=None,
        # num_outs=1
    # )
# ]

backbone_cfg=dict(type='YOLOv7', weights='src/mmtracking/yolov7-tiny.pt', return_idx=1)


adapter_cfg=dict(type='ConvAdapter', 
        interpolate_size=(28,20),
        interpolate_fn='avgpool'
)

adapter_cfgs = {('zed_camera_left', 'node_1'): adapter_cfg,
              ('zed_camera_left', 'node_2'): adapter_cfg,
              ('zed_camera_left', 'node_3'): adapter_cfg,
              ('zed_camera_left', 'node_4'): adapter_cfg}

backbone_cfgs = {'zed_camera_left': backbone_cfg}

model = dict(type='DetectorEnsemble',
    backbone_cfgs=backbone_cfgs,
    adapter_cfgs=adapter_cfgs,
    output_head_cfg=dict(type='AnchorOutputHead',
        include_z=False,
        predict_full_cov=True,
        cov_add=30.0,
        input_dim=256,
        predict_rotation=False,
        predict_velocity=False,
        num_sa_layers=0,
        to_cm=True,
        mlp_dropout_rate=0.0,
        interval_sizes=[25,25],
        binary_prob=False,
        scale_binary_prob=False
    ),
    entropy_loss_weight=1,
    entropy_loss_type='mse',
    track_eval=True,
    pos_loss_weight=1,
    num_queries=1,
    mod_dropout_rate=0.0,
    loss_type='nll',
)


# orig_bs = 2
# orig_lr = 1e-4 
# factor = 4
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    shuffle=True, #trainset shuffle only
    train=trainset,
    test=testset,
    val1=valset1,
    val2=valset2,
)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
    # paramwise_cfg=dict(
        # custom_keys={
            # 'backbone': dict(lr_mult=0.1),
            # 'sampling_offsets': dict(lr_mult=0.1),
            # 'reference_points': dict(lr_mult=0.1)
        # }
    # )
)

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
total_epochs = 50
lr_config = dict(policy='step', step=[40])
evaluation = dict(metric=['bbox', 'track'], interval=1e8)

find_unused_parameters = True

checkpoint_config = dict(interval=total_epochs)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
