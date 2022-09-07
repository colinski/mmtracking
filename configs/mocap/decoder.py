_base_ = [
    # '../_base_/models/detr.py',
    # '../../_base_/datasets/mot_challenge.py', 
    # '../../_base_/default_runtime.py',
     #'../../_base_/datasets/mot15-half.py', 
]
custom_imports = dict(
        imports=[
            'mmtrack.models.mocap.decoder',
            # 'mmtrack.models.trackers.trackformer_tracker'
        ],
        allow_failed_imports=False)

checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'  # noqa
img_backbone_cfg=dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    #out_indices=(0,1,2,3),
    out_indices=(3, ),
    frozen_stages=1,
    norm_cfg=dict(type='SyncBN', requires_grad=False),
    norm_eval=True,
    style='pytorch',
    init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint=checkpoint)
)

img_neck_cfg=dict(type='ChannelMapper',
    in_channels=[2048],
    kernel_size=1,
    out_channels=256,
    act_cfg=None,
    norm_cfg=dict(type='BN'),
    num_outs=1
)

depth_backbone_cfg=dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    #out_indices=(0,1,2,3),
    out_indices=(3, ),
    frozen_stages=1,
    norm_cfg=dict(type='SyncBN', requires_grad=False),
    norm_eval=True,
    style='pytorch',
    init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint=checkpoint)
)

depth_neck_cfg=dict(type='ChannelMapper',
    in_channels=[2048],
    kernel_size=1,
    out_channels=256,
    act_cfg=None,
    norm_cfg=dict(type='BN'),
    num_outs=1
)

model = dict(type='DecoderMocapModel',
    img_backbone_cfg=img_backbone_cfg,
    img_neck_cfg=img_neck_cfg,
    depth_backbone_cfg=depth_backbone_cfg,
    depth_neck_cfg=depth_neck_cfg
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=True)

img_pipeline = [
    dict(type='LoadFromNumpyArray'),
    dict(type='Resize', img_scale=(270, 480), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    # dict(type='ImageToTensor', keys=['img']),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]


depth_pipeline = [
    dict(type='LoadFromNumpyArray', force_float32=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', mean=[0], std=[20000], to_rgb=False),
    dict(type='Normalize', mean=[1], std=[0.5], to_rgb=False),
    # dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
    # dict(type='ImageToTensor', keys=['img']),
]

azimuth_pipeline = [
    dict(type='LoadFromNumpyArray', force_float32=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

range_pipeline = [
    dict(type='LoadFromNumpyArray', force_float32=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    # dict(type='Normalize', mean=[0], std=[20000], to_rgb=False),
    # dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

audio_pipeline = [
    dict(type='LoadAudio'),
    dict(type='LoadFromNumpyArray'),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]



# test_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(
        # type='MultiScaleFlipAug',
        # img_scale=(1280, 720),
        # flip=False,
        # transforms=[
            # dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            # dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            # dict(type='ImageToTensor', keys=['img']),
            # dict(type='VideoCollect', keys=['img'])
        # ])
# ]


# valid_keys=['mocap', 'zed_camera_left', 'zed_camera_depth', 
        # 'range_doppler', 'mic_waveform', 'realsense_camera_depth',
        # 'realsense_camera_img', 'azimuth_static']

#valid_keys=['mocap', 'zed_camera_left', 'zed_camera_depth']
# valid_keys=['mocap', 'zed_camera_depth']
valid_keys=['mocap', 'zed_camera_left']


# valid_keys=['mocap', 'zed_camera_left', 'zed_camera_depth', 
        # 'azimuth_static', 'range_doppler']

hdf5_fnames = ['1656096647489_1656096707489.hdf5',
               '1656096767489_1656096827489.hdf5',
               #'1656096827489_1656096887489.hdf5'
               # '1656096887489_1656096947489.hdf5'
               '1656096467489_1656096527489.hdf5']

start_times = [1656096536271, 1656096636977, 1656096735894, 1656096834093, 1656096932467, 1656097031849, 1656097129679, 1656097228149]
end_times   = [1656096626261, 1656096726967, 1656096825884, 1656096924083, 1656097022457, 1656097121839, 1656097219669, 1656097318139]

shuffle = True
classes = ('truck', )
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,
    shuffle=shuffle,
    train=dict(type='HDF5Dataset',
        hdf5_fname='data/node_1_debug.hdf5',
        start_times=[start_times[0]],
        end_times=[end_times[0]],
        valid_keys=valid_keys,
        img_pipeline=img_pipeline,
        depth_pipeline=depth_pipeline,
        azimuth_pipeline=azimuth_pipeline,
        range_pipeline=range_pipeline,
        audio_pipeline=audio_pipeline,
        is_random=shuffle
    ),
    val=dict(type='HDF5Dataset',
        hdf5_fname='data/node_1_debug.hdf5',
        start_times=[start_times[1]],
        end_times=[end_times[1]],
        # start_time=1656096735894,
        # end_time=1656096825884,
        valid_keys=valid_keys,
        img_pipeline=img_pipeline,
        depth_pipeline=depth_pipeline,
        azimuth_pipeline=azimuth_pipeline,
        range_pipeline=range_pipeline,
        audio_pipeline=audio_pipeline,
        vid_path='logs/'
    ),
    test=dict(type='HDF5Dataset',
        hdf5_fname='data/hdf5/' + hdf5_fnames[-1],
        valid_keys=valid_keys,
        img_pipeline=img_pipeline,
        depth_pipeline=depth_pipeline,
        azimuth_pipeline=azimuth_pipeline,
        range_pipeline=range_pipeline,
        audio_pipeline=audio_pipeline
    ),
)


optimizer = dict(
    type='AdamW',
    lr=1e-4,
    # lr=1e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
total_epochs = 100
lr_config = dict(policy='step', step=[int(total_epochs * 0.8)])
#evaluation = dict(metric=['bbox', 'track'], interval=1, tmpdir='/home/csamplawski/logs/tmp')
evaluation = dict(metric=['bbox', 'track'], interval=20)

find_unused_parameters = True

checkpoint_config = dict(interval=100)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
