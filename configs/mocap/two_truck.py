_base_ = [
    # '../_base_/models/detr.py',
    # '../../_base_/datasets/mot_challenge.py', 
    # '../../_base_/default_runtime.py',
     #'../../_base_/datasets/mot15-half.py', 
]
custom_imports = dict(
        imports=[
            'mmtrack.models.mocap.decoder',
            'mmtrack.models.mocap.single',
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
    img_model_cfg=dict(type='SingleModalityModel',
        backbone_cfg=img_backbone_cfg,
        neck_cfg=img_neck_cfg
    ),
    img_backbone_cfg=img_backbone_cfg,
    img_neck_cfg=img_neck_cfg,
    depth_backbone_cfg=depth_backbone_cfg,
    depth_neck_cfg=depth_neck_cfg,
    num_sa_layers=6,
    track_eval=False,
    mse_loss_weight=0.1,
    max_age=15
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

chunks = [
    (1656092267499, 1656092279279, 1179),
    (1656093005110, 1656093016490, 1139),
    (1656093841541, 1656093851531, 1000),
    (1656094219359, 1656094229349, 1000),
    (1656094457034, 1656094467024, 1000),
    (1656094678719, 1656094688709, 1000),
    (1656096536271, 1656096626261, 9000),
    (1656096636977, 1656096726967, 9000),
    (1656096735894, 1656096825884, 9000),
    (1656096834093, 1656096924083, 9000),
    (1656096932467, 1656097022457, 9000),
    (1656097031849, 1656097121839, 9000),
    (1656097129679, 1656097219669, 9000),
    (1656097228149, 1656097318139, 9000),
    (1656097329340, 1656097419330, 9000),
    (1656097432033, 1656097522023, 9000),
    (1656097532216, 1656097622206, 9000),
    (1656097630102, 1656097720092, 9000),
    (1656097741924, 1656097831914, 9000),
    (1656097870762, 1656097960752, 9000),
    (1656098036912, 1656098050262, 1336),
    (1656098099061, 1656098189051, 9000),
    (1656098197412, 1656098287402, 9000),
    (1656098295730, 1656098385720, 9000),
    (1656098396896, 1656098486886, 9000),
    (1656098495878, 1656098585868, 9000),
    (1656098594029, 1656098684019, 9000),
    (1656098691980, 1656098781970, 9000),
    (1656098793221, 1656098883211, 9000),
    (1656098891303, 1656098981293, 9000),
    (1656098990211, 1656099080201, 9000),
    (1656099333415, 1656099423405, 9000),
    (1656099433158, 1656099523148, 9000),
    (1656099531139, 1656099621129, 9000),
    (1656099629089, 1656099719079, 9000),
    (1656099727383, 1656099817373, 9000),
    (1656099825918, 1656099915908, 9000),
    (1656099962435, 1656100052425, 9000),
    (1656100060467, 1656100150457, 9000),
    (1656100198993, 1656100288983, 9000),
    (1656100296549, 1656100386539, 9000),
    (1656100395475, 1656100485465, 9000),
]

shuffle = True
classes = ('truck', )
valset=dict(type='HDF5Dataset',
    hdf5_fname='/home/csamplawski/eight/iobt/data_624/node_4_debug.hdf5',
    start_times=[chunks[30][0]],
    end_times=[chunks[30][1]],
    valid_keys=valid_keys,
    img_pipeline=img_pipeline,
    depth_pipeline=depth_pipeline,
    azimuth_pipeline=azimuth_pipeline,
    range_pipeline=range_pipeline,
    audio_pipeline=audio_pipeline,
    vid_path='logs/two_trucks/',
    is_random=False,
    remove_first_frame=True,
    max_len=None,
)

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=0,
    shuffle=shuffle,
    train=dict(type='HDF5Dataset',
        hdf5_fname='data/node_1_debug.hdf5',
        start_times=[chunks[25][0]],
        end_times=[chunks[25][1]],
        valid_keys=valid_keys,
        img_pipeline=img_pipeline,
        depth_pipeline=depth_pipeline,
        azimuth_pipeline=azimuth_pipeline,
        range_pipeline=range_pipeline,
        audio_pipeline=audio_pipeline,
        is_random=shuffle,
        remove_first_frame=True,
        max_len=None,
    ),
    val=valset,
    test=valset
)


optimizer = dict(
    type='AdamW',
    lr=1e-4*4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }
    )
)

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
total_epochs = 100
lr_config = dict(policy='step', step=[int(total_epochs * 0.8)])
#evaluation = dict(metric=['bbox', 'track'], interval=1, tmpdir='/home/csamplawski/logs/tmp')
evaluation = dict(metric=['bbox', 'track'], interval=total_epochs)

find_unused_parameters = True

checkpoint_config = dict(interval=total_epochs)
log_config = dict(
    interval=10,
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
