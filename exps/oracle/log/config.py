custom_imports = dict(
    imports=[
        'mmtrack.models.mocap.decoder', 'mmtrack.models.mocap.single',
        'mmtrack.models.mocap.oracle'
    ],
    allow_failed_imports=False)
model = dict(
    type='OracleModel',
    track_eval=True,
    no_update=True,
    cov=[0.1, 0.1, 0.1],
    max_age=100000000.0)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_pipeline = [
    dict(type='LoadFromNumpyArray'),
    dict(type='Resize', img_scale=(270, 480), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'])
]
depth_pipeline = [
    dict(type='LoadFromNumpyArray', force_float32=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', mean=[0], std=[20000], to_rgb=False),
    dict(type='Normalize', mean=[1], std=[0.5], to_rgb=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'])
]
azimuth_pipeline = [
    dict(type='LoadFromNumpyArray', force_float32=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'])
]
range_pipeline = [
    dict(type='LoadFromNumpyArray', force_float32=True),
    dict(type='Resize', img_scale=(256, 16), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', mean=[4353], std=[705], to_rgb=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'])
]
audio_pipeline = [
    dict(type='LoadAudio'),
    dict(type='LoadFromNumpyArray'),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'])
]
chunks = [(1662065698934, 1662065756964), (1662065858970, 1662065918960),
          (1662065994888, 1662066294878), (1662066459583, 1662066759573),
          (1662066873706, 1662067173696), (1662067271425, 1662067571415),
          (1662067900609, 1662068200599), (1662068331340, 1662068631330),
          (1662068790675, 1662069090665), (1662069405041, 1662069705031)]
valid_keys = ['mocap', 'zed_camera_left']
shuffle = True
classes = ('truck', )
data_root = '/home/csamplawski/data'
valset = dict(
    type='HDF5Dataset',
    hdf5_fname='/home/csamplawski/data/data_901_node_1.hdf5',
    start_times=[1662065994888],
    end_times=[1662066294878],
    valid_keys=['mocap', 'zed_camera_left'],
    img_pipeline=[
        dict(type='LoadFromNumpyArray'),
        dict(type='Resize', img_scale=(270, 480), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img'])
    ],
    depth_pipeline=[
        dict(type='LoadFromNumpyArray', force_float32=True),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='Normalize', mean=[0], std=[20000], to_rgb=False),
        dict(type='Normalize', mean=[1], std=[0.5], to_rgb=False),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img'])
    ],
    azimuth_pipeline=[
        dict(type='LoadFromNumpyArray', force_float32=True),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img'])
    ],
    range_pipeline=[
        dict(type='LoadFromNumpyArray', force_float32=True),
        dict(type='Resize', img_scale=(256, 16), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='Normalize', mean=[4353], std=[705], to_rgb=False),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img'])
    ],
    audio_pipeline=[
        dict(type='LoadAudio'),
        dict(type='LoadFromNumpyArray'),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img'])
    ],
    vid_path='exps/oracle/',
    is_random=False,
    remove_first_frame=True,
    max_len=50,
    limit_axis=False,
    draw_cov=True)
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=0,
    shuffle=True,
    train=dict(
        type='HDF5Dataset',
        hdf5_fname='/home/csamplawski/data/data_901_node_1.hdf5',
        start_times=[1662065994888],
        end_times=[1662066294878],
        valid_keys=['mocap', 'zed_camera_left'],
        img_pipeline=[
            dict(type='LoadFromNumpyArray'),
            dict(type='Resize', img_scale=(270, 480), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        depth_pipeline=[
            dict(type='LoadFromNumpyArray', force_float32=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', mean=[0], std=[20000], to_rgb=False),
            dict(type='Normalize', mean=[1], std=[0.5], to_rgb=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        azimuth_pipeline=[
            dict(type='LoadFromNumpyArray', force_float32=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        range_pipeline=[
            dict(type='LoadFromNumpyArray', force_float32=True),
            dict(type='Resize', img_scale=(256, 16), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', mean=[4353], std=[705], to_rgb=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        audio_pipeline=[
            dict(type='LoadAudio'),
            dict(type='LoadFromNumpyArray'),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        is_random=True,
        remove_first_frame=True,
        max_len=None),
    val=dict(
        type='HDF5Dataset',
        hdf5_fname='/home/csamplawski/data/data_901_node_1.hdf5',
        start_times=[1662065994888],
        end_times=[1662066294878],
        valid_keys=['mocap', 'zed_camera_left'],
        img_pipeline=[
            dict(type='LoadFromNumpyArray'),
            dict(type='Resize', img_scale=(270, 480), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        depth_pipeline=[
            dict(type='LoadFromNumpyArray', force_float32=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', mean=[0], std=[20000], to_rgb=False),
            dict(type='Normalize', mean=[1], std=[0.5], to_rgb=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        azimuth_pipeline=[
            dict(type='LoadFromNumpyArray', force_float32=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        range_pipeline=[
            dict(type='LoadFromNumpyArray', force_float32=True),
            dict(type='Resize', img_scale=(256, 16), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', mean=[4353], std=[705], to_rgb=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        audio_pipeline=[
            dict(type='LoadAudio'),
            dict(type='LoadFromNumpyArray'),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        vid_path='exps/oracle/',
        is_random=False,
        remove_first_frame=True,
        max_len=50,
        limit_axis=False,
        draw_cov=True),
    test=dict(
        type='HDF5Dataset',
        hdf5_fname='/home/csamplawski/data/data_901_node_1.hdf5',
        start_times=[1662065994888],
        end_times=[1662066294878],
        valid_keys=['mocap', 'zed_camera_left'],
        img_pipeline=[
            dict(type='LoadFromNumpyArray'),
            dict(type='Resize', img_scale=(270, 480), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        depth_pipeline=[
            dict(type='LoadFromNumpyArray', force_float32=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', mean=[0], std=[20000], to_rgb=False),
            dict(type='Normalize', mean=[1], std=[0.5], to_rgb=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        azimuth_pipeline=[
            dict(type='LoadFromNumpyArray', force_float32=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        range_pipeline=[
            dict(type='LoadFromNumpyArray', force_float32=True),
            dict(type='Resize', img_scale=(256, 16), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', mean=[4353], std=[705], to_rgb=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        audio_pipeline=[
            dict(type='LoadAudio'),
            dict(type='LoadFromNumpyArray'),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        vid_path='exps/oracle/',
        is_random=False,
        remove_first_frame=True,
        max_len=50,
        limit_axis=False,
        draw_cov=True))
optimizer = dict(
    type='AdamW',
    lr=0.0004,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1))))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
total_epochs = 1
lr_config = None
evaluation = dict(metric=['bbox', 'track'], interval=100000000.0)
find_unused_parameters = True
checkpoint_config = dict(interval=1)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
work_dir = 'exps/oracle/log'
gpu_ids = [0]
