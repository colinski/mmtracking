data_root = '/home/csamplawski/data/mmm/2022-09-01'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_pipeline = [
    dict(type='DecodeJPEG'),
    dict(type='LoadFromNumpyArray'),
    dict(type='Resize', img_scale=(270, 480), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

r50_pipeline = [
    dict(type='LoadFromNumpyArray', force_float32=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

depth_pipeline = [
    dict(type='LoadFromNumpyArray', force_float32=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', mean=[0], std=[20000], to_rgb=False),
    dict(type='Normalize', mean=[1], std=[0.5], to_rgb=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

azimuth_pipeline = [
    dict(type='LoadFromNumpyArray', force_float32=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

range_pipeline = [
    dict(type='LoadFromNumpyArray', force_float32=True, transpose=True),
    dict(type='Resize', img_scale=(256, 16), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', mean=[4353], std=[705], to_rgb=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

audio_pipeline = [
    dict(type='LoadFromNumpyArray', force_float32=True, transpose=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

pipelines = {
    'zed_camera_left_r50': r50_pipeline,
    'zed_camera_left': img_pipeline,
    'zed_camera_depth': depth_pipeline,
    'azimuth_static': azimuth_pipeline,
    'range_doppler': range_pipeline,
    'mic_waveform': audio_pipeline,
    'realsense_camera_img': img_pipeline,
    'realsense_camera_depth': img_pipeline
}

chunks = [
    (1662065698934, 1662065756964), #0 5804
    (1662065858970, 1662065918960), #1 6000
    (1662065994888, 1662066294878), #2 30000
    (1662066459583, 1662066759573), #3 30000
    (1662066873706, 1662067173696), #4 30000
    (1662067271425, 1662067571415), #5 30000
    (1662067900609, 1662068200599), #6 30000
    (1662068331340, 1662068631330), #7 30000
    (1662068790675, 1662069090665), #8 30000
    (1662069405041, 1662069705031), #9 30000
]

# valid_keys=['mocap', 'zed_camera_left', 'zed_camera_depth', 
        # 'range_doppler', 'azimuth_static', 'mic_waveform',
        # 'realsense_camera_depth', 'realsense_camera_img']

# data_root = '/work/csamplawski_umass_edu/data/mmm/2022-09-01'

# classes = ('truck', )

trainset=dict(type='HDF5Dataset',
    # hdf5_fnames=[
        # f'{data_root}/mocap.hdf5',
        # f'{data_root}/node_1/zed_r50.hdf5',
        # f'{data_root}/node_2/zed_r50.hdf5',
        # f'{data_root}/node_3/zed_r50.hdf5',
        # f'{data_root}/node_4/zed_r50.hdf5',
    # ],
    start_time=chunks[2][0],
    end_time=chunks[2][0] + int(2.5*60*1000),
    # num_future_frames=0,
    # num_past_frames=5,
    # valid_keys=['mocap', 'zed_camera_left_r50'],
    pipelines=pipelines,
    # max_len=None,
)

valset=dict(type='HDF5Dataset',
    # hdf5_fnames=[
        # f'{data_root}/mocap.hdf5',
        # f'{data_root}/node_1/zed_r50.hdf5',
        # f'{data_root}/node_2/zed_r50.hdf5',
        # f'{data_root}/node_3/zed_r50.hdf5',
        # f'{data_root}/node_4/zed_r50.hdf5',
        # f'{data_root}/node_1/zed.hdf5',
        # f'{data_root}/node_2/zed.hdf5',
        # f'{data_root}/node_3/zed.hdf5',
        # f'{data_root}/node_4/zed.hdf5',
    # ],
    start_time=chunks[2][0] + int(2.5*60*1000),
    end_time=chunks[2][1],
    num_future_frames=0,
    num_past_frames=5,
    valid_keys=['mocap', 'zed_camera_left_r50', 'zed_camera_left'],
    pipelines=pipelines,
    vid_path='exps_r50/truck1_node123_r50/',
    max_len=500,
    limit_axis=True,
    draw_cov=True,
)

orig_bs = 2
orig_lr = 1e-4
factor = 4
data = dict(
    samples_per_gpu=orig_bs * factor,
    workers_per_gpu=0,
    shuffle=True, #trainset shuffle only
    train=trainset,
    val=valset,
    test=valset
)

optimizer = dict(
    type='AdamW',
    lr=orig_lr * factor,
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
total_epochs = 50
lr_config = dict(policy='step', step=[int(total_epochs * 0.8)])
#evaluation = dict(metric=['bbox', 'track'], interval=1, tmpdir='/home/csamplawski/logs/tmp')
evaluation = dict(metric=['bbox', 'track'], interval=1e8)

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
