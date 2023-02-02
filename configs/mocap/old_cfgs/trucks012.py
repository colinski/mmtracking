_base_ = [
]
custom_imports = dict(
        imports=[
            'mmtrack.models.mocap.decoder',
            'mmtrack.models.mocap.single',
            'mmtrack.models.mocap.oracle',
        ],
        allow_failed_imports=False)

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

valid_mods=['mocap', 'zed_camera_left', 'zed_camera_depth', 'zed_camera_left_r50',
            'range_doppler', 'azimuth_static', 'mic_waveform',
            'realsense_camera_depth', 'realsense_camera_img']

valid_nodes=[1,3]

data_root = 'data/mmm/2022-09-01/trucks0_lightsT_obstaclesF/train'
trainset0=dict(type='HDF5Dataset',
    hdf5_fnames=[
        f'{data_root}/mocap.hdf5',
        f'{data_root}/node_1/mmwave.hdf5',
        f'{data_root}/node_2/mmwave.hdf5',
        f'{data_root}/node_3/mmwave.hdf5',
        f'{data_root}/node_4/mmwave.hdf5',
        f'{data_root}/node_1/realsense.hdf5',
        f'{data_root}/node_2/realsense.hdf5',
        f'{data_root}/node_3/realsense.hdf5',
        f'{data_root}/node_4/realsense.hdf5',
        f'{data_root}/node_1/respeaker.hdf5',
        f'{data_root}/node_2/respeaker.hdf5',
        f'{data_root}/node_3/respeaker.hdf5',
        f'{data_root}/node_4/respeaker.hdf5',
        f'{data_root}/node_1/zed.hdf5',
        f'{data_root}/node_2/zed.hdf5',
        f'{data_root}/node_3/zed.hdf5',
        f'{data_root}/node_4/zed.hdf5',
        f'{data_root}/node_1/zed_r50.hdf5',
        f'{data_root}/node_2/zed_r50.hdf5',
        f'{data_root}/node_3/zed_r50.hdf5',
        f'{data_root}/node_4/zed_r50.hdf5',
    ],
    name='train',
    uid=9234,
    num_future_frames=0,
    num_past_frames=5,
    valid_mods=['mocap', 'zed_camera_left_r50'],
    valid_nodes=valid_nodes,
    pipelines=pipelines,
    include_z=False,
)

data_root = 'data/mmm/2022-09-01/trucks1_lightsT_obstaclesF/train'
trainset1=dict(type='HDF5Dataset',
    hdf5_fnames=[
        f'{data_root}/mocap.hdf5',
        f'{data_root}/node_1/mmwave.hdf5',
        f'{data_root}/node_2/mmwave.hdf5',
        f'{data_root}/node_3/mmwave.hdf5',
        f'{data_root}/node_4/mmwave.hdf5',
        f'{data_root}/node_1/realsense.hdf5',
        f'{data_root}/node_2/realsense.hdf5',
        f'{data_root}/node_3/realsense.hdf5',
        f'{data_root}/node_4/realsense.hdf5',
        f'{data_root}/node_1/respeaker.hdf5',
        f'{data_root}/node_2/respeaker.hdf5',
        f'{data_root}/node_3/respeaker.hdf5',
        f'{data_root}/node_4/respeaker.hdf5',
        f'{data_root}/node_1/zed.hdf5',
        f'{data_root}/node_2/zed.hdf5',
        f'{data_root}/node_3/zed.hdf5',
        f'{data_root}/node_4/zed.hdf5',
        f'{data_root}/node_1/zed_r50.hdf5',
        f'{data_root}/node_2/zed_r50.hdf5',
        f'{data_root}/node_3/zed_r50.hdf5',
        f'{data_root}/node_4/zed_r50.hdf5',
    ],
    name='train',
    uid=1234,
    num_future_frames=0,
    num_past_frames=5,
    valid_mods=['mocap', 'zed_camera_left_r50'],
    valid_nodes=valid_nodes,
    pipelines=pipelines,
    include_z=False,
)

data_root = 'data/mmm/2022-09-01/trucks2_lightsT_obstaclesF/train'
trainset2=dict(type='HDF5Dataset',
    hdf5_fnames=[
        f'{data_root}/mocap.hdf5',
        f'{data_root}/node_1/mmwave.hdf5',
        f'{data_root}/node_2/mmwave.hdf5',
        f'{data_root}/node_3/mmwave.hdf5',
        f'{data_root}/node_4/mmwave.hdf5',
        f'{data_root}/node_1/realsense.hdf5',
        f'{data_root}/node_2/realsense.hdf5',
        f'{data_root}/node_3/realsense.hdf5',
        f'{data_root}/node_4/realsense.hdf5',
        f'{data_root}/node_1/respeaker.hdf5',
        f'{data_root}/node_2/respeaker.hdf5',
        f'{data_root}/node_3/respeaker.hdf5',
        f'{data_root}/node_4/respeaker.hdf5',
        f'{data_root}/node_1/zed.hdf5',
        f'{data_root}/node_2/zed.hdf5',
        f'{data_root}/node_3/zed.hdf5',
        f'{data_root}/node_4/zed.hdf5',
        f'{data_root}/node_1/zed_r50.hdf5',
        f'{data_root}/node_2/zed_r50.hdf5',
        f'{data_root}/node_3/zed_r50.hdf5',
        f'{data_root}/node_4/zed_r50.hdf5',
    ],
    name='train',
    uid=2234,
    num_future_frames=0,
    num_past_frames=5,
    valid_mods=['mocap', 'zed_camera_left_r50'],
    valid_nodes=valid_nodes,
    pipelines=pipelines,
    include_z=False,
)


data_root = 'data/mmm/2022-09-01/trucks2_lightsT_obstaclesF/test'
valset=dict(type='HDF5Dataset',
    hdf5_fnames=[
        f'{data_root}/mocap.hdf5',
        f'{data_root}/node_1/mmwave.hdf5',
        f'{data_root}/node_2/mmwave.hdf5',
        f'{data_root}/node_3/mmwave.hdf5',
        f'{data_root}/node_4/mmwave.hdf5',
        f'{data_root}/node_1/realsense.hdf5',
        f'{data_root}/node_2/realsense.hdf5',
        f'{data_root}/node_3/realsense.hdf5',
        f'{data_root}/node_4/realsense.hdf5',
        f'{data_root}/node_1/respeaker.hdf5',
        f'{data_root}/node_2/respeaker.hdf5',
        f'{data_root}/node_3/respeaker.hdf5',
        f'{data_root}/node_4/respeaker.hdf5',
        f'{data_root}/node_1/zed.hdf5',
        f'{data_root}/node_2/zed.hdf5',
        f'{data_root}/node_3/zed.hdf5',
        f'{data_root}/node_4/zed.hdf5',
        f'{data_root}/node_1/zed_r50.hdf5',
        f'{data_root}/node_2/zed_r50.hdf5',
        f'{data_root}/node_3/zed_r50.hdf5',
        f'{data_root}/node_4/zed_r50.hdf5',
    ],
    name='val',
    uid=4321,
    num_future_frames=0,
    num_past_frames=5,
    valid_mods=['mocap', 'zed_camera_left_r50', 'zed_camera_left'],
    valid_nodes=valid_nodes,
    pipelines=pipelines,
    limit_axis=True,
    draw_cov=True,
    include_z=False,
    max_len=500,
)

# data = dict(
    # samples_per_gpu=1,
    # workers_per_gpu=2,
    # shuffle=True, #trainset shuffle only
    # train=[trainset0, trainset1, trainset2],
    # val=valset,
    # test=valset
# )

checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'  # noqa

r50_backbone_cfg=[
    dict(type='ChannelMapper',
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='BN'),
        num_outs=1
    ),
]

r50_model_cfg=dict(type='SingleModalityModel', ffn_cfg=None)
# r50_model_cfg=dict(type='DETRModalityModel')

model_cfgs = {('zed_camera_left_r50', 'node_1'): r50_model_cfg,
              ('zed_camera_left_r50', 'node_3'): r50_model_cfg}
backbone_cfgs = {'zed_camera_left_r50': r50_backbone_cfg}

# model = dict(type='DecoderMocapModelV2',
model = dict(type='DecoderMocapModel',
    model_cfgs=model_cfgs,
    backbone_cfgs=backbone_cfgs,
    track_eval=True,
    mse_loss_weight=0.0,
    max_age=5,
    grid_loss=True,
    include_z=False,
    mean_scale=[7,5],
    pos_loss_weight=0.1,
    predict_full_cov=True,
    #num_queries=5,
    add_grid_to_mean=True
)


# data_root = '/home/csamplawski/eight/data_901/trucks1_lightsT_obstaclesF/test'
# trainset0=dict(type='HDF5Dataset',
    # valid_mods=['mocap', 'zed_camera_left_r50'],
    # valid_nodes=[1, 3],
    # uid=9234,
# )

# trainset1=dict(type='HDF5Dataset',
    # valid_mods=['mocap', 'zed_camera_left_r50'],
    # valid_nodes=[1, 3],
    # uid=1234,
# )

# trainset2=dict(type='HDF5Dataset',
    # valid_mods=['mocap', 'zed_camera_left_r50'],
    # valid_nodes=[1, 3],
    # uid=2234,
# )

# valset=dict(type='HDF5Dataset',
    # valid_mods=['mocap', 'zed_camera_left', 'zed_camera_left_r50'],
    # valid_nodes=[1, 3],
# )



# valset=dict(type='HDF5Dataset',
    # hdf5_fnames=[
        # f'{data_root}/mocap.hdf5',
        # f'{data_root}/node_1/zed.hdf5',
        # f'{data_root}/node_1/zed_r50.hdf5',
        # f'{data_root}/node_3/zed.hdf5',
        # f'{data_root}/node_3/zed_r50.hdf5',
    # ],
    # name='val',
    # uid=4321,
    # start_time=chunks[2][0] + int(2.5*60*1000),
    # end_time=chunks[2][1],
    # num_future_frames=0,
    # num_past_frames=5,
    # valid_keys=['mocap', 'zed_camera_left_r50', 'zed_camera_left'],
    # pipelines=pipelines,
    # max_len=50,
    # limit_axis=True,
    # draw_cov=True,
    # include_z=False,
# )

orig_bs = 2
orig_lr = 1e-4 
factor = 4
data = dict(
    samples_per_gpu=orig_bs * factor,
    workers_per_gpu=0,
    shuffle=True, #trainset shuffle only
    train=[trainset0, trainset1, trainset2],
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
#lr_config = dict(policy='step', step=[int(total_epochs * 0.8)], warmup='linear', warmup_iters=100)
#evaluation = dict(metric=['bbox', 'track'], interval=1, tmpdir='/home/csamplawski/logs/tmp')
evaluation = dict(metric=['bbox', 'track'], interval=1e8)

find_unused_parameters = True

checkpoint_config = dict(interval=total_epochs)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='TensorboardLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
