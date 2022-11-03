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
            'mmtrack.models.mocap.oracle',
        ],
        allow_failed_imports=False)


checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'  # noqa


img_backbone_cfg=[
    dict(type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint=checkpoint)
    ),
    dict(type='ChannelMapper',
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='BN'),
        num_outs=1
    )
]

img_model_cfg=dict(type='SingleModalityModel', ffn_cfg=None)

model_cfgs = {('zed_camera_left', 'node_1'): img_model_cfg,
              ('zed_camera_left', 'node_4'): img_model_cfg}
backbone_cfgs = {'zed_camera_left': img_backbone_cfg}

model = dict(type='DecoderMocapModel',
    model_cfgs=model_cfgs,
    backbone_cfgs=backbone_cfgs,
    track_eval=True,
    mse_loss_weight=0.0,
    max_age=5,
    # init_cfg=dict(type='Pretrained', checkpoint='logs/single_truck_901_zed/latest.pth')
)

# model = dict(type='OracleModel', track_eval=True, no_update=False, mean_cov=[0.05,0.05,0.05], cov=[0.1,0.1,0.1], max_age=1e8)

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

valid_keys=['mocap', 'zed_camera_left']
# data_root = '/home/csamplawski/data/mmm/2022-09-01'
data_root = '/work/csamplawski_umass_edu/data/mmm/2022-09-01'
hdf5_fnames=[
    f'{data_root}/mocap.hdf5',
    f'{data_root}/node_1/zed.hdf5',
    # f'{data_root}/node_1/realsense.hdf5',
    # f'{data_root}/node_1/respeaker.hdf5',
    # f'{data_root}/node_1/mmwave.hdf5',
    # f'{data_root}/node_2/zed.hdf5',
    # f'{data_root}/node_2/realsense.hdf5',
    # f'{data_root}/node_2/respeaker.hdf5',
    # f'{data_root}/node_2/mmwave.hdf5',
    # f'{data_root}/node_3/zed.hdf5',
    # f'{data_root}/node_3/realsense.hdf5',
    # f'{data_root}/node_3/respeaker.hdf5',
    # f'{data_root}/node_3/mmwave.hdf5',
    f'{data_root}/node_4/zed.hdf5',
    # f'{data_root}/node_4/realsense.hdf5',
    # f'{data_root}/node_4/respeaker.hdf5',
    # f'{data_root}/node_4/mmwave.hdf5',
]

classes = ('truck', )
trainset=dict(type='HDF5Dataset',
    hdf5_fnames=hdf5_fnames,
    start_time=chunks[2][0],
    end_time=chunks[2][0] + int(2.5*60*1000),
    num_future_frames=0,
    num_past_frames=5,
    valid_keys=valid_keys,
    pipelines=pipelines,
    max_len=None,
),

valset=dict(type='HDF5Dataset',
    hdf5_fnames=hdf5_fnames,
    start_time=chunks[2][0] + int(2.5*60*1000),
    end_time=chunks[2][1],
    num_future_frames=0,
    num_past_frames=5,
    valid_keys=valid_keys,
    pipelines=pipelines,
    vid_path='exps/single_truck_node_14_5p0f/',
    max_len=-1,
    limit_axis=True,
    draw_cov=True,
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    shuffle=True, #trainset shuffle only
    train=trainset,
    val=valset,
    test=valset
)


optimizer = dict(
    type='AdamW',
    lr=1e-4,
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
