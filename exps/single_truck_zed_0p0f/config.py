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
            'mmtrack.models.mocap.bg_model'
            # 'mmtrack.models.trackers.trackformer_tracker'
        ],
        allow_failed_imports=False)

checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'  # noqa
img_backbone_cfg=dict(
    type='ResNet',
    depth=50,
    num_stages=4,
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


# bg_model_cfg=dict(type='BackgroundModel', 
    # init_cfg=dict(type='Pretrained', 
        # checkpoint='logs/bg_model_zed/latest.pth'
    # )
# )

model = dict(type='DecoderMocapModel',
    img_model_cfg=dict(type='SingleModalityModel',
        backbone_cfg=img_backbone_cfg,
        neck_cfg=img_neck_cfg,
        ffn_cfg=None,
    ),
    time_attn_cfg=None,
    track_eval=False,
    mse_loss_weight=0.0,
    max_age=5,
    # init_cfg=dict(type='Pretrained', checkpoint='logs/single_truck_901_zed/latest.pth')
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

valid_keys=['mocap', 'zed_camera_left']

classes = ('truck', )
data_root = '/home/csamplawski/data/'
valset=dict(type='HDF5Dataset',
    hdf5_fname=f'{data_root}/data_901_node_1.hdf5',
    num_past_frames=0,
    num_future_frames=0,
    start_times=[chunks[2][0] + int(2.5*60*1000)],
    end_times=[chunks[2][1]],
    valid_keys=valid_keys,
    img_pipeline=img_pipeline,
    vid_path='exps/single_truck_zed_0p0f/',
    is_random=False,
    remove_first_frame=True,
    max_len=10,
)

shuffle = True
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=0,
    shuffle=shuffle,
    train=dict(type='HDF5Dataset',
        num_past_frames=0,
        num_future_frames=0,
        hdf5_fname=f'{data_root}/data_901_node_1.hdf5',
        start_times=[chunks[2][0]],
        end_times=[chunks[2][0] + int(2.5*60*1000)],
        valid_keys=valid_keys,
        img_pipeline=img_pipeline,
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
total_epochs = 50
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
