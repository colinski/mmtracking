_base_ = [
    '../_base_/datasets/mmm/2022-09-01/trucks2_lightsT_obstaclesF.py'
]


valid_mods=['mocap', 'zed_camera_left', 'zed_camera_depth', 'zed_camera_left_r50',
            'range_doppler', 'azimuth_static', 'mic_waveform',
            'realsense_camera_depth', 'realsense_camera_img']

valid_nodes=[1,3]

# data_root = 'data/mmm/2022-09-01/trucks0_lightsT_obstaclesF/train'
trainset=dict(type='HDF5Dataset',
    name='train',
    uid=9234,
    num_future_frames=0,
    num_past_frames=9,
    valid_nodes=[1,3],
    valid_mods=['mocap', 'zed_camera_left_r50'],
    include_z=False,
)


# data_root = 'data/mmm/2022-09-01/trucks1_lightsT_obstaclesF/test'
valset=dict(type='HDF5Dataset',
    name='val',
    uid=4321,
    num_future_frames=0,
    num_past_frames=0,
    valid_mods=['mocap', 'zed_camera_left_r50', 'zed_camera_left'],
    valid_nodes=[1,3],
    limit_axis=True,
    draw_cov=True,
    include_z=False,
    max_len=500,
)

# data = dict(
    # samples_per_gpu=8,
    # workers_per_gpu=2,
    # shuffle=True, #trainset shuffle only
    # train=trainset,
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

model_cfgs = {('zed_camera_left_r50', 'node_1'): r50_model_cfg,
              ('zed_camera_left_r50', 'node_3'): r50_model_cfg}
backbone_cfgs = {'zed_camera_left_r50': r50_backbone_cfg}

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
    num_queries=2,
    add_grid_to_mean=False,
    autoregressive=True
    #match_by_id=True
)


orig_bs = 2
orig_lr = 1e-4 
factor = 4
data = dict(
    samples_per_gpu=orig_bs * factor,
    workers_per_gpu=2,
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
