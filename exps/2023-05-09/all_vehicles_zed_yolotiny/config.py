_base_ = [
    '../../../configs/_base_/datasets/mmm/mocap_data.py'
]

img_norm_cfg = dict(mean=[0,0,0], std=[255,255,255], to_rgb=True)
img_pipeline = [
    dict(type='DecodeJPEG'),
    dict(type='LoadFromNumpyArray'),
    dict(type='Resize', img_scale=(480, 288), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

mocap_pipeline = [
    #dict(type='PruneObjects'),
    #dict(type='ScaleMocap', x_min=-2162.78244, y_min=-1637.84491),
    dict(type='ScaleMocap', x_min=-2500, y_min=-2500),
    dict(type='mm2cm'),
    dict(type='DropZ'),
    dict(type='PadObjects', pad_size=10)
]

pipelines = {
    'zed_camera_left': img_pipeline,
    'mocap': mocap_pipeline
}
trainset=dict(type='HDF5Dataset',
    pickle_paths=[
        '/dev/shm/All_vehicle_1_6D/train',
        '/dev/shm/Bus_empty_1_6D/train',
        '/dev/shm/Car_empty_1_6D/train',
    ],
    pipelines=pipelines
)

valset=[
    dict(type='HDF5Dataset',
        pickle_paths=['/dev/shm/All_vehicle_1_6D/val'],
        pipelines=pipelines
    ),
    dict(type='HDF5Dataset',
        pickle_paths=['/dev/shm/Bus_empty_1_6D/val'],
        pipelines=pipelines
    ),
    dict(type='HDF5Dataset',
        pickle_paths=['/dev/shm/Car_empty_1_6D/val'],
        pipelines=pipelines
    )
]

testset=[
    dict(type='HDF5Dataset',
        pickle_paths=['/dev/shm/All_vehicle_1_6D/test'],
        pipelines=pipelines
    ),
    dict(type='HDF5Dataset',
        pickle_paths=['/dev/shm/Bus_empty_1_6D/test'],
        pipelines=pipelines
    ),
    dict(type='HDF5Dataset',
        pickle_paths=['/dev/shm/Car_empty_1_6D/test'],
        pipelines=pipelines
    )
]


import os
work_dir = os.environ['WORK']
backbone_cfg=dict(type='YOLOv7', 
    weights=f'{work_dir}/src/mmtracking/checkpoints/yolov7-tiny.pt', 
    return_idx=1
)

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
        cov_add=50.0,
        input_dim=256,
        mlp_dropout_rate=0.0,
        interval_sizes=[25,25],
        binary_prob=True,
        scale_binary_prob=True
    ),
    entropy_loss_weight=1,
    entropy_loss_type='mse',
    pos_loss_weight=1,
)


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    shuffle=True, #trainset shuffle only
    train=trainset,
    val=valset,
    test=testset
)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
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
