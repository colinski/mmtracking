img_norm_cfg = dict(mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)
img_pipeline = [
    dict(type='DecodeJPEG'),
    dict(type='LoadFromNumpyArray'),
    dict(type='Resize', img_scale=(480, 288), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True),
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
    dict(type='LoadFromNumpyArray', force_float32=True, transpose=True),
    dict(type='Resize', img_scale=(256, 16), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', mean=[4353], std=[705], to_rgb=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'])
]
audio_pipeline = [
    dict(type='LoadFromNumpyArray', force_float32=True, transpose=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'])
]
pipelines = dict(
    zed_camera_left=[
        dict(type='DecodeJPEG'),
        dict(type='LoadFromNumpyArray'),
        dict(type='Resize', img_scale=(480, 288), keep_ratio=False),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(
            type='Normalize', mean=[0, 0, 0], std=[255, 255, 255],
            to_rgb=True),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img'])
    ],
    zed_camera_depth=[
        dict(type='LoadFromNumpyArray', force_float32=True),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='Normalize', mean=[0], std=[20000], to_rgb=False),
        dict(type='Normalize', mean=[1], std=[0.5], to_rgb=False),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img'])
    ],
    azimuth_static=[
        dict(type='LoadFromNumpyArray', force_float32=True),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img'])
    ],
    range_doppler=[
        dict(type='LoadFromNumpyArray', force_float32=True, transpose=True),
        dict(type='Resize', img_scale=(256, 16), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='Normalize', mean=[4353], std=[705], to_rgb=False),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img'])
    ],
    mic_waveform=[
        dict(type='LoadFromNumpyArray', force_float32=True, transpose=True),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img'])
    ],
    realsense_camera_img=[
        dict(type='DecodeJPEG'),
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
    realsense_camera_depth=[
        dict(type='DecodeJPEG'),
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
    mocap=[
        dict(type='ScaleMocap', x_min=-2162.78244, y_min=-1637.84491),
        dict(type='mm2cm'),
        dict(type='DropZ'),
        dict(type='PadObjects', pad_size=6)
    ])
trainset = dict(
    type='HDF5Dataset',
    pipelines=dict(
        zed_camera_left=[
            dict(type='DecodeJPEG'),
            dict(type='LoadFromNumpyArray'),
            dict(type='Resize', img_scale=(480, 288), keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[0, 0, 0],
                std=[255, 255, 255],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        zed_camera_depth=[
            dict(type='LoadFromNumpyArray', force_float32=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', mean=[0], std=[20000], to_rgb=False),
            dict(type='Normalize', mean=[1], std=[0.5], to_rgb=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        azimuth_static=[
            dict(type='LoadFromNumpyArray', force_float32=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        range_doppler=[
            dict(
                type='LoadFromNumpyArray', force_float32=True, transpose=True),
            dict(type='Resize', img_scale=(256, 16), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', mean=[4353], std=[705], to_rgb=False),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        mic_waveform=[
            dict(
                type='LoadFromNumpyArray', force_float32=True, transpose=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ],
        realsense_camera_img=[
            dict(type='DecodeJPEG'),
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
        realsense_camera_depth=[
            dict(type='DecodeJPEG'),
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
        mocap=[
            dict(type='ScaleMocap', x_min=-2162.78244, y_min=-1637.84491),
            dict(type='mm2cm'),
            dict(type='DropZ'),
            dict(type='PadObjects', pad_size=6)
        ]),
    pickle_paths=[
        '/dev/shm/trucks1_lightsT_obstaclesF/train',
        '/dev/shm/trucks2_lightsT_obstaclesF/train'
    ])
mocap_pipeline = [
    dict(type='ScaleMocap', x_min=-2162.78244, y_min=-1637.84491),
    dict(type='mm2cm'),
    dict(type='DropZ'),
    dict(type='PadObjects', pad_size=6)
]
valset = [
    dict(
        type='HDF5Dataset',
        pickle_paths=['/dev/shm/trucks1_lightsT_obstaclesF/val'],
        pipelines=dict(
            zed_camera_left=[
                dict(type='DecodeJPEG'),
                dict(type='LoadFromNumpyArray'),
                dict(type='Resize', img_scale=(480, 288), keep_ratio=False),
                dict(type='RandomFlip', flip_ratio=0.0),
                dict(
                    type='Normalize',
                    mean=[0, 0, 0],
                    std=[255, 255, 255],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img'])
            ],
            mocap=[
                dict(type='ScaleMocap', x_min=-2162.78244, y_min=-1637.84491),
                dict(type='mm2cm'),
                dict(type='DropZ'),
                dict(type='PadObjects', pad_size=6)
            ])),
    dict(
        type='HDF5Dataset',
        pickle_paths=['/dev/shm/trucks2_lightsT_obstaclesF/val'],
        pipelines=dict(
            zed_camera_left=[
                dict(type='DecodeJPEG'),
                dict(type='LoadFromNumpyArray'),
                dict(type='Resize', img_scale=(480, 288), keep_ratio=False),
                dict(type='RandomFlip', flip_ratio=0.0),
                dict(
                    type='Normalize',
                    mean=[0, 0, 0],
                    std=[255, 255, 255],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img'])
            ],
            mocap=[
                dict(type='ScaleMocap', x_min=-2162.78244, y_min=-1637.84491),
                dict(type='mm2cm'),
                dict(type='DropZ'),
                dict(type='PadObjects', pad_size=6)
            ]))
]
testset = [
    dict(
        type='HDF5Dataset',
        pickle_paths=['/dev/shm/trucks1_lightsT_obstaclesF/test'],
        pipelines=dict(
            zed_camera_left=[
                dict(type='DecodeJPEG'),
                dict(type='LoadFromNumpyArray'),
                dict(type='Resize', img_scale=(480, 288), keep_ratio=False),
                dict(type='RandomFlip', flip_ratio=0.0),
                dict(
                    type='Normalize',
                    mean=[0, 0, 0],
                    std=[255, 255, 255],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img'])
            ],
            mocap=[
                dict(type='ScaleMocap', x_min=-2162.78244, y_min=-1637.84491),
                dict(type='mm2cm'),
                dict(type='DropZ'),
                dict(type='PadObjects', pad_size=6)
            ])),
    dict(
        type='HDF5Dataset',
        pickle_paths=['/dev/shm/trucks2_lightsT_obstaclesF/test'],
        pipelines=dict(
            zed_camera_left=[
                dict(type='DecodeJPEG'),
                dict(type='LoadFromNumpyArray'),
                dict(type='Resize', img_scale=(480, 288), keep_ratio=False),
                dict(type='RandomFlip', flip_ratio=0.0),
                dict(
                    type='Normalize',
                    mean=[0, 0, 0],
                    std=[255, 255, 255],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img'])
            ],
            mocap=[
                dict(type='ScaleMocap', x_min=-2162.78244, y_min=-1637.84491),
                dict(type='mm2cm'),
                dict(type='DropZ'),
                dict(type='PadObjects', pad_size=6)
            ]))
]
work_dir = '/work/csamplawski_umass_edu/src/mmtracking/exps/2023-05-01/trucks12/log'
backbone_cfg = dict(
    type='YOLOv7',
    weights=
    '/work/csamplawski_umass_edu//src/mmtracking/checkpoints/yolov7-tiny.pt',
    return_idx=1)
adapter_cfg = dict(
    type='ConvAdapter', interpolate_size=(28, 20), interpolate_fn='avgpool')
adapter_cfgs = dict({
    ('zed_camera_left', 'node_1'):
    dict(
        type='ConvAdapter',
        interpolate_size=(28, 20),
        interpolate_fn='avgpool'),
    ('zed_camera_left', 'node_2'):
    dict(
        type='ConvAdapter',
        interpolate_size=(28, 20),
        interpolate_fn='avgpool'),
    ('zed_camera_left', 'node_3'):
    dict(
        type='ConvAdapter',
        interpolate_size=(28, 20),
        interpolate_fn='avgpool'),
    ('zed_camera_left', 'node_4'):
    dict(
        type='ConvAdapter',
        interpolate_size=(28, 20),
        interpolate_fn='avgpool')
})
backbone_cfgs = dict(
    zed_camera_left=dict(
        type='YOLOv7',
        weights=
        '/work/csamplawski_umass_edu//src/mmtracking/checkpoints/yolov7-tiny.pt',
        return_idx=1))
model = dict(
    type='DetectorEnsemble',
    backbone_cfgs=dict(
        zed_camera_left=dict(
            type='YOLOv7',
            weights=
            '/work/csamplawski_umass_edu//src/mmtracking/checkpoints/yolov7-tiny.pt',
            return_idx=1)),
    adapter_cfgs=dict({
        ('zed_camera_left', 'node_1'):
        dict(
            type='ConvAdapter',
            interpolate_size=(28, 20),
            interpolate_fn='avgpool'),
        ('zed_camera_left', 'node_2'):
        dict(
            type='ConvAdapter',
            interpolate_size=(28, 20),
            interpolate_fn='avgpool'),
        ('zed_camera_left', 'node_3'):
        dict(
            type='ConvAdapter',
            interpolate_size=(28, 20),
            interpolate_fn='avgpool'),
        ('zed_camera_left', 'node_4'):
        dict(
            type='ConvAdapter',
            interpolate_size=(28, 20),
            interpolate_fn='avgpool')
    }),
    output_head_cfg=dict(
        type='AnchorOutputHead',
        include_z=False,
        cov_add=50.0,
        input_dim=256,
        mlp_dropout_rate=0.0,
        interval_sizes=[25, 25],
        binary_prob=True,
        scale_binary_prob=True),
    entropy_loss_weight=1,
    entropy_loss_type='mse',
    pos_loss_weight=1)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    shuffle=True,
    train=dict(
        type='HDF5Dataset',
        pickle_paths=[
            '/dev/shm/trucks1_lightsT_obstaclesF/train',
            '/dev/shm/trucks2_lightsT_obstaclesF/train'
        ],
        pipelines=dict(
            zed_camera_left=[
                dict(type='DecodeJPEG'),
                dict(type='LoadFromNumpyArray'),
                dict(type='Resize', img_scale=(480, 288), keep_ratio=False),
                dict(type='RandomFlip', flip_ratio=0.0),
                dict(
                    type='Normalize',
                    mean=[0, 0, 0],
                    std=[255, 255, 255],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img'])
            ],
            mocap=[
                dict(type='ScaleMocap', x_min=-2162.78244, y_min=-1637.84491),
                dict(type='mm2cm'),
                dict(type='DropZ'),
                dict(type='PadObjects', pad_size=6)
            ])),
    val=[
        dict(
            type='HDF5Dataset',
            pickle_paths=['/dev/shm/trucks1_lightsT_obstaclesF/val'],
            pipelines=dict(
                zed_camera_left=[
                    dict(type='DecodeJPEG'),
                    dict(type='LoadFromNumpyArray'),
                    dict(
                        type='Resize', img_scale=(480, 288), keep_ratio=False),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[255, 255, 255],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ],
                mocap=[
                    dict(
                        type='ScaleMocap',
                        x_min=-2162.78244,
                        y_min=-1637.84491),
                    dict(type='mm2cm'),
                    dict(type='DropZ'),
                    dict(type='PadObjects', pad_size=6)
                ])),
        dict(
            type='HDF5Dataset',
            pickle_paths=['/dev/shm/trucks2_lightsT_obstaclesF/val'],
            pipelines=dict(
                zed_camera_left=[
                    dict(type='DecodeJPEG'),
                    dict(type='LoadFromNumpyArray'),
                    dict(
                        type='Resize', img_scale=(480, 288), keep_ratio=False),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[255, 255, 255],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ],
                mocap=[
                    dict(
                        type='ScaleMocap',
                        x_min=-2162.78244,
                        y_min=-1637.84491),
                    dict(type='mm2cm'),
                    dict(type='DropZ'),
                    dict(type='PadObjects', pad_size=6)
                ]))
    ],
    test=[
        dict(
            type='HDF5Dataset',
            pickle_paths=['/dev/shm/trucks1_lightsT_obstaclesF/test'],
            pipelines=dict(
                zed_camera_left=[
                    dict(type='DecodeJPEG'),
                    dict(type='LoadFromNumpyArray'),
                    dict(
                        type='Resize', img_scale=(480, 288), keep_ratio=False),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[255, 255, 255],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ],
                mocap=[
                    dict(
                        type='ScaleMocap',
                        x_min=-2162.78244,
                        y_min=-1637.84491),
                    dict(type='mm2cm'),
                    dict(type='DropZ'),
                    dict(type='PadObjects', pad_size=6)
                ])),
        dict(
            type='HDF5Dataset',
            pickle_paths=['/dev/shm/trucks2_lightsT_obstaclesF/test'],
            pipelines=dict(
                zed_camera_left=[
                    dict(type='DecodeJPEG'),
                    dict(type='LoadFromNumpyArray'),
                    dict(
                        type='Resize', img_scale=(480, 288), keep_ratio=False),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[0, 0, 0],
                        std=[255, 255, 255],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ],
                mocap=[
                    dict(
                        type='ScaleMocap',
                        x_min=-2162.78244,
                        y_min=-1637.84491),
                    dict(type='mm2cm'),
                    dict(type='DropZ'),
                    dict(type='PadObjects', pad_size=6)
                ]))
    ])
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
total_epochs = 50
lr_config = dict(policy='step', step=[40])
evaluation = dict(metric=['bbox', 'track'], interval=100000000.0)
find_unused_parameters = True
checkpoint_config = dict(interval=50)
log_config = dict(
    interval=1,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
gpu_ids = [0]
