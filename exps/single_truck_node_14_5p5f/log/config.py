custom_imports = dict(
    imports=[
        'mmtrack.models.mocap.decoder', 'mmtrack.models.mocap.single',
        'mmtrack.models.mocap.oracle'
    ],
    allow_failed_imports=False)
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'
img_backbone_cfg = [
    dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'
        )),
    dict(
        type='ChannelMapper',
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='BN'),
        num_outs=1)
]
img_model_cfg = dict(type='SingleModalityModel', ffn_cfg=None)
model_cfgs = dict({
    ('zed_camera_left', 'node_1'):
    dict(type='SingleModalityModel', ffn_cfg=None),
    ('zed_camera_left', 'node_4'):
    dict(type='SingleModalityModel', ffn_cfg=None)
})
backbone_cfgs = dict(zed_camera_left=[
    dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'
        )),
    dict(
        type='ChannelMapper',
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='BN'),
        num_outs=1)
])
model = dict(
    type='DecoderMocapModel',
    model_cfgs=dict({
        ('zed_camera_left', 'node_1'):
        dict(type='SingleModalityModel', ffn_cfg=None),
        ('zed_camera_left', 'node_4'):
        dict(type='SingleModalityModel', ffn_cfg=None)
    }),
    backbone_cfgs=dict(zed_camera_left=[
        dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            frozen_stages=1,
            norm_cfg=dict(type='SyncBN', requires_grad=False),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained',
                prefix='backbone.',
                checkpoint=
                'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth'
            )),
        dict(
            type='ChannelMapper',
            in_channels=[2048],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='BN'),
            num_outs=1)
    ]),
    track_eval=False,
    mse_loss_weight=0.0,
    max_age=5)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_pipeline = [
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
    ])
chunks = [(1662065698934, 1662065756964), (1662065858970, 1662065918960),
          (1662065994888, 1662066294878), (1662066459583, 1662066759573),
          (1662066873706, 1662067173696), (1662067271425, 1662067571415),
          (1662067900609, 1662068200599), (1662068331340, 1662068631330),
          (1662068790675, 1662069090665), (1662069405041, 1662069705031)]
valid_keys = ['mocap', 'zed_camera_left']
data_root = '/home/csamplawski/data/mmm/2022-09-01'
hdf5_fnames = [
    '/home/csamplawski/data/mmm/2022-09-01/mocap.hdf5',
    '/home/csamplawski/data/mmm/2022-09-01/node_1/zed.hdf5',
    '/home/csamplawski/data/mmm/2022-09-01/node_4/zed.hdf5'
]
classes = ('truck', )
trainset = ({
    'type':
    'HDF5Dataset',
    'hdf5_fnames': [
        '/home/csamplawski/data/mmm/2022-09-01/mocap.hdf5',
        '/home/csamplawski/data/mmm/2022-09-01/node_1/zed.hdf5',
        '/home/csamplawski/data/mmm/2022-09-01/node_4/zed.hdf5'
    ],
    'start_time':
    1662065994888,
    'end_time':
    1662066144888,
    'num_future_frames':
    5,
    'num_past_frames':
    5,
    'valid_keys': ['mocap', 'zed_camera_left'],
    'pipelines': {
        'zed_camera_left': [{
            'type': 'DecodeJPEG'
        }, {
            'type': 'LoadFromNumpyArray'
        }, {
            'type': 'Resize',
            'img_scale': (270, 480),
            'keep_ratio': True
        }, {
            'type': 'RandomFlip',
            'flip_ratio': 0.0
        }, {
            'type': 'Normalize',
            'mean': [123.675, 116.28, 103.53],
            'std': [58.395, 57.12, 57.375],
            'to_rgb': True
        }, {
            'type': 'DefaultFormatBundle'
        }, {
            'type': 'Collect',
            'keys': ['img']
        }],
        'zed_camera_depth': [{
            'type': 'LoadFromNumpyArray',
            'force_float32': True
        }, {
            'type': 'RandomFlip',
            'flip_ratio': 0.0
        }, {
            'type': 'Normalize',
            'mean': [0],
            'std': [20000],
            'to_rgb': False
        }, {
            'type': 'Normalize',
            'mean': [1],
            'std': [0.5],
            'to_rgb': False
        }, {
            'type': 'DefaultFormatBundle'
        }, {
            'type': 'Collect',
            'keys': ['img']
        }],
        'azimuth_static': [{
            'type': 'LoadFromNumpyArray',
            'force_float32': True
        }, {
            'type': 'RandomFlip',
            'flip_ratio': 0.0
        }, {
            'type': 'DefaultFormatBundle'
        }, {
            'type': 'Collect',
            'keys': ['img']
        }],
        'range_doppler': [{
            'type': 'LoadFromNumpyArray',
            'force_float32': True,
            'transpose': True
        }, {
            'type': 'Resize',
            'img_scale': (256, 16),
            'keep_ratio': True
        }, {
            'type': 'RandomFlip',
            'flip_ratio': 0.0
        }, {
            'type': 'Normalize',
            'mean': [4353],
            'std': [705],
            'to_rgb': False
        }, {
            'type': 'DefaultFormatBundle'
        }, {
            'type': 'Collect',
            'keys': ['img']
        }],
        'mic_waveform': [{
            'type': 'LoadFromNumpyArray',
            'force_float32': True,
            'transpose': True
        }, {
            'type': 'RandomFlip',
            'flip_ratio': 0.0
        }, {
            'type': 'DefaultFormatBundle'
        }, {
            'type': 'Collect',
            'keys': ['img']
        }],
        'realsense_camera_img': [{
            'type': 'DecodeJPEG'
        }, {
            'type': 'LoadFromNumpyArray'
        }, {
            'type': 'Resize',
            'img_scale': (270, 480),
            'keep_ratio': True
        }, {
            'type': 'RandomFlip',
            'flip_ratio': 0.0
        }, {
            'type': 'Normalize',
            'mean': [123.675, 116.28, 103.53],
            'std': [58.395, 57.12, 57.375],
            'to_rgb': True
        }, {
            'type': 'DefaultFormatBundle'
        }, {
            'type': 'Collect',
            'keys': ['img']
        }],
        'realsense_camera_depth': [{
            'type': 'DecodeJPEG'
        }, {
            'type': 'LoadFromNumpyArray'
        }, {
            'type': 'Resize',
            'img_scale': (270, 480),
            'keep_ratio': True
        }, {
            'type': 'RandomFlip',
            'flip_ratio': 0.0
        }, {
            'type': 'Normalize',
            'mean': [123.675, 116.28, 103.53],
            'std': [58.395, 57.12, 57.375],
            'to_rgb': True
        }, {
            'type': 'DefaultFormatBundle'
        }, {
            'type': 'Collect',
            'keys': ['img']
        }]
    },
    'max_len':
    None
}, )
valset = dict(
    type='HDF5Dataset',
    hdf5_fnames=[
        '/home/csamplawski/data/mmm/2022-09-01/mocap.hdf5',
        '/home/csamplawski/data/mmm/2022-09-01/node_1/zed.hdf5',
        '/home/csamplawski/data/mmm/2022-09-01/node_4/zed.hdf5'
    ],
    start_time=1662066144888,
    end_time=1662066294878,
    num_future_frames=0,
    num_past_frames=5,
    valid_keys=['mocap', 'zed_camera_left'],
    pipelines=dict(
        zed_camera_left=[
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
        ]),
    vid_path='exps/single_truck_node_4_5p5f/',
    max_len=500,
    limit_axis=True,
    draw_cov=True)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    shuffle=True,
    train=({
        'type':
        'HDF5Dataset',
        'hdf5_fnames': [
            '/home/csamplawski/data/mmm/2022-09-01/mocap.hdf5',
            '/home/csamplawski/data/mmm/2022-09-01/node_1/zed.hdf5',
            '/home/csamplawski/data/mmm/2022-09-01/node_4/zed.hdf5'
        ],
        'start_time':
        1662065994888,
        'end_time':
        1662066144888,
        'num_future_frames':
        5,
        'num_past_frames':
        5,
        'valid_keys': ['mocap', 'zed_camera_left'],
        'pipelines': {
            'zed_camera_left': [{
                'type': 'DecodeJPEG'
            }, {
                'type': 'LoadFromNumpyArray'
            }, {
                'type': 'Resize',
                'img_scale': (270, 480),
                'keep_ratio': True
            }, {
                'type': 'RandomFlip',
                'flip_ratio': 0.0
            }, {
                'type': 'Normalize',
                'mean': [123.675, 116.28, 103.53],
                'std': [58.395, 57.12, 57.375],
                'to_rgb': True
            }, {
                'type': 'DefaultFormatBundle'
            }, {
                'type': 'Collect',
                'keys': ['img']
            }],
            'zed_camera_depth': [{
                'type': 'LoadFromNumpyArray',
                'force_float32': True
            }, {
                'type': 'RandomFlip',
                'flip_ratio': 0.0
            }, {
                'type': 'Normalize',
                'mean': [0],
                'std': [20000],
                'to_rgb': False
            }, {
                'type': 'Normalize',
                'mean': [1],
                'std': [0.5],
                'to_rgb': False
            }, {
                'type': 'DefaultFormatBundle'
            }, {
                'type': 'Collect',
                'keys': ['img']
            }],
            'azimuth_static': [{
                'type': 'LoadFromNumpyArray',
                'force_float32': True
            }, {
                'type': 'RandomFlip',
                'flip_ratio': 0.0
            }, {
                'type': 'DefaultFormatBundle'
            }, {
                'type': 'Collect',
                'keys': ['img']
            }],
            'range_doppler': [{
                'type': 'LoadFromNumpyArray',
                'force_float32': True,
                'transpose': True
            }, {
                'type': 'Resize',
                'img_scale': (256, 16),
                'keep_ratio': True
            }, {
                'type': 'RandomFlip',
                'flip_ratio': 0.0
            }, {
                'type': 'Normalize',
                'mean': [4353],
                'std': [705],
                'to_rgb': False
            }, {
                'type': 'DefaultFormatBundle'
            }, {
                'type': 'Collect',
                'keys': ['img']
            }],
            'mic_waveform': [{
                'type': 'LoadFromNumpyArray',
                'force_float32': True,
                'transpose': True
            }, {
                'type': 'RandomFlip',
                'flip_ratio': 0.0
            }, {
                'type': 'DefaultFormatBundle'
            }, {
                'type': 'Collect',
                'keys': ['img']
            }],
            'realsense_camera_img': [{
                'type': 'DecodeJPEG'
            }, {
                'type': 'LoadFromNumpyArray'
            }, {
                'type': 'Resize',
                'img_scale': (270, 480),
                'keep_ratio': True
            }, {
                'type': 'RandomFlip',
                'flip_ratio': 0.0
            }, {
                'type': 'Normalize',
                'mean': [123.675, 116.28, 103.53],
                'std': [58.395, 57.12, 57.375],
                'to_rgb': True
            }, {
                'type': 'DefaultFormatBundle'
            }, {
                'type': 'Collect',
                'keys': ['img']
            }],
            'realsense_camera_depth': [{
                'type': 'DecodeJPEG'
            }, {
                'type': 'LoadFromNumpyArray'
            }, {
                'type': 'Resize',
                'img_scale': (270, 480),
                'keep_ratio': True
            }, {
                'type': 'RandomFlip',
                'flip_ratio': 0.0
            }, {
                'type': 'Normalize',
                'mean': [123.675, 116.28, 103.53],
                'std': [58.395, 57.12, 57.375],
                'to_rgb': True
            }, {
                'type': 'DefaultFormatBundle'
            }, {
                'type': 'Collect',
                'keys': ['img']
            }]
        },
        'max_len':
        None
    }, ),
    val=dict(
        type='HDF5Dataset',
        hdf5_fnames=[
            '/home/csamplawski/data/mmm/2022-09-01/mocap.hdf5',
            '/home/csamplawski/data/mmm/2022-09-01/node_1/zed.hdf5',
            '/home/csamplawski/data/mmm/2022-09-01/node_4/zed.hdf5'
        ],
        start_time=1662066144888,
        end_time=1662066294878,
        num_future_frames=0,
        num_past_frames=5,
        valid_keys=['mocap', 'zed_camera_left'],
        pipelines=dict(
            zed_camera_left=[
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
                    type='LoadFromNumpyArray',
                    force_float32=True,
                    transpose=True),
                dict(type='Resize', img_scale=(256, 16), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.0),
                dict(type='Normalize', mean=[4353], std=[705], to_rgb=False),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img'])
            ],
            mic_waveform=[
                dict(
                    type='LoadFromNumpyArray',
                    force_float32=True,
                    transpose=True),
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
            ]),
        vid_path='exps/single_truck_node_4_5p5f/',
        max_len=500,
        limit_axis=True,
        draw_cov=True),
    test=dict(
        type='HDF5Dataset',
        hdf5_fnames=[
            '/home/csamplawski/data/mmm/2022-09-01/mocap.hdf5',
            '/home/csamplawski/data/mmm/2022-09-01/node_1/zed.hdf5',
            '/home/csamplawski/data/mmm/2022-09-01/node_4/zed.hdf5'
        ],
        start_time=1662066144888,
        end_time=1662066294878,
        num_future_frames=0,
        num_past_frames=5,
        valid_keys=['mocap', 'zed_camera_left'],
        pipelines=dict(
            zed_camera_left=[
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
                    type='LoadFromNumpyArray',
                    force_float32=True,
                    transpose=True),
                dict(type='Resize', img_scale=(256, 16), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.0),
                dict(type='Normalize', mean=[4353], std=[705], to_rgb=False),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img'])
            ],
            mic_waveform=[
                dict(
                    type='LoadFromNumpyArray',
                    force_float32=True,
                    transpose=True),
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
            ]),
        vid_path='exps/single_truck_node_4_5p5f/',
        max_len=500,
        limit_axis=True,
        draw_cov=True))
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1))))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
total_epochs = 50
lr_config = dict(policy='step', step=[40])
evaluation = dict(metric=['bbox', 'track'], interval=100000000.0)
find_unused_parameters = True
checkpoint_config = dict(interval=50)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
work_dir = 'exps/single_truck_node_14_5p5f/log'
gpu_ids = [0]
