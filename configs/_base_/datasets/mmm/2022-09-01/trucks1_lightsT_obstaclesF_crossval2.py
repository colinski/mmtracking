_base_ = []

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
    'realsense_camera_r50': r50_pipeline,
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

valid_nodes=[1,2,3,4]

data_root = 'data/mmm/2022-09-01/trucks1_lightsT_obstaclesF/train'
trainset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
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
            # f'{data_root}/node_1/zed_r50.hdf5',
            # f'{data_root}/node_2/zed_r50.hdf5',
            # f'{data_root}/node_3/zed_r50.hdf5',
            # f'{data_root}/node_4/zed_r50.hdf5',
            # f'{data_root}/node_1/realsense_r50.hdf5',
            # f'{data_root}/node_2/realsense_r50.hdf5',
            # f'{data_root}/node_3/realsense_r50.hdf5',
            # f'{data_root}/node_4/realsense_r50.hdf5',

        ],
        valid_mods=valid_mods,
        valid_nodes=valid_nodes,
        fifths=[0,1,3,4]
    ),
    num_future_frames=0,
    num_past_frames=0,
    pipelines=pipelines,
)

data_root = 'data/mmm/2022-09-01/trucks1_lightsT_obstaclesF/val'
valset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
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
            # f'{data_root}/node_1/zed_r50.hdf5',
            # f'{data_root}/node_2/zed_r50.hdf5',
            # f'{data_root}/node_3/zed_r50.hdf5',
            # f'{data_root}/node_4/zed_r50.hdf5',
            # f'{data_root}/node_1/realsense_r50.hdf5',
            # f'{data_root}/node_2/realsense_r50.hdf5',
            # f'{data_root}/node_3/realsense_r50.hdf5',
            # f'{data_root}/node_4/realsense_r50.hdf5',

        ],
        valid_mods=valid_mods,
        valid_nodes=valid_nodes,
    ),
    num_future_frames=0,
    num_past_frames=0,
    pipelines=pipelines,
)

data_root = 'data/mmm/2022-09-01/trucks1_lightsT_obstaclesF/train'
testset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
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
            # f'{data_root}/node_1/zed_r50.hdf5',
            # f'{data_root}/node_2/zed_r50.hdf5',
            # f'{data_root}/node_3/zed_r50.hdf5',
            # f'{data_root}/node_4/zed_r50.hdf5',
            # f'{data_root}/node_1/realsense_r50.hdf5',
            # f'{data_root}/node_2/realsense_r50.hdf5',
            # f'{data_root}/node_3/realsense_r50.hdf5',
            # f'{data_root}/node_4/realsense_r50.hdf5',

        ],
        valid_mods=valid_mods,
        valid_nodes=valid_nodes,
        fifths=[2]
    ),
    num_future_frames=0,
    num_past_frames=0,
    pipelines=pipelines,
)


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    shuffle=True, #trainset shuffle only
    train=trainset,
    val=valset,
    test=testset
)
