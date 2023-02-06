_base_ = [
    'trucks1_lightsT_obstaclesF.py'
]

data_root = 'data/mmm/2022-09-01/trucks2_lightsF_obstaclesF/train'
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
        ]
    )
)

data_root = 'data/mmm/2022-09-01/trucks2_lightsF_obstaclesF/test'
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
        ]
    )
)

data = dict(
    train=trainset,
    val=valset,
    test=valset
)
