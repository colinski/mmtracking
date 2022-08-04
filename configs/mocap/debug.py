_base_ = [
    '../_base_/models/detr.py',
    # '../../_base_/datasets/mot_challenge.py', 
    # '../../_base_/default_runtime.py',
     #'../../_base_/datasets/mot15-half.py', 
]
custom_imports = dict(
        imports=[
            'mmtrack.models.mocap.base',
            # 'mmtrack.models.trackers.trackformer_tracker'
        ],
        allow_failed_imports=False)

model = dict(type='BaseMocapModel')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    # dict(type='SeqLoadAnnotations', with_bbox=True),
    # dict(type='SeqResize',
        # img_scale=(800, 1333),
        # share_params=True,
        # keep_ratio=True,
    # ),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(
        type='VideoCollect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
            'gt_instance_ids'
        ]),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]

depth_pipeline = [
    dict(type='LoadFromNumpyArray', force_float32=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', mean=[0], std=[20000], to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
    # dict(type='ImageToTensor', keys=['img']),
]

azimuth_pipeline = [
    dict(type='LoadFromNumpyArray', force_float32=True),
    dict(type='Resize', img_scale=(384, 384), keep_ratio=True),
    dict(type='ImageToTensor', keys=['img']),
]

range_pipeline = [
    dict(type='LoadFromNumpyArray', force_float32=True),
    dict(type='ImageToTensor', keys=['img']),
]


img_pipeline = [
    dict(type='LoadFromNumpyArray'),
    # dict(type='Resize', img_scale=(720, 1280), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # dict(type='ImageToTensor', keys=['img']),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),

]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 720),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]


classes = ('car', )
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    shuffle=False,
    train=dict(type='HDF5Dataset',
        hdf5_fname='/home/csamplawski/data/1656096647489_1656096707489.hdf5',
        img_pipeline=img_pipeline,
        depth_pipeline=depth_pipeline,
        azimuth_pipeline=azimuth_pipeline,
        range_pipeline=range_pipeline
    ),
    val=dict(type='HDF5Dataset',
        hdf5_fname='/home/csamplawski/data/1656096647489_1656096707489.hdf5',
        img_pipeline=img_pipeline,
        depth_pipeline=depth_pipeline,
        azimuth_pipeline=azimuth_pipeline,
        range_pipeline=range_pipeline
    ),
    test=dict(type='HDF5Dataset',
        hdf5_fname='/home/csamplawski/data/1656096647489_1656096707489.hdf5',
        img_pipeline=img_pipeline,
        depth_pipeline=depth_pipeline,
        azimuth_pipeline=azimuth_pipeline,
        range_pipeline=range_pipeline
    ),
)


optimizer = dict(
    type='AdamW',
    #lr=2e-4,
    lr=1e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[6])
total_epochs = 10
evaluation = dict(metric=['bbox', 'track'], interval=1e8)

find_unused_parameters = True

checkpoint_config = dict(interval=10)
log_config = dict(
    interval=50,
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
