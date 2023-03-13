_base_ = [
    '../_base_/datasets/mmm/2022-09-01/trucks1_lightsF_obstaclesT.py'
]

trainset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir='/dev/shm/cache_train/',
        num_future_frames=0,
        num_past_frames=9,
        valid_nodes=[1,2,3,4],
        valid_mods=['mocap', 'zed_camera_left'],
        include_z=False,
    ),
    num_future_frames=0,
    num_past_frames=99,
)

valset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir='/dev/shm/cache_val/',
        valid_nodes=[1,2,3,4],
        valid_mods=['mocap', 'zed_camera_left'],
        include_z=False,
    ),
    num_future_frames=0,
    num_past_frames=0,
    limit_axis=True,
    draw_cov=True,
)

testset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir='/dev/shm/cache_test/',
        valid_nodes=[1,2,3,4],
        valid_mods=['mocap', 'zed_camera_left'],
        include_z=False,
    ),
    num_future_frames=0,
    num_past_frames=0,
    limit_axis=True,
    draw_cov=True,
)



model_cfg=dict(type='LinearEncoder', in_len=1, out_len=1, ffn_cfg=dict(type='SLP', in_channels=256))

model_cfgs = {('zed_camera_left', 'node_1'): model_cfg,
              ('zed_camera_left', 'node_2'): model_cfg,
              ('zed_camera_left', 'node_3'): model_cfg,
              ('zed_camera_left', 'node_4'): model_cfg}

backbone_cfgs = {'zed_camera_left': dict(type='TVResNet50CrossAttn')}

model = dict(type='KFDETR',
        output_head_cfg=dict(type='OutputHead',
         include_z=False,
         predict_full_cov=True,
         cov_add=1.0,
         input_dim=256,
         predict_rotation=True,
         predict_velocity=False,
         num_sa_layers=0,
         to_cm=True,
         mlp_dropout_rate=0.0
    ),
    model_cfgs=model_cfgs,
    backbone_cfgs=backbone_cfgs,
    track_eval=True,
    pos_loss_weight=1,
    num_queries=1,
    mod_dropout_rate=0.0,
    loss_type='nll',
    init_cfg=dict(type='Pretrained', checkpoint='logs/trucks1_lightsF_obstaclesT_zed_nodes1234_det/latest.pth'),
    freeze_backbone=True,
    kf_train=True
)


# orig_bs = 2
# orig_lr = 1e-4 
# factor = 4
data = dict(
    samples_per_gpu=1,
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
    # paramwise_cfg=dict(
        # custom_keys={
            # 'backbone': dict(lr_mult=0.1),
            # 'sampling_offsets': dict(lr_mult=0.1),
            # 'reference_points': dict(lr_mult=0.1)
        # }
    # )
)

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
total_epochs = 5
lr_config = dict(policy='step', step=[4])
evaluation = dict(metric=['track'], interval=1e8)

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
