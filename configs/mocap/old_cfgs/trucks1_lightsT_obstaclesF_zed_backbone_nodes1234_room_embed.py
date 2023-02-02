_base_ = [
    '../_base_/datasets/mmm/2022-09-01/trucks1_lightsT_obstaclesF.py'
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
    num_past_frames=9,
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

backbone_cfg=dict(type='TVResNet50')

model_cfg=dict(type='ModalityEncoder', ffn_cfg=dict(type='SLP'))

model_cfgs = {('zed_camera_left', 'node_1'): model_cfg,
              ('zed_camera_left', 'node_2'): model_cfg,
              ('zed_camera_left', 'node_3'): model_cfg,
              ('zed_camera_left', 'node_4'): model_cfg}
backbone_cfgs = {'zed_camera_left': backbone_cfg}

model = dict(type='DecoderMocapModel',
        output_head_cfg=dict(type='OutputHead',
         include_z=False,
         predict_full_cov=True,
         cov_add=30.0,
         predict_rotation=True,
         predict_velocity=False,
         num_sa_layers=0,
         to_cm=True,
         mlp_dropout_rate=0.0
    ),
    model_cfgs=model_cfgs,
    backbone_cfgs=backbone_cfgs,
    track_eval=True,
    mse_loss_weight=0.0,
    max_age=5,
    grid_loss=True,
    pos_loss_weight=1,
    num_queries=1,
    autoregressive=True,
    global_ca_layers=0,
    mod_dropout_rate=0.0,
)


orig_bs = 2
orig_lr = 1e-4 
factor = 4
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    shuffle=True, #trainset shuffle only
    train=trainset,
    val=valset,
    test=valset
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
