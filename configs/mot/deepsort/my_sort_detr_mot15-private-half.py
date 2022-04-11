_base_ = [
    '../../_base_/models/detr.py',
    '../../_base_/datasets/mot15.py', 
    '../../_base_/default_runtime.py'
    #'../../_base_/datasets/mot_challenge.py', 
]

custom_imports = dict(imports=['mmtrack.models.mot.kf'], allow_failed_imports=False)

link = 'https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'

model = dict(type='KFSORT',
    detector=dict(
        init_cfg=dict(type='Pretrained',checkpoint=link),
    ),
    score_thres=0.9,
    iou_thres=0.3,
)

dataset_type = 'MOTChallengeDataset'
data_root = 'data/MOT15/'
# data = dict(
    # samples_per_gpu=2,
    # workers_per_gpu=2,
    # train=dict(
        # type=dataset_type,
        # visibility_thr=-1,
        # ann_file=data_root + 'annotations/half-train_cocoformat.json',
        # img_prefix=data_root + 'train',
        # ref_img_sampler=dict(
            # num_ref_imgs=1,
            # frame_range=10,
            # filter_key_img=True,
            # method='uniform'),
        # pipeline=train_pipeline),
    # val=dict(
        # type=dataset_type,
        # ann_file=data_root + 'annotations/half-val_cocoformat.json',
        # img_prefix=data_root + 'train',
        # ref_img_sampler=None,
        # pipeline=test_pipeline),
    # test=dict(
        # type=dataset_type,
        # ann_file=data_root + 'annotations/half-val_cocoformat.json',
        # img_prefix=data_root + 'train',
        # ref_img_sampler=None,
        # pipeline=test_pipeline))


# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 100,
    step=[3])
# runtime settings
total_epochs = 4
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
