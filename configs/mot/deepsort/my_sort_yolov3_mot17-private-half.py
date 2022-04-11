_base_ = [
    '../../_base_/models/yolov3.py',
    '../../_base_/datasets/mot_challenge.py', '../../_base_/default_runtime.py'
]

custom_imports = dict(imports=['mmtrack.models.mot.kf'], allow_failed_imports=False)

link = 'https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_mobilenetv2_mstrain-416_300e_coco/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth'

model = dict(
    type='KFSORT',
    detector=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=link,
    ),
    ),
    score_thres=0.3,
    iou_thres=0.3,
)

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
