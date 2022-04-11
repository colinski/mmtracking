_base_ = [
    '../../_base_/models/yolov3.py',
    '../../_base_/datasets/mot20-full.py', 
    '../../_base_/default_runtime.py'
]

#link = 'https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'

model = dict(
    type='DeepSORT',
    motion=dict(type='KalmanFilter', center_only=False),
    tracker=dict(
        type='SortTracker', obj_score_thr=0.3, match_iou_thr=0.3, reid=None))
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
