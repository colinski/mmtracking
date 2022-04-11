_base_ = [
    '../../_base_/models/yolov3.py',
    '../../_base_/datasets/mot15-full.py', 
    '../../_base_/default_runtime.py'
    #'../../_base_/datasets/mot_challenge.py', 
]

custom_imports = dict(imports=['mmtrack.models.mot.kf'], allow_failed_imports=False)


model = dict(type='KFSORT',
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
