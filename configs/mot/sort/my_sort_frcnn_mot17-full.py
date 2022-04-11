_base_ = [
    #'../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/models/frcnn.py',
    '../../_base_/datasets/mot17-full.py', 
    '../../_base_/default_runtime.py'
]

custom_imports = dict(imports=['mmtrack.models.mot.kf'], allow_failed_imports=False)

#link = 'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth'  # noqa: E501
model = dict(
    type='KFSORT',
    score_thres=0.5,
    iou_thres=0.5,
    # detector=dict(
        # rpn_head=dict(bbox_coder=dict(clip_border=False)),
        # roi_head=dict(
            # bbox_head=dict(bbox_coder=dict(clip_border=False), num_classes=1)),
        # init_cfg=dict(
            # type='Pretrained',
            # checkpoint=link,
        # )
    # ),
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
