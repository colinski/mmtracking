from mmdet.models.detectors import DeformableDETR
from mmdet.apis import init_detector, inference_detector

config_file = '/home/csamplawski/src/mmdetection/configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py'
checkpoint_file = 'deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'
model = init_detector(config_file, checkpoint_file, device='cuda')  # or device='cuda:0'
model = model.eval()

import ipdb; ipdb.set_trace() # noqa
# inference_detector(model, 'demo/demo.jpg')
