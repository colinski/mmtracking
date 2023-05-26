# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models.backbones.resnet import Bottleneck, ResNet
from mmdet.models.builder import BACKBONES
from mmcls.models.backbones.convnext import ConvNeXtBlock
from mmcv.runner import BaseModule, auto_fp16
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
import torch
import torch.nn.functional as F
from cad.attn import ResCrossAttn, ResSelfAttn
from models.common import Conv, DWConv
from mmdet.apis import init_detector, inference_detector

@BACKBONES.register_module()
class YOLOv7(BaseModule):
    def __init__(self, 
            out_channels=256,
            weights='src/mmtracking/yolov7-tiny.pt'
        ):
        super().__init__()
        # for w in weights if isinstance(weights, list) else [weights]:
            # attempt_download(w)
        ckpt = torch.load(weights)
        self.yolo = ckpt['ema' if ckpt.get('ema') else 'model'].float()
        
        # Compatibility updates
        for m in self.yolo.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

        #config_file = '/home/csamplawski/src/mmdetection/configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py'
        #checkpoint_file = '/home/csamplawski/src/mmtracking/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'
        # config_file = 'src/mmdetection/configs/detr/detr_r50_8x2_150e_coco.py'
        # checkpoint_file = 'src/mmtracking/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'
        # self.detr = init_detector(config_file, checkpoint_file, device='cuda')  # or device='cuda:0'
        # self.detr = self.detr.eval()
    
    #@torch.no_grad()
    def forward(self, x):
        output = self.yolo.forward_once(x, apply_detector=False)
        return (output[-1], )


@BACKBONES.register_module()
class PretrainedDETR(BaseModule):
    def __init__(self, 
            out_channels=256,
        ):
        super().__init__()
        #config_file = '/home/csamplawski/src/mmdetection/configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py'
        #checkpoint_file = '/home/csamplawski/src/mmtracking/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'
        config_file = 'src/mmdetection/configs/detr/detr_r50_8x2_150e_coco.py'
        checkpoint_file = 'src/mmtracking/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'
        self.detr = init_detector(config_file, checkpoint_file, device='cuda')  # or device='cuda:0'
        self.detr = self.detr.eval()
    
    #@torch.no_grad()
    def forward(self, x):
        x = self.detr.backbone(x)[0]
        # x = self.detr.neck(x)
        B, D, H, W = x.shape
        masks = x.new_zeros((B, H, W)).to(torch.bool)
        bbox_head = self.detr.bbox_head
        x = bbox_head.input_proj(x)
        #masks = F.interpolate(masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        pos_embed = bbox_head.positional_encoding(masks)
        outs_dec, _ = bbox_head.transformer(x, masks, bbox_head.query_embedding.weight, pos_embed)
        # cls_scores = bbox_head.fc_cls(outs_dec)
        # bbox_preds = bbox_head.fc_reg(bbox_head.activate(bbox_head.reg_ffn(outs_dec))).sigmoid()
        return (outs_dec[-1], )

#@BACKBONES.register_module()
class ResNet50Stem(nn.Sequential):
    def __init__(self, frozen=True):
        r50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        super().__init__(
            r50.conv1,
            r50.bn1,
            r50.relu,
            r50.maxpool
        )
        self.frozen = frozen
        if self.frozen:
            self.forward = self._forward_frozen
    
    @torch.no_grad()
    def _forward_frozen(self, x):
        for layer in self:
            x = layer.eval()(x)
        return x

@BACKBONES.register_module()
class TVResNet50CrossAttn(BaseModule):
    def __init__(self, 
                 cross_attn_cfg=dict(type='QKVAttention',
                     qk_dim=256,
                     num_heads=8, 
                     in_proj=True, 
                     out_proj=True,
                     attn_drop=0.0, 
                     seq_drop=0.0,
                     v_dim=None
                 ),
                 out_channels=256,
                 norm_cfg=dict(type='BN')
        ):
        super().__init__()
        self.global_cross_attn = ResCrossAttn(cross_attn_cfg)
        self.global_pos_encoding = nn.Embedding(1, out_channels)
        self.stem = ResNet50Stem(frozen=True)
        self.layers = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=7, stride=2, padding=3),
            build_norm_layer(norm_cfg, out_channels)[1],

            ConvNeXtBlock(out_channels, layer_scale_init_value=0.0),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=2, padding=3),
            build_norm_layer(norm_cfg, out_channels)[1],
            
            ConvNeXtBlock(out_channels, layer_scale_init_value=0.0),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=2, padding=3),
            build_norm_layer(norm_cfg, out_channels)[1]
        )
               
    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = x.flatten(2).permute(0,2,1)
        B, L, D = x.shape
        global_pos_embeds = self.global_pos_encoding.weight.unsqueeze(0) # 1 x 2 x 256
        global_pos_embeds = global_pos_embeds.expand(B, -1, -1)
        global_pos_embeds = self.global_cross_attn(global_pos_embeds, x)
        return (global_pos_embeds, )

@BACKBONES.register_module()
class AudioBackbone(BaseModule):
    def __init__(self, 
                 input_dim=4,
                 out_channels=256,
                 norm_cfg=dict(type='BN')
        ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, out_channels, kernel_size=7, stride=2, padding=3),
            build_norm_layer(norm_cfg, out_channels)[1],

            ConvNeXtBlock(out_channels, layer_scale_init_value=0.0),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=2, padding=3),
            build_norm_layer(norm_cfg, out_channels)[1],
            
            ConvNeXtBlock(out_channels, layer_scale_init_value=0.0),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=2, padding=3),
            build_norm_layer(norm_cfg, out_channels)[1]
        )
               
    def forward(self, x):
        x = self.layers(x)
        return (x, )

@BACKBONES.register_module()
class mmWaveBackbone(BaseModule):
    def __init__(self, 
                 input_dim=256,
                 out_channels=256,
                 norm_cfg=dict(type='BN')
        ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            #nn.Conv2d(input_dim, input_dim, kernel_size=7, padding=3, stride=2),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
            #nn.Conv2d(input_dim, input_dim, kernel_size=7, padding=3, stride=2),
            nn.GELU(),
        )

        # self.stem = ResNet50Stem(frozen=True)
        # self.layers = nn.Sequential(
            # nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=3),
            # build_norm_layer(norm_cfg, out_channels)[1],

            # ConvNeXtBlock(out_channels, layer_scale_init_value=0.0),
            
            # nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=1, padding=3),
            # build_norm_layer(norm_cfg, out_channels)[1],
            
            # ConvNeXtBlock(out_channels, layer_scale_init_value=0.0),
            
            # nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=1, padding=3),
            # build_norm_layer(norm_cfg, out_channels)[1]
        # )
               
    def forward(self, x):
        x = self.mlp(x)
        x = x.permute(0,3, 1,2)
        #x = self.stem(x)
        #x = self.layers(x)
        return (x, )
