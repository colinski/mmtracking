# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lap 
from mmdet.models import build_detector, build_head, build_backbone, build_neck
from collections import OrderedDict

import torch.distributed as dist
from mmtrack.core import outs2results, results2outs
# from mmtrack.models.mot import BaseMultiObjectTracker
from mmcv.runner import BaseModule, auto_fp16
from ..builder import MODELS, build_tracker

from mmdet.core import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy, reduce_mean
import copy

import torch.distributions as D
from .base import BaseMocapModel
from mmdet.models import build_loss
from cad.pos import AnchorEncoding
from cad.attn import ResCrossAttn
from collections import defaultdict

def linear_assignment(cost_matrix):
    cost_matrix = cost_matrix.cpu().detach().numpy()
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    assign_idx = np.array([[y[i], i] for i in x if i >= 0])
    sort_idx = np.argsort(assign_idx[:, 1])
    assign_idx = assign_idx[sort_idx]
    assign_idx = torch.from_numpy(assign_idx)
    return assign_idx.long()

@MODELS.register_module()
class DecoderMocapModel(BaseMocapModel):
    def __init__(self,
                 img_backbone_cfg=None,
                 img_neck_cfg=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        cross_attn_cfg = dict(type='QKVAttention',
                 qk_dim=256,
                 num_heads=8, 
                 in_proj=True, 
                 out_proj=True,
                 attn_drop=0.1, 
                 seq_drop=0.0,
                 return_weights=False,
                 v_dim=None
        )
        
        self.img_neck = img_neck_cfg
        if self.img_neck is not None:
            self.img_neck = build_neck(img_neck_cfg)

        self.img_backbone = build_backbone(img_backbone_cfg)

        self.img_pos_encoding = AnchorEncoding(dim=256, learned=False, out_proj=False)
        self.global_pos_encoding = AnchorEncoding(dim=256, learned=False, out_proj=False)

        self.img_cross_attn = ResCrossAttn(cross_attn_cfg)
        self.global_cross_attn = ResCrossAttn(cross_attn_cfg)

        self.pool = nn.AvgPool2d((20, 1))
        self.ctn = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )
        
        # self.mean_head = nn.Linear(256, 3)
        self.mean_head = nn.Sequential(
            nn.Linear(256, 3),
            #nn.Sigmoid()
        )

        self.cov_head = nn.Sequential(
            nn.Linear(256, 3),
            nn.Softplus()
        )
        
        #truck, node, drone, no obj
        self.num_classes = 3
        self.cls_head = nn.Linear(256, self.num_classes)
        
        self.obj_head = nn.Linear(256, 1)

        self.dist = D.Normal

        focal_loss_cfg = dict(type='FocalLoss',
            use_sigmoid=True, 
            gamma=2.0, alpha=0.25, reduction='none',
            loss_weight=1.0, activated=True)

        self.focal_loss = build_loss(focal_loss_cfg)
        self.bce_loss = nn.BCELoss(reduction='none')

    
    #def forward(self, data, return_loss=True, **kwargs):
    def forward(self, data, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(data, **kwargs)
        else:
            return self.forward_test(data, **kwargs)


    def _forward(self, data, **kwargs):
        if 'zed_camera_left' in data.keys():
            img = data['zed_camera_left']['img']
            img_metas = data['zed_camera_left']['img_metas']
            img_metas[0]['batch_input_shape'] = (img.shape[2], img.shape[3])
            bs = img.shape[0]
            
            # bbox_head = self.img_detector.bbox_head
            feats = self.img_backbone(img)
            if self.img_neck:
                feats = self.img_neck(feats)
            if len(feats) > 1:
                target_shape = (feats[2].shape[2], feats[2].shape[3])
                feats = [F.interpolate(f, target_shape) for f in feats] 
                feats = torch.cat(feats, dim=1)
            else:
                feats = feats[0]
            feats = feats.permute(0, 2, 3, 1)


            img_pos_embeds = self.img_pos_encoding(None).unsqueeze(0)
            img_pos_embeds = img_pos_embeds.expand(bs, -1, -1, -1)
            query_embeds_img = self.img_cross_attn(img_pos_embeds, feats)

            global_pos_embeds = self.global_pos_encoding(None).unsqueeze(0)
            global_pos_embeds = global_pos_embeds.expand(bs, -1, -1, -1)

            final_embeds = self.global_cross_attn(global_pos_embeds, query_embeds_img)
            # final_embeds = self.global_cross_attn(global_pos_embeds, feats)
            final_embeds = final_embeds.reshape(bs, -1, 256)


        else:
            global_pos_embeds = self.global_pos_encoding(None).unsqueeze(0)
            final_embeds = global_pos_embeds.reshape(1, -1, 256)
                   
        final_embeds = self.ctn(final_embeds)#.mean(dim=0)

        mean = self.mean_head(final_embeds)
        cov = self.cov_head(final_embeds)
        cls_logits = self.cls_head(final_embeds)
        obj_logits = self.obj_head(final_embeds)
        return mean, cov, cls_logits, obj_logits


    def forward_test(self, data, **kwargs):
        mean, cov, cls_logits, obj_logits = self._forward(data, **kwargs)
        obj_probs = F.sigmoid(obj_logits[0]).squeeze()
        is_obj = obj_probs >= 0.5
        mean = mean[:, is_obj]
        cov = cov[:, is_obj]
        
        result = {
            'pred_position': mean[0].cpu().detach().unsqueeze(0).numpy()
        }
        return result

    def forward_train(self, data, **kwargs):
        losses = defaultdict(list)
        
        mean, cov, cls_logits, obj_logits = self._forward(data, **kwargs)
        bs = len(mean)
        for i in range(bs):
            is_missing = data['missing']['zed_camera_left'][i]
            if is_missing:
                continue

            dist = self.dist(mean[i], cov[i])
            dist = D.Independent(dist, 1) #Nq independent Gaussians

            gt_pos = data['mocap']['gt_positions'][i]#[-2].unsqueeze(0)
            gt_labels = data['mocap']['gt_labels'][i]#[-2].unsqueeze(0)

            z_is_zero = gt_pos[:, -1] == 0.0

            is_node = gt_labels == 0
            final_mask = ~z_is_zero & ~is_node
            gt_pos = gt_pos[final_mask]
            gt_labels = gt_labels[final_mask]

            

            pos_log_probs = [dist.log_prob(pos) for pos in gt_pos]
            pos_neg_log_probs = -torch.stack(pos_log_probs, dim=-1) #Nq x num_objs
            assign_idx = linear_assignment(pos_neg_log_probs)

            pos_loss = 0
            for pred_idx, gt_idx in assign_idx:
                pos_loss += pos_neg_log_probs[pred_idx, gt_idx]
            pos_loss /= len(assign_idx)
            pos_loss /= 10
            losses['pos_loss'].append(pos_loss)

            low_end = 0.2
            obj_targets = pos_neg_log_probs.new_zeros(len(pos_neg_log_probs)) + low_end
            obj_targets = obj_targets.float()
            obj_targets[assign_idx[:, 0]] = 1.0 - low_end
            
            obj_probs = F.sigmoid(obj_logits[i])
            
            # obj_loss_vals = self.focal_loss(obj_probs, obj_targets.long())
            # losses['pos_obj_loss'] = obj_loss_vals[obj_targets == 1].mean()
            # losses['neg_obj_loss'] = obj_loss_vals[obj_targets == 0].mean()

            obj_loss_vals = self.bce_loss(obj_probs.squeeze(), obj_targets)
            losses['pos_obj_loss'].append(obj_loss_vals[obj_targets == 1.0 - low_end].mean())
            losses['neg_obj_loss'].append(obj_loss_vals[obj_targets == low_end].mean())

        losses = {k: torch.stack(v).mean() for k, v in losses.items()}
        return losses
        

    def simple_test(self, img, img_metas, rescale=False):
        pass

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
            ``num_samples``.

            - ``loss`` is a tensor for back propagation, which can be a
            weighted sum of multiple losses.
            - ``log_vars`` contains all the variables to be sent to the
            logger.
            - ``num_samples`` indicates the batch size (when the model is
            DDP, it means the batch size on each GPU), which is used for
            averaging the logs.
        """
        losses = self(data)
        loss, log_vars = self._parse_losses(losses)
        
        num_samples = len(data['mocap']['gt_positions'])

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
            which may be a weighted sum of all losses, log_vars contains
            all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars



    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

