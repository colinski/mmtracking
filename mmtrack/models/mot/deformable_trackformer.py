# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models import build_detector, build_head

from mmtrack.core import outs2results, results2outs
from mmtrack.models.mot import BaseMultiObjectTracker
from ..builder import MODELS, build_tracker

from mmdet.core import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy, reduce_mean


@MODELS.register_module()
class Trackformer(BaseMultiObjectTracker):
    def __init__(self,
                 detector=None,
                 track_head=None,
                 tracker=None,
                 learn_track_pos=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if detector is not None:
            self.detector = build_detector(detector)

        if track_head is not None:
            self.track_head = build_head(track_head)

        if tracker is not None:
            self.tracker = build_tracker(tracker)

        
        self.track_pos = None
        if learn_track_pos:
            pos = self.detector.bbox_head.query_embedding.weight[:, 0:256]
            self.track_pos = pos.mean(dim=0, keepdims=True).detach()
            self.track_pos = torch.nn.Parameter(self.track_pos)

        
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices,
                      ref_img,
                      ref_img_metas,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      **kwargs):
        """Forward function during training.

         Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                each item has a shape (num_gts, 4).
            gt_labels (list[Tensor]): Ground truth labels of all images.
                each has a shape (num_gts,).
            gt_match_indices (list(Tensor)): Mapping from gt_instance_ids to
                ref_gt_instance_ids of the same tracklet in a pair of images.
            ref_img (Tensor): of shape (N, C, H, W) encoding input reference
                images. Typically these should be mean centered and std scaled.
            ref_img_metas (list[dict]): list of reference image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape',
                and 'img_norm_cfg'.
            ref_gt_bboxes (list[Tensor]): Ground truth bboxes of the
                reference image, each item has a shape (num_gts, 4).
            ref_gt_labels (list[Tensor]): Ground truth labels of all
                reference images, each has a shape (num_gts,).
            gt_masks (list[Tensor]) : Masks for each bbox, has a shape
                (num_gts, h , w).
            gt_bboxes_ignore (list[Tensor], None): Ground truth bboxes to be
                ignored, each item has a shape (num_ignored_gts, 4).
            ref_gt_bboxes_ignore (list[Tensor], None): Ground truth bboxes
                of reference images to be ignored,
                each item has a shape (num_ignored_gts, 4).
            ref_gt_masks (list[Tensor]) : Masks for each reference bbox,
                has a shape (num_gts, h , w).

        Returns:
            dict[str : Tensor]: All losses.
        """
        img_metas[0]['batch_input_shape'] = (img.shape[2], img.shape[3])
        bbox_head = self.detector.bbox_head
        gt_bboxes = gt_bboxes[0]
        gt_labels = gt_labels[0]
        ref_gt_bboxes = ref_gt_bboxes[0]
        ref_gt_labels = ref_gt_labels[0]
        gt_match_indices = gt_match_indices[0]

        #import ipdb; ipdb.set_trace() # noqa

        H, W , _ = img_metas[0]['img_shape']
        factor = gt_bboxes.new_tensor([W, H, W, H]).unsqueeze(0)

        ###################################################################
        #backbone forward pass with both imgs in a batch
        imgs = torch.cat([img, ref_img], dim=0)
        all_feats = self.detector.extract_feat(imgs)
        all_feats = [feat.split(len(imgs) // 2, dim=0) for feat in all_feats]
        feats = [feat[0] for feat in all_feats]
        ref_feats = [feat[1] for feat in all_feats]
        ###################################################################
        
       
        ###################################################################
        #transformer and output heads using first frame (img)
        query_embeds = bbox_head.query_embedding.weight 
        query_embeds, init_ref, inter_ref = bbox_head.forward_transformer(
            feats, query_embeds, img_metas
        )
        cls_scores, bbox_preds = bbox_head.forward_output_heads(
            query_embeds, init_ref, inter_ref
        )
        query_embeds = query_embeds[-1].squeeze()
        cls_score = cls_scores[-1].squeeze()
        bbox_pred = bbox_preds[-1].squeeze()
        ###################################################################

                
        ###################################################################
        #calc loss
        pred_idx, gt_idx = self.assign(
            bbox_pred, cls_score, 
            gt_bboxes, gt_labels, img_metas
        )
        
        labels = gt_bboxes.new_ones(len(bbox_pred)).long() * bbox_head.num_classes
        label_weights = gt_bboxes.new_ones(len(bbox_pred))
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        
        labels[pred_idx] = gt_labels[gt_idx]
        bbox_weights[pred_idx] = 1.0

                
        # pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        # pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        # bbox_targets[pos_inds] = pos_gt_bboxes_targets
        
        bbox_targets[pred_idx] = bbox_xyxy_to_cxcywh(gt_bboxes / factor)[gt_idx]

        num_pos = len(pred_idx)
        num_neg = query_embeds.shape[0] - num_pos
        loss_cls, loss_bbox, loss_iou = self.detr_loss(
            cls_score.unsqueeze(0), bbox_pred.unsqueeze(0),
            labels, label_weights,
            bbox_targets, bbox_weights,
            num_pos, num_neg, img_metas
        )
        losses = {
            'curr.loss_cls': loss_cls,
            'curr.loss_bbox': loss_bbox,
            'curr.loss_iou': loss_iou
        }
        ###################################################################


        ###################################################################
        #track embeds are copies of matched query embeds
        #append corresponding positional encodings 
        track_embeds = query_embeds[pred_idx]
        query_pos, _ = torch.split(bbox_head.query_embedding.weight, track_embeds.shape[-1], dim=1)
        if self.track_pos is not None:
            track_pos = query_pos[pred_idx]
            track_pos += self.track_pos.expand(len(track_embeds), -1)
        else:
            track_pos = query_pos[pred_idx]
        track_embeds  = torch.cat([track_pos, track_embeds], dim=-1)

        #concat all embeds and run through transformer and output heads
        all_embeds = torch.cat([bbox_head.query_embedding.weight, track_embeds], dim=0)
        all_embeds, init_ref, inter_ref = bbox_head.forward_transformer(
            ref_feats, all_embeds, img_metas
        )
        ref_cls_scores, ref_bbox_preds = bbox_head.forward_output_heads(
            all_embeds, init_ref, inter_ref
        )
        ref_cls_score = ref_cls_scores[-1].squeeze()
        ref_bbox_pred = ref_bbox_preds[-1].squeeze()
        ###################################################################

        ###################################################################
        #calc loss round 2
        num_bboxes = len(ref_bbox_pred)
        num_tracks = len(track_embeds)
        num_query = num_bboxes - len(track_embeds)
        labels = ref_gt_bboxes.new_ones(num_bboxes).long() * bbox_head.num_classes
        label_weights = ref_gt_bboxes.new_ones(num_bboxes)
        bbox_targets = torch.zeros_like(ref_bbox_pred)
        bbox_weights = torch.zeros_like(ref_bbox_pred)
        
        #index ref_bboxes so it lines up with gt_bboxes through time
        cont_bboxes = ref_gt_bboxes[gt_match_indices]
        cont_labels = ref_gt_labels[gt_match_indices]
        
        #idx[i] == -1 => object has left the scene
        #in this case match with zero box, which will be ignored in loss
        #assign background class for finished tracks (not ignored in loss)
        is_over = gt_match_indices == -1
        cont_bboxes[is_over] = 0
        cont_labels[is_over] = bbox_head.num_classes

        #set vals at end of targets/labels (where track embeds are)
        #we are able to skip matching algo because we know track id
        #this assumes track embeds are the last elements in array
        bbox_targets[num_query:] = bbox_xyxy_to_cxcywh(cont_bboxes / factor)[gt_idx]
        labels[num_query:] = cont_labels[gt_idx]
        
        
        #find all ref_bboxes not matched to track
        #these are objects that appeared this frame
        full_idx = range(len(ref_gt_bboxes))
        new_idx = [i for i in full_idx  if i not in gt_match_indices[~is_over]]
        new_idx = ref_gt_bboxes.new(new_idx).long()
        new_bboxes = ref_gt_bboxes[new_idx]
        new_labels = ref_gt_labels[new_idx]

        #select predictions from first 300 predictions
        #i.e. the output of normal detr
        query_bbox_pred = ref_bbox_pred[0:num_query]
        query_cls_score  = ref_cls_score[0:num_query]

        #match with new bboxes
        pred_idx, gt_idx = self.assign(
            query_bbox_pred, query_cls_score,
            new_bboxes, new_labels, img_metas
        )
        
        #write in assigned boxes to targets from locations in pos_ids
        #this assume track embeds are the last elements in array
        bbox_targets[pred_idx] = bbox_xyxy_to_cxcywh(new_bboxes / factor)[gt_idx]
        labels[pred_idx] = ref_gt_labels[gt_idx]
        
        #find all bbox targets that are [0,0,0,0]
        #these are ignored in the loss
        non_zero = bbox_targets.sum(dim=-1) != 0
        bbox_weights[non_zero] = 1.0
        num_pos = non_zero.sum().item()
        num_neg = num_bboxes - num_pos
        
        loss_cls, loss_bbox, loss_iou = self.detr_loss(
            ref_cls_score.unsqueeze(0), ref_bbox_pred.unsqueeze(0),
            labels, label_weights,
            bbox_targets, bbox_weights,
            num_pos, num_neg, img_metas
        )

        losses['ref.loss_cls'] = loss_cls
        losses['ref.loss_bbox'] = loss_bbox
        losses['ref.loss_iou'] = loss_iou
        return losses
    
    def detr_loss(self, cls_scores, bbox_preds,
                    labels, label_weights,
                    bbox_targets, bbox_weights,
                    num_total_pos, num_total_neg,
                    img_metas):
        bbox_head = self.detector.bbox_head

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_scores = cls_scores.reshape(-1, bbox_head.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0
        cls_avg_factor += num_total_neg * bbox_head.bg_cls_weight
        if bbox_head.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = bbox_head.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = bbox_head.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = bbox_head.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def assign(self, bbox_pred, cls_score, gt_bboxes, gt_labels, img_metas):
        assign_result = self.detector.bbox_head.assigner.assign(
            bbox_pred, cls_score, 
            gt_bboxes, gt_labels,
            img_metas[0],
        )
        sampling_result = self.detector.bbox_head.sampler.sample(
            assign_result, bbox_pred, gt_bboxes
        )
        return sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds


    def simple_test(self, img, img_metas, rescale=False):
        """Test forward.

         Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool): whether to rescale the bboxes.

        Returns:
            dict[str : Tensor]: Track results.
        """
        return dict()
