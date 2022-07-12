# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lap 
from mmdet.models import build_detector, build_head

from mmtrack.core import outs2results, results2outs
from mmtrack.models.mot import BaseMultiObjectTracker
from ..builder import MODELS, build_tracker

from mmdet.core import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy, reduce_mean


def linear_assignment(cost_matrix):
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i], i] for i in x if i >= 0])

@MODELS.register_module()
class Trackformer(BaseMultiObjectTracker):
    def __init__(self,
                 detector=None,
                 mode='loc',
                 thres_new=0.5,
                 thres_cont=0.5,
                 thres_reid=5,
                 max_age=1,
                 min_hits=0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if detector is not None:
            self.detector = build_detector(detector)
        self.tracks = []
        self.sleeping_tracks = []
        self.num_queries = self.detector.bbox_head.query_embedding.weight.shape[0]
        self.thres_new = thres_new
        self.thres_cont = thres_cont
        self.thres_reid = thres_reid
        self.max_age = max_age
        self.min_hits = min_hits
        self.frame_count = 0
        self.dist_fn = torch.nn.PairwiseDistance(p=2, keepdim=True)

        self.pool = torch.nn.AvgPool2d((50, 1))
        
        # self.fc_cls = torch.nn.Linear(256, 81)
        # self.fc_coord = torch.nn.Linear(256, 2)
        self.ctn = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )
        self.coord_pred_head = nn.Linear(256, 2)

        self.ctns, self.pred_heads = {}, {}
        for idx in range(1, 5):
            node_name = 'node_%d' % idx
            # self.ctns[node_name] = nn.Sequential(
                # nn.Linear(256, 256),
                # nn.ReLU(),
                # nn.Linear(256, 256),
                # nn.ReLU()
            # )
            self.pred_heads[node_name] = nn.Linear(256, 2)
        
        self.pred_heads = nn.ModuleDict(self.pred_heads)
        # self.ctns = nn.ModuleDict(self.ctns)

        # self.detector.bbox_head.fc_cls = torch.nn.Linear(256, 81)
        # self.detector.bbox_head.fc_reg = torch.nn.Linear(256, 4)
        # cls_weight = torch.tensor([1.0, 0.1])
        # self.nll_loss = torch.nn.NLLLoss(weight=cls_weight, reduction='none')

    def reset(self):
        self.tracks = []
        self.sleeping_tracks = []
        self.frame_count = 0
     
    def forward_train_(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices=None,
                      ref_img=None,
                      ref_img_metas=None,
                      ref_gt_bboxes=None,
                      ref_gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      **kwargs):
        img_metas[0]['batch_input_shape'] = (img.shape[2], img.shape[3])
        bbox_head = self.detector.bbox_head
        gt_coords = [box[:, 0:2] for box in gt_bboxes]
        ref_gt_coords = [box[:, 0:2] for box in ref_gt_bboxes]
        img = torch.cat([img, ref_img], dim=0) 
        gt_coords = gt_coords + ref_gt_coords
        # gt_coords = [gc.unsqueeze(0) for gc in gt_coords]
        gt_coords = torch.cat(gt_coords, dim=0)
        gt_coords = gt_coords.squeeze()
        img_metas = img_metas + ref_img_metas
       
        query_embeds = bbox_head.query_embedding.weight 
        with torch.no_grad():
            feats = self.detector.extract_feat(img)[0]
            query_embeds = bbox_head.forward_transformer(
                feats, query_embeds, img_metas
            )

        # node_name = 'node_%s' % '1'
        # ctn = self.ctns[node_name]
        # pred_head = self.pred_heads[node_name]
        # coord_preds = ctn(query_embeds)
        # coord_preds = coord_preds.mean(dim=0)
        # coord_preds = coord_preds.mean(dim=1)
        # final_preds = pred_head(coord_preds).sigmoid()
        # ctn = self.ctns[node_name]

        coord_preds = self.ctn(query_embeds)
        coord_preds = coord_preds.mean(dim=0)
        coord_preds = coord_preds.mean(dim=1)

        
        final_preds = []
        for idx, meta in enumerate(img_metas):
            fname = meta['filename']
            base = fname.split('/')[-1].split('.')[0]
            node_id = base.split('_')[-1]
            node_name = 'node_%s' % node_id
            pred_head = self.pred_heads[node_name]
            # ctn = self.ctns[node_name]

            # coord_preds = ctn(query_embeds)
            # coord_preds = coord_preds.mean(dim=0)
            # coord_preds = coord_preds.mean(dim=1)
            preds = pred_head(coord_preds[idx]).sigmoid()
            final_preds.append(preds)
        final_preds = torch.stack(final_preds)
        
        mse_loss = self.dist_fn(final_preds, gt_coords).mean()
        losses = {'loss_mse': mse_loss}
        return losses
   
    def forward_train__(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices=None,
                      ref_img=None,
                      ref_img_metas=None,
                      ref_gt_bboxes=None,
                      ref_gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      **kwargs):
        import ipdb; ipdb.set_trace() # noqa
        img_metas[0]['batch_input_shape'] = (img.shape[2], img.shape[3])
        bbox_head = self.detector.bbox_head
        gt_coords = [box[:, 0:2] for box in gt_bboxes]
        ref_gt_coords = [box[:, 0:2] for box in ref_gt_bboxes]
        img = torch.cat([img, ref_img], dim=0) 
        gt_coords = gt_coords + ref_gt_coords
        gt_coords = [gc.unsqueeze(0) for gc in gt_coords]
        gt_coords = torch.cat(gt_coords, dim=0)
        img_metas = img_metas + ref_img_metas
       
        with torch.no_grad():
            feats = self.detector.extract_feat(img)[0]
            query_embeds = bbox_head.query_embedding.weight 
            query_embeds = bbox_head.forward_transformer(
                feats, query_embeds, img_metas
            )

        coord_preds = self.mlp_coord(query_embeds)
        
        coord_preds = self.pool(coord_preds)
        coord_preds = coord_preds.mean(dim=0).sigmoid()

        mse_loss = self.dist_fn(coord_preds, gt_coords).mean()
        losses = {'loss_mse': mse_loss}
        return losses
    
    #two trucks!
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices=None,
                      ref_img=None,
                      ref_img_metas=None,
                      ref_gt_bboxes=None,
                      ref_gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      **kwargs):
        img_metas[0]['batch_input_shape'] = (img.shape[2], img.shape[3])
        bbox_head = self.detector.bbox_head
        gt_coords = [box[:, 0:2] for box in gt_bboxes][0]
        ref_gt_coords = [box[:, 0:2] for box in ref_gt_bboxes][0]
        gt_match_indices = gt_match_indices[0]
        assert gt_match_indices[0] == 0 and gt_match_indices[1] == 1

        #transformer and output heads using first frame (img)
        with torch.no_grad():
            feats = self.detector.extract_feat(img)[0]
        query_embeds = bbox_head.query_embedding.weight 
        query_embeds = bbox_head.forward_transformer(
            feats, query_embeds, img_metas
        )

        query_embeds = query_embeds.mean(dim=0).squeeze()
        coord_embeds = self.ctn(query_embeds)
        coord_preds = self.coord_pred_head(coord_embeds).sigmoid()
        dists = self.dist_fn(coord_preds.unsqueeze(0), gt_coords.unsqueeze(1))
        dists = dists.squeeze().t()
        matches = linear_assignment(dists.detach().cpu().numpy()) 
        
        mse_loss_val = 0
        for (pred_idx, gt_idx) in matches:
            mse_loss_val += dists[pred_idx, gt_idx]
        mse_loss_val = mse_loss_val / len(matches)

        track_embeds = query_embeds[matches[:, 0]]
        losses = {'loss_mse_frame1': mse_loss_val}
        
        with torch.no_grad():
            feats = self.detector.extract_feat(ref_img)[0]
        track_embeds = bbox_head.forward_transformer(
            feats, track_embeds, img_metas
        )
        track_embeds = track_embeds.mean(dim=0).squeeze()
        coord_embeds = self.ctn(track_embeds)
        coord_preds = self.coord_pred_head(coord_embeds).sigmoid()

        loss_val = 0
        for idx, cp in enumerate(coord_preds):
            dist = self.dist_fn(cp, ref_gt_coords[idx])
            loss_val += dist
        loss_val /= len(coord_preds)
        # import ipdb; ipdb.set_trace() # noqa
        # dists = self.dist_fn(coord_preds.unsqueeze(0), ref_gt_coords.unsqueeze(1))
        # dists = torch.diag(dists.squeeze())

        losses['loss_mse_frame2'] = loss_val
        return losses

    def forward_train_(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices=None,
                      ref_img=None,
                      ref_img_metas=None,
                      ref_gt_bboxes=None,
                      ref_gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      **kwargs):
        img_metas[0]['batch_input_shape'] = (img.shape[2], img.shape[3])
        bbox_head = self.detector.bbox_head
        gt_coords = [box[:, 0:2] for box in gt_bboxes][0]
        ref_gt_coords = [box[:, 0:2] for box in ref_gt_bboxes][0]
        gt_match_indices = gt_match_indices[0]
        assert gt_match_indices[0] == 0 and gt_match_indices[1] == 1

        # gt_labels = gt_labels[0]
        # ref_gt_coords = ref_gt_bboxes[0][:, 0:2]
        # ref_gt_labels = ref_gt_labels[0]
        # gt_match_indices = gt_match_indices[0]


        # H, W , _ = img_metas[0]['img_shape']
        # factor = gt_bboxes.new_tensor([W, H, W, H]).unsqueeze(0)

        ###################################################################
        #backbone forward pass with both imgs in a batch
        #imgs = torch.cat([img, ref_img], dim=0)
        #all_feats = self.detector.extract_feat(imgs)[0]
        #all_feats = [feat.split(len(imgs) // 2, dim=0) for feat in all_feats]
        #ref_feats = self.detector.extract_feat(ref_img)[0]
        ###################################################################
        # img = torch.cat([img, ref_img], dim=0) 
        # gt_coords = gt_coords + ref_gt_coords
        # gt_coords = [gc.unsqueeze(0) for gc in gt_coords]
        # gt_coords = torch.cat(gt_coords, dim=0)
        # img_metas = img_metas + ref_img_metas
       
        ###################################################################
        #transformer and output heads using first frame (img)
        with torch.no_grad():
            feats = self.detector.extract_feat(img)[0]
        query_embeds = bbox_head.query_embedding.weight 
        query_embeds = bbox_head.forward_transformer(
            feats, query_embeds, img_metas
        )

        coord_embeds = self.ctn(query_embeds)
        coord_embeds = self.pool(query_embeds)
        coord_embeds = coord_embeds.mean(dim=0) #bs x 2 x 256
        B, Nq, D = coord_embeds.shape
        coord_embeds = coord_embeds.reshape(B*Nq, D)
        coord_preds = self.coord_pred_head(coord_embeds).sigmoid()
        mse_loss = self.dist_fn(coord_preds, gt_coords).mean()
        losses = {'loss_mse_frame1': mse_loss}

        with torch.no_grad():
            feats = self.detector.extract_feat(ref_img)[0]
        query_embeds = bbox_head.forward_transformer(
            feats, coord_embeds, img_metas
        )
        
        coord_embeds = self.ctn(query_embeds)
        coord_embeds = coord_embeds.mean(dim=0) #bs x 2 x 256
        coord_preds = self.coord_pred_head(coord_embeds).sigmoid()
        mse_loss = self.dist_fn(coord_preds, ref_gt_coords).mean()
        losses['loss_mse_frame2'] = mse_loss
        # coord_preds = coord_preds.mean(dim=0).mean(dim=1).sigmoid()
        

        return losses
        # cls_scores = bbox_head.fc_cls(query_embeds)
        # bbox_preds = bbox_head.fc_reg(bbox_head.activate(bbox_head.reg_ffn(query_embeds))).sigmoid()
        
        cls_scores = self.fc_cls(query_embeds)
        bbox_preds = self.fc_coord(bbox_head.activate(bbox_head.reg_ffn(query_embeds))).sigmoid()


        # cls_scores, bbox_preds = bbox_head.forward_output_heads(
            # query_embeds, init_ref, inter_ref
        # )
        query_embeds = query_embeds[-1].squeeze()
        cls_prob = F.softmax(cls_scores[-1].squeeze(), dim=-1)
        cls_prob = torch.cat([
            cls_prob[:, 0:-2].sum(dim=-1).unsqueeze(-1),
            cls_prob[:, -1].unsqueeze(-1)
        ], dim=-1)
        
        coord_pred = bbox_preds[-1].squeeze()[:, 0:2]

        dists = self.dist_fn(coord_pred, gt_coords)
        matches = linear_assignment(dists.detach().cpu().numpy()) 


        mse_loss_val = 0
        labels = gt_coords.new_ones(len(coord_pred)).long()
        for (pred_idx, gt_idx) in matches:
            labels[pred_idx] = 0
            mse_loss_val += dists[pred_idx].squeeze()
        mse_loss_val = mse_loss_val / len(matches)


        nll_loss_vals = self.nll_loss(cls_prob.log(), labels)
        losses = {'loss_nll': nll_loss_vals.mean(), 'loss_mse': mse_loss_val}
        return losses
            


        ###################################################################

                
        ###################################################################
        #calc loss
        pred_idx, gt_idx = self.assign(
            bbox_pred, cls_score, 
            gt_bboxes, gt_labels, img_metas
        )
        
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
            num_pos, num_neg, bbox_head.bg_cls_weight,
            img_metas
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

        #concat all embeds and run through transformer and output heads
        all_embeds = torch.cat([bbox_head.query_embedding.weight, track_embeds], dim=0)

        all_embeds = bbox_head.forward_transformer(
            ref_feats, all_embeds, img_metas
        )
        ref_cls_scores = bbox_head.fc_cls(all_embeds)
        ref_bbox_preds = bbox_head.fc_reg(bbox_head.activate(bbox_head.reg_ffn(all_embeds))).sigmoid()

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
        
        # label_weights[0:num_query] = 0.1
        loss_cls, loss_bbox, loss_iou = self.detr_loss(
            ref_cls_score.unsqueeze(0), ref_bbox_pred.unsqueeze(0),
            labels, label_weights,
            bbox_targets, bbox_weights,
            num_pos, num_neg, bbox_head.bg_cls_weight, img_metas
        )

        losses['ref.loss_cls'] = loss_cls
        losses['ref.loss_bbox'] = loss_bbox
        losses['ref.loss_iou'] = loss_iou
        return losses

    def detr_loss(self, cls_scores, bbox_preds,
                    labels, label_weights,
                    bbox_targets, bbox_weights,
                    num_total_pos, num_total_neg,
                    bg_cls_weight=1.0,
                    img_metas=None):
        bbox_head = self.detector.bbox_head

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_scores = cls_scores.reshape(-1, bbox_head.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0
        cls_avg_factor += num_total_neg * bg_cls_weight
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
        if img_metas[0]['frame_id'] == 0:
            self.reset()

        img_metas[0]['batch_input_shape'] = (img.shape[2], img.shape[3])
        feats = self.detector.extract_feat(img)

        track_embeds = img.new_empty(0, 256)
        if len(self.tracks) > 0:
            track_embeds = torch.cat([t.state for t in self.tracks], dim=0)  
        
        bbox_head = self.detector.bbox_head
        query_embeds = torch.cat([
            bbox_head.query_embedding.weight, 
            track_embeds
        ], dim=0)
        
        #if self.frame_count > 0:
        #    track_embeds = tracker.qdropout.eval()(track_embeds)
            
        query_embeds = bbox_head.forward_transformer(
            feats[0], query_embeds, img_metas
        )
        cls_scores = bbox_head.fc_cls(query_embeds)
        bbox_preds = bbox_head.fc_reg(bbox_head.activate(bbox_head.reg_ffn(query_embeds))).sigmoid()
        query_embeds = query_embeds[-1].squeeze()
        cls_score = torch.softmax(cls_scores[-1].squeeze(), dim=-1)
        bbox_pred = bbox_preds[-1].squeeze()

        #isolate preds from existing tracks
        track_score = cls_score[self.num_queries:]
        track_embeds = query_embeds[self.num_queries:]
        track_bbox_pred = bbox_pred[self.num_queries:]
        
        #check if alive or should be put to sleep
        alive_mask = track_score[:, 0] >= self.thres_cont
        alive_idx = torch.arange(len(alive_mask))[alive_mask]
        sleep_idx = torch.arange(len(alive_mask))[~alive_mask]
        
        for idx in sleep_idx:
            self.sleeping_tracks.append(self.tracks[idx])
        
        #update alive tracks
        for idx in alive_idx:
            self.tracks[idx].update(track_embeds[idx], track_bbox_pred[idx], track_score[idx][0])
        self.tracks = [t for i, t in enumerate(self.tracks) if i in alive_idx] #remove sleeping tracks
        
        #check for new detections
        new_mask  = cls_score[0:self.num_queries, 0] >= self.thres_new #person class probability
        new_idx = torch.arange(len(new_mask))[new_mask]
        new_embeds = query_embeds[0:self.num_queries][new_idx]
        new_bbox_preds = bbox_pred[0:self.num_queries][new_idx]
        new_cls_score = cls_score[0:self.num_queries][new_idx]

        #new detections may be sleeping objects
        sleeping_embeds = new_embeds.new_empty(0, 256)
        if len(self.sleeping_tracks) > 0:
            sleeping_embeds = torch.cat([t.embed for t in self.sleeping_tracks])
            
                
        dists = torch.cdist(new_embeds, sleeping_embeds) #num_new x num_sleep
        sleep_rm_idx = []
        for i in range(len(new_embeds)):
            matched = False
            for j in range(len(sleeping_embeds)):
                dist = dists[i, j]
                if dist <= self.thres_reid and j not in sleep_rm_idx:
                    old_track = self.sleeping_tracks[j]
                    old_track.update(new_embeds[i], new_bbox_preds[i], new_cls_score[i][0])
                    self.tracks.append(old_track)
                    matched = True
                    sleep_rm_idx.append(j)
                    break
            if not matched:
                new_embed = query_embeds[0:self.num_queries][new_idx[i]]
                new_bbox_pred = bbox_pred[0:self.num_queries][new_idx[i]]
                new_score = bbox_pred[0:self.num_queries][new_idx[i]][0]
                new_track = Track(new_embed, new_bbox_pred, new_score)
                self.tracks.append(new_track)
        
        self.sleeping_tracks = [t for i, t in enumerate(self.sleeping_tracks) if i not in sleep_rm_idx]
        # track_bboxes = torch.cat([t.bbox.unsqueeze(0) for t in self.tracks])
        # track_ages = torch.tensor([t.age for t in self.tracks]).long().cuda()
        # selected_idx = nms(track_bboxes, -track_ages)
        # self.tracks = [t for i, t in enumerate(self.tracks) if i in selected_idx]
        
        bboxes, ids, scores = [bbox_preds.new_empty(0,4)], [], []
        for track in self.tracks:
            onstreak = track.hit_streak >= self.min_hits
            #warmingup = self.frame_count <= self.min_hits
            #if track.wasupdated and (onstreak or warmingup):
            if onstreak:
                bbox = bbox_cxcywh_to_xyxy(track.bbox)#.unsqueeze(0)
                bboxes.append(bbox)
                scores.append(track.score)
                ids.append(track.id)

        bboxes = torch.cat(bboxes, dim=0)
        scores = bboxes.new_tensor(scores).unsqueeze(1)
        ids = torch.tensor(ids)

        H, W , _ = img_metas[0]['ori_shape']
        factor = bboxes.new_tensor([W, H, W, H]).unsqueeze(0)
        bboxes = bboxes * factor
        bboxes = torch.cat([bboxes, scores], dim=-1)

        # if len(torch.unique(ids)) != len(ids):
            # import ipdb; ipdb.set_trace() # noqa
        labels = ids.new_zeros(ids.shape)
        track_res = outs2results(bboxes=bboxes, labels=labels, ids=ids, num_classes=1)
        det_res = outs2results(bboxes=bboxes, labels=labels, ids=None, num_classes=1)
        self.frame_count += 1
        results = dict(
            det_bboxes=det_res['bbox_results'],
            track_bboxes=track_res['bbox_results']
        )
        # results = dict(
            # track_bboxes=[bboxes.cpu()],
            # track_ids=[ids.cpu()],
            # track_labels=[labels.cpu()]
        # )

        return results

class Track:
    count = 0
    def __init__(self, embed, bbox, score):
        self.embed = embed.unsqueeze(0)
        self.bbox = bbox.unsqueeze(0)
        self.score = score.item()
        self.id = Track.count
        Track.count += 1
        self.hit_streak = 0
        self.age = 0
        self.time_since_update = 1
    
    @property
    def state(self):
        return self.embed
        #return torch.cat([self.pos, self.embed], dim=-1)
    
    @property
    def pred(self):
        return self.bbox
    
    def update(self, embed, bbox, score):
        self.embed = embed.unsqueeze(0)
        self.bbox = bbox.unsqueeze(0)
        self.score = score.item()
        self.time_since_update = 0
        self.hit_streak += 1
        self.age += 1
        
    @property
    def wasupdated(self):
        return self.time_since_update < 1
