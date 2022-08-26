# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lap 
from mmdet.models import build_detector, build_head
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

def linear_assignment(cost_matrix):
    cost_matrix = cost_matrix.cpu().detach().numpy()
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    assign_idx = np.array([[y[i], i] for i in x if i >= 0])
    sort_idx = np.argsort(assign_idx[:, 1])
    assign_idx = assign_idx[sort_idx]
    assign_idx = torch.from_numpy(assign_idx)
    return assign_idx.long()

@MODELS.register_module()
class TransformerMocapModel(BaseMocapModel):
    def __init__(self,
                 detector=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if detector is not None:
            self.img_detector = build_detector(detector)
            self.depth_detector = copy.deepcopy(self.img_detector)
            self.shared_head = copy.deepcopy(self.img_detector.bbox_head)

        
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
            gamma=2.0, alpha=0.25, reduction='mean',
            loss_weight=1.0, activated=False)

        self.focal_loss = build_loss(focal_loss_cfg)


    
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
            
            bbox_head = self.img_detector.bbox_head
            with torch.no_grad():
                feats = self.img_detector.extract_feat(img)[0]
                query_embeds = bbox_head.query_embedding.weight 
                query_embeds_img = bbox_head.forward_transformer(
                    feats, query_embeds, img_metas
                )
                query_embeds_img = self.pool(query_embeds_img)
                # query_embeds_img = query_embeds_img[:, :, 0:5]
        else:
            print('using prior')
            bs = 1
            bbox_head = self.img_detector.bbox_head
            query_embeds_img = bbox_head.query_embedding.weight.unsqueeze(0).detach()

        # img = data['zed_camera_left']['img']
        # img_metas = data['zed_camera_left']['img_metas']
        # img_metas[0]['batch_input_shape'] = (img.shape[2], img.shape[3])
        # bs = img.shape[0]
        
        # bbox_head = self.img_detector.bbox_head
        # with torch.no_grad():
            # feats = self.img_detector.extract_feat(img)[0]
        # query_embeds = bbox_head.query_embedding.weight 
        # query_embeds_img = bbox_head.forward_transformer(
            # feats, query_embeds, img_metas
        # ).mean(dim=0)
       
        final_embeds = query_embeds_img
        final_embeds = self.ctn(final_embeds).mean(dim=0)

        mean = self.mean_head(final_embeds)
        cov = self.cov_head(final_embeds)
        cls_logits = self.cls_head(final_embeds)
        obj_logits = self.obj_head(final_embeds)
        return mean, cov, cls_logits, obj_logits


    def forward_test(self, data, **kwargs):
        mean, cov, cls_logits, obj_logits = self._forward(data, **kwargs)
        # pred_pos = mean.mean(dim=1)
        #dist = self.dist(mean[0][0], cov[0][0])
        # dist = D.Independent(dist, 1) #Nq independent Gaussians

        obj_probs = F.sigmoid(obj_logits[0]).squeeze()
        is_obj = obj_probs >= 0.5
        # print(obj_probs.min(), obj_probs.max(), obj_probs.mean(), is_obj.sum())
        mean = mean[:, is_obj]
        cov = cov[:, is_obj]
        
        dist = self.dist(mean[0], cov[0])
        dist = D.Independent(dist, 1) #Nq independent Gaussians

        pred_pos = dist.sample([10])#.unsqueeze(0)


        # pred_pos = mean[0][0].unsqueeze(0)
        result = {
            #'pred_position': pred_pos.cpu().detach().numpy(),
            'pred_position': mean[0].cpu().detach().unsqueeze(0).numpy()
            # 'pred_cov': cov[0].cpu().detach().unsqueeze(0).numpy()
            #'pred_obj_prob': obj_probs.cpu().detach().unsqueeze(0).numpy()
        }
        return result

        # assert len(mean) == 1
        # mean, cov, cls_logits, obj_logits = mean[0], cov[0], cls_logits[0], obj_logits[0]
        # cls_probs = F.softmax(cls_logits, dim=-1)
        # obj_probs = F.softmax(obj_logits, dim=-1)

        # is_obj = obj_probs[:, 1] >= 0.0
        # cls_probs = cls_probs[is_obj]
        # mean = mean[is_obj]
        # cov = cov[is_obj]
        # dist = self.dist(mean, cov)
        # dist = D.Independent(dist, 1) #Nq independent Gaussians
        
        # return {
            # 'pred_position': mean.cpu().detach().numpy(),
            # 'gt_position': data['mocap']['gt_positions'][0][-2].cpu().numpy()
        # }         

         
        # return {
            # 'position': mean.cpu().detach().numpy(),
            # 'cls_probs': cls_probs.cpu().detach().numpy()
        # }         

    def forward_train(self, data, **kwargs):
        # if 'zed_camera_left' in data.keys():
            # img = data['zed_camera_left']['img']
            # img_metas = data['zed_camera_left']['img_metas']
            # img_metas[0]['batch_input_shape'] = (img.shape[2], img.shape[3])
            # bs = img.shape[0]
            
            # bbox_head = self.img_detector.bbox_head
            # with torch.no_grad():
                # feats = self.img_detector.extract_feat(img)[0]
            # query_embeds = bbox_head.query_embedding.weight 
            # query_embeds_img = bbox_head.forward_transformer(
                # feats, query_embeds, img_metas
            # ).mean(dim=0)
        # else:
            # bs = 1
            # bbox_head = self.img_detector.bbox_head
            # query_embeds_img = bbox_head.query_embedding.weight.unsqueeze(0)
        losses = {}
        mean, cov, cls_logits, obj_logits = self._forward(data, **kwargs)


        # img = data['zed_camera_depth']['img']
        # img_metas = data['zed_camera_depth']['img_metas']
        # img_metas[0]['batch_input_shape'] = (img.shape[2], img.shape[3])
        # img = img.expand(-1, 3, -1, -1)
        
        # bbox_head = self.depth_detector.bbox_head
        # with torch.no_grad():
            # feats = self.depth_detector.extract_feat(img)[0]
        # query_embeds = bbox_head.query_embedding.weight 
        # query_embeds_depth = bbox_head.forward_transformer(
            # feats, query_embeds, img_metas
        # ).mean(dim=0)

        # final_embeds = torch.cat([query_embeds_img, query_embeds_depth], dim=1)

        # final_embeds = query_embeds_img
        # final_embeds = self.ctn(final_embeds)

        # mean = self.mean_head(final_embeds)
        # cov = self.cov_head(final_embeds)
        # cls_logits = self.cls_head(final_embeds)
        # obj_logits = self.obj_head(final_embeds)

        bs = len(mean)
        dist = self.dist(mean[0], cov[0])
        dist = D.Independent(dist, 1) #Nq independent Gaussians

        gt_pos = data['mocap']['gt_positions'][0]#[-2].unsqueeze(0)
        gt_labels = data['mocap']['gt_labels'][0]#[-2].unsqueeze(0)

        is_node = gt_labels == 0
        gt_pos = gt_pos[~is_node]
        gt_labels = gt_labels[~is_node]

        def calc_mse(pred, gt):
            return (pred - gt)**2

        # pos_log_probs = [(mean[0] - pos)**2 for pos in gt_pos]
        # pos_log_probs = torch.stack(pos_log_probs, dim=0).mean(dim=-1) #num_objs x Nq
        # pos_neg_log_probs = pos_log_probs.t() #Nq x num_objs
         
        pos_log_probs = [dist.log_prob(pos) for pos in gt_pos]
        pos_neg_log_probs = -torch.stack(pos_log_probs, dim=-1) #Nq x num_objs
        assign_idx = linear_assignment(pos_neg_log_probs)

        pos_loss = 0
        for pred_idx, gt_idx in assign_idx:
            pos_loss += pos_neg_log_probs[pred_idx, gt_idx]
            # cls_loss += corr_log_probs[pred_idx, gt_idx]
        pos_loss /= len(assign_idx)
        # pos_loss /= 10

        # obj_neg_log_probs = -F.log_softmax(obj_logits[0], dim=-1)
        low_end = 0.2
        obj_targets = pos_neg_log_probs.new_zeros(len(pos_neg_log_probs)) + low_end
        obj_targets = obj_targets.float()
        obj_targets[assign_idx[:, 0]] = 1.0 - low_end
        obj_probs = F.sigmoid(obj_logits[0])
        obj_log_probs = F.logsigmoid(obj_logits[0]).squeeze()

        

        bce_loss = torch.nn.BCELoss(reduction='none')
        obj_loss_vals = bce_loss(obj_probs.squeeze(), obj_targets)
        # obj_loss = obj_loss_vals[obj_targets == 0.2].mean() + obj_loss_vals[obj_targets == 0.8].mean()

        losses['pos_obj_loss'] = obj_loss_vals[obj_targets == 1.0 - low_end].mean()
        losses['neg_obj_loss'] = obj_loss_vals[obj_targets == low_end].mean()

        # obj_loss = self.focal_loss(obj_logits[0], obj_targets)

        # obj_loss = (obj_targets.float() - obj_probs.squeeze()).abs()
        # obj_loss = obj_loss.mean()

        # sum_obj_probs = obj_probs.sum()
        # count_loss = (sum_obj_probs - len(gt_pos)).abs()


        # corr_probs = [obj_neg_log_probs[idx, target] for idx, target in enumerate(obj_targets)] 
        # corr_probs = torch.stack(corr_probs)
        # obj_loss = corr_probs.sum() #/ len(pos_log_probs)#len(assign_idx)

        # cls_loss /= len(assign_idx)


        #return {'pos_loss': pos_loss, 'obj_loss': obj_loss, 'count_loss': count_loss}
        # return {'obj_loss': obj_loss}
        losses['pos_loss'] = pos_loss
        return losses
        
        # pred_pos = mean.mean(dim=1)
        # sq_diff = (gt_pos - pred_pos)**2
        # return {
            # 'pos_loss': sq_diff.mean()
        # }


        bs = len(mean)
        for i in range(bs):
            gt_pos = data['mocap']['gt_positions'][i][-2].unsqueeze(0)
            gt_labels = data['mocap']['gt_labels'][i][-2].unsqueeze(0)

            dist = self.dist(mean[i], cov[i])
            dist = D.Independent(dist, 1) #Nq independent Gaussians

            pos_log_probs = [dist.log_prob(pos) for pos in gt_pos]
            pos_log_probs = -torch.stack(pos_log_probs, dim=-1) #Nq x num_objs
            
            cls_log_probs = F.log_softmax(cls_logits[i], dim=-1) #Nq x num_classes
            corr_log_probs = [cls_log_probs[:, label] for label in gt_labels]
            corr_log_probs = -torch.stack(corr_log_probs, dim=-1) #Nq x num_objs

            combined_probs = pos_log_probs + corr_log_probs

            assign_idx = linear_assignment(combined_probs)
            
            pos_loss, cls_loss = 0, 0
            for pred_idx, gt_idx in assign_idx:
                pos_loss += pos_log_probs[pred_idx, gt_idx]
                cls_loss += corr_log_probs[pred_idx, gt_idx]
            pos_loss /= len(assign_idx)
            cls_loss /= len(assign_idx)
            
            obj_probs = -F.log_softmax(obj_logits[i], dim=-1)
            obj_targets = cls_log_probs.new_zeros(len(cls_log_probs)).long()
            obj_targets[assign_idx[:, 0]] = 1

            corr_probs = [obj_probs[idx, target] for idx, target in enumerate(obj_targets)] 
            corr_probs = torch.stack(corr_probs)
            obj_loss = corr_probs.sum() #/ len(pos_log_probs)#len(assign_idx)

            obj_probs = F.softmax(obj_logits[i], dim=-1)
            sum_obj_probs = obj_probs[:, 1].sum()

            count_loss = (sum_obj_probs - len(gt_pos))**2
            
            loss = {
                'pos_loss': pos_loss,
                # 'cls_loss': cls_loss,
                # 'obj_loss': obj_loss,
                # 'count_loss': count_loss
            }



        # dists = self.dist(mean, cov)

                
        
        # preds = self.coord_pred_head(final_embeds)
        # preds = preds.squeeze()
        # preds = preds.mean(dim=0).mean(dim=0)
    
        # log_probs = dists.log_prob(gt_pos)


        # loss = -(dists.log_prob(gt_pos))
        # loss = (gt_pos - preds)**2
        return loss
        # return {'loss': loss.mean()}
        # gt_ids = data['mocap']['gt_ids'][-2]
        # gt_labels = data['mocap']['gt_labels'][-2]


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

