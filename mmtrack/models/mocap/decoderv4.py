# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
import lap 
from mmdet.models import build_detector, build_head, build_backbone, build_neck
from collections import OrderedDict
import torch.distributed as dist
from mmtrack.core import outs2results, results2outs
from mmcv.runner import BaseModule, auto_fp16
from ..builder import MODELS, build_tracker, build_model
from mmdet.core import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy, reduce_mean
import copy
import torch.distributions as D
from .base import BaseMocapModel
from mmdet.models import build_loss
from cad.pos import AnchorEncoding
from cad.attn import ResCrossAttn, ResSelfAttn
from cad.models.detr import DETRDecoder
from collections import defaultdict
from mmtrack.models.mot.kalman_track import MocapTrack
import time
from mmcv.cnn.bricks.registry import FEEDFORWARD_NETWORK, ATTENTION
from mmcv import build_from_cfg
from pyro.contrib.tracking.measurements import PositionMeasurement
from mmtrack.models.mocap.tracker import Tracker

def calc_grid_loss(dist, grid, scale=1):
    No, G, f = grid.shape
    grid = grid.reshape(No*G, 2)
    log_grid_pdf = dist.log_prob(grid.unsqueeze(1)) * scale
    log_grid_pdf = log_grid_pdf.reshape(No, G, -1)
    logsum = torch.logsumexp(log_grid_pdf, dim=1).t()
    return logsum


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
                 backbone_cfgs=None,
                 model_cfgs=None,
                 bce_target=0.99,
                 remove_zero_at_train=True,
                 output_head_cfg=dict(type='OutputHead',
                     include_z=False,
                     predict_full_cov=True,
                     cov_add=0.01,
                     predict_rotation=True
                 ),
                 time_attn_cfg=dict(type='DeformableAttention2D',
                     qk_dim=256,
                     num_heads=8, 
                     num_levels=2,
                     num_ref_points=4,
                     attn_drop=0.1,
                 ),
                 spatial_attn_cfg=dict(type='DeformableAttention2D',
                     qk_dim=256,
                     num_heads=8, 
                     num_levels=5,
                     num_ref_points=4,
                     attn_drop=0.1,
                 ),
                 num_output_sa_layers=6,
                 max_age=5,
                 min_hits=3,
                 track_eval=False,
                 mse_loss_weight=0.0,
                 pos_loss_weight=0.1,
                 grid_loss=False,
                 num_queries=None,
                 grid_size=(10,10),
                 match_by_id=False,
                 autoregressive=False,
                 global_ca_layers=1,
                 mod_dropout_rate=0.0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = Tracker()
        self.num_classes = 2
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = None
        self.frame_count = 0 
        self.track_eval = track_eval
        self.mse_loss_weight = mse_loss_weight
        self.pos_loss_weight = pos_loss_weight
        self.grid_loss = grid_loss
        self.prev_frame = None
        # self.include_z = include_z
        # self.remove_zero_at_train = remove_zero_at_train
        self.bce_target = bce_target
        # self.register_buffer('mean_scale', torch.tensor(mean_scale))
        # self.add_grid_to_mean = add_grid_to_mean
        self.match_by_id = match_by_id
        self.autoregressive = autoregressive
        self.mod_dropout = nn.Dropout3d(mod_dropout_rate)
        
        self.output_head = build_model(output_head_cfg)
        
        self.backbones = nn.ModuleDict()
        for key, cfg in backbone_cfgs.items():
            self.backbones[key] = build_backbone(cfg)
        
        self.models = nn.ModuleDict()

        # for key, cfg in model_cfgs.items():
            # mod, node = key
            # self.models[mod + '_' + node] = build_model(cfg)
        
        self.mse_loss = nn.MSELoss(reduction='none')

        self.room_queries = nn.Embedding(50*70, 256)
        
        self.num_queries = num_queries
        # if self.num_queries is not None:
            # self.global_pos_encoding = nn.Embedding(self.num_queries, 256)
        # else: 
            # self.global_pos_encoding = AnchorEncoding(dim=256, grid_size=grid_size, learned=False, out_proj=False)
        
        # self.global_ca_layers = global_ca_layers
        # if global_ca_layers > 1:
            # self.global_cross_attn = nn.ModuleList([ResCrossAttn(cross_attn_cfg)]*global_ca_layers)
        # else:
            # self.global_cross_attn = ResCrossAttn(cross_attn_cfg)

        # self.time_attn = build_from_cfg(time_attn_cfg, ATTENTION)
        self.spatial_attn = nn.Sequential(
                build_from_cfg(spatial_attn_cfg, ATTENTION),
                #build_from_cfg(spatial_attn_cfg, ATTENTION),
                # build_from_cfg(spatial_attn_cfg, ATTENTION),
                # build_from_cfg(spatial_attn_cfg, ATTENTION),
                # build_from_cfg(spatial_attn_cfg, ATTENTION),
                # build_from_cfg(spatial_attn_cfg, ATTENTION)
        )
        
        self.bce_loss = nn.BCELoss(reduction='none')


    
    
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
            if self.track_eval:
                return self.forward_track(data, **kwargs)
            else:
                return self.forward_test(data, **kwargs)
    
    def forward_track(self, datas, return_unscaled=False, **kwargs):
        det_embeds = [self._forward_single(data) for data in datas]
        views = [x[0].unsqueeze(0) for x in det_embeds[0]]

        #det_embeds = torch.stack(det_embeds, dim=0)[-1] # 1 x L x D
        
        is_first_frame = False
        if self.prev_frame is None:
            roomQ = self.room_queries.weight.unsqueeze(0)
            roomQ = roomQ.view(1, 50, 70, 256)
            roomQ = roomQ.permute(0,3,1,2)

            #track_embeds = self.global_pos_encoding.weight.unsqueeze(0) # 1 x L x D
            is_first_frame = True
        else:
            roomQ = self.prev_frame['embeds']
            # track_embeds = self.prev_frame['embeds']

        # if self.global_ca_layers > 1:
            # for layer in self.global_cross_attn:
                # track_embeds = layer(track_embeds, det_embeds)
        # else:
            # track_embeds, A = self.global_cross_attn(track_embeds, det_embeds, return_weights=True)

        out = self.spatial_attn(views + [roomQ])
        roomQ = out[-1]

        output = self.output_head(roomQ)
        pred_rot = output['rot']
        pred_dist = output['dist']

        if is_first_frame:
            result = {
                'track_means': pred_dist.loc[0].detach().cpu(),
                'track_covs': pred_dist.covariance_matrix[0].detach().cpu(),
                'track_obj_probs': torch.ones(self.num_queries).float(),
                'track_ids': torch.arange(len(pred_dist.loc[0])),
                'slot_ids': torch.arange(len(pred_dist.loc[0])),
                'track_rot': pred_rot.cpu()[0],
                #'attn_weights': A.cpu()[0]
            }
            self.prev_frame = {'dist': pred_dist, 'rot': pred_rot, 'embeds': roomQ.detach(), 'ids': torch.arange(2)}
            # self.prev_frame = {'dist': pred_dist, 'embeds': track_embeds.detach(), 'ids': torch.arange(2)}
            return result

        
        prev_dist = self.prev_frame['dist']
        prev_rot = self.prev_frame['rot']

        prev_mean, prev_cov = prev_dist.loc[0], prev_dist.covariance_matrix[0]
        pred_mean, pred_cov = pred_dist.loc[0], pred_dist.covariance_matrix[0]
        new_ids = torch.zeros(len(prev_mean))
        if self.num_queries > 1:


            kl_vals = torch.zeros(2,2).cuda()
            for i in range(2):
                p = D.MultivariateNormal(pred_mean[i], pred_cov[i])
                for j in range(2):
                    q = D.MultivariateNormal(prev_mean[j], pred_cov[j])
                    kl_vals[i,j] = torch.distributions.kl_divergence(p,q)
            
            rot_scores = torch.cdist(pred_rot[0], prev_rot[0])
            scores = kl_vals + 100 * rot_scores
            assign_idx = linear_assignment(scores)

            prev_ids = self.prev_frame['ids']
            for pred_idx, prev_idx in assign_idx:
                new_ids[pred_idx] = prev_ids[prev_idx]
        
        self.prev_frame['ids'] = new_ids
        self.prev_frame['dist'] = pred_dist
        self.prev_frame['rot'] = pred_rot
        self.prev_frame['embeds'] = roomQ

        # det_mean, det_cov = dist.loc, dist.covariance_matrix
        # det_obj_probs = output['obj_logits']
        # det_mean, det_cov, det_obj_probs = det_mean[0], det_cov[0], det_obj_probs[0].squeeze()

        # track_mean, track_cov, _ = self.convert(curr)
        # track_mean, track_cov = track_mean[0], track_cov[0]

        result = {
            'track_means': pred_mean.detach().cpu(),
            'track_covs': pred_cov.detach().cpu(),
            'track_obj_probs': torch.ones(2).float(),
            'track_ids': new_ids,
            'slot_ids': torch.arange(2),
            'track_rot': pred_rot.cpu()[0],
            #'attn_weights': A.cpu()[0]
        }
        # result = self.tracker(result)


        return result

    def forward_train(self, datas, return_unscaled=False, **kwargs):
        losses = defaultdict(list)
        mocaps = [d[('mocap', 'mocap')] for d in datas]
        mocaps = mmcv.parallel.collate(mocaps)

        gt_positions = mocaps['gt_positions']
        gt_positions = gt_positions.transpose(0,1)
        

        # B, T, L, f = gt_positions.shape
        # gt_positions = gt_positions.reshape(B*T, L, f)
        # gt_positions = gt_positions.reshape(T*B, N, C)

        gt_ids = mocaps['gt_ids']
        T, B, f = gt_ids.shape
        gt_ids = gt_ids.transpose(0,1)
        # gt_ids = gt_ids.reshape(T*B, f)

        gt_grids = mocaps['gt_grids']
        T, B, N, Np, f = gt_grids.shape
        gt_grids = gt_grids.transpose(0,1)
        # gt_grids = gt_grids.reshape(T*B, N, Np, f)
        
        gt_rots = mocaps['gt_rot']
        gt_rots = gt_rots.transpose(0,1)
        
        
        angles = torch.zeros(B, T, 2).cuda()
        for i in range(B):
            for j in range(T):
                for k in range(2):
                    rot = gt_rots[i,j,k]

                    if rot[4] <= 0:
                        rads = torch.arcsin(rot[3]) / (2*torch.pi)
                    else:
                        rads = torch.arcsin(rot[1]) / (2*torch.pi)
                    angles[i,j, k] = rads
        gt_rots = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)



        # gt_rots = torch.cat([gt_rots[..., 0:2], gt_rots[..., 3:5]], dim=-1)
        
        
        all_embeds = [self._forward_single(data) for data in datas]
        roomQ = self.room_queries.weight.unsqueeze(0)
        roomQ = roomQ.view(1, 50, 70, 256)
        roomQ = roomQ.permute(0,3,1,2)
        # all_embeds = torch.stack(all_embeds, dim=0) 
        # all_embeds = all_embeds.transpose(0, 1) #B T L D
        
        # output_vals = self.ctn(final_embeds)
        # output_vals = self.output_head(final_embeds) #B L No
        # output_vals = output_vals.reshape(Nt, B, L, -1)
        # output_vals = output_vals.transpose(0,1) #B x Nt x L x No

        # bs = len(output_vals)
        all_outputs = []
        for i in range(B):
            #global_pos_embeds = self.global_pos_encoding.weight.unsqueeze(0) # 1 x 2 x 256
            roomQ = self.room_queries.weight.unsqueeze(0)
            roomQ = roomQ.view(1, 50, 70, 256)
            roomQ = roomQ.permute(0,3,1,2)
            
            for j in range(T):
                views = [x[i].unsqueeze(0) for x in all_embeds[j]] + [roomQ]
                views = self.spatial_attn(views)
                roomQ = views[-1]
                # import ipdb; ipdb.set_trace() # noqa
                # if self.global_ca_layers > 1:
                    # for layer in self.global_cross_attn:
                        # global_pos_embeds = layer(global_pos_embeds, all_embeds[i,j].unsqueeze(0))
                # else:
                    # global_pos_embeds = self.global_cross_attn(global_pos_embeds, all_embeds[i,j].unsqueeze(0))
                
                
                output = self.output_head(roomQ)
                dist = output['dist']


                pred_rot = output['rot'][0] #No x 2
                rot_dists = torch.cdist(pred_rot, gt_rots[i][j]) #needs tranpose?
                # vel_dists = torch.cdist(output['vel'].squeeze(), vel)

                # curr = self.ctn(global_pos_embeds)
                # mean, cov, obj_logits = self.convert(curr)
                # mean, cov, obj_logits = mean[0], cov[0], obj_logits[0]
                # dist = self.dist(mean, cov)
                grid = gt_grids[i][j]
                grid = grid[0].unsqueeze(0)
                # idx = torch.tensor([1,0]).long()
                # grid = grid[idx]
                No, G, f = grid.shape
                grid = grid.reshape(No*G, 2)
                log_grid_pdf = dist.log_prob(grid.unsqueeze(1)) #* 1.5
                log_grid_pdf = log_grid_pdf.reshape(-1, G, No)
                logsum = torch.logsumexp(log_grid_pdf, dim=1)#.t() #need transpose?
                pos_neg_log_probs = -logsum
                if self.match_by_id:
                    assign_idx = torch.stack([gt_ids[i,j]]*2, dim=-1).cpu()
                else:
                    assign_idx = linear_assignment(pos_neg_log_probs*self.pos_loss_weight + rot_dists)
                if len(logsum) == 1: #one object
                    assign_idx = torch.zeros(1, 2).long()
                
                pos_loss, rot_loss, vel_loss, count = 0, 0, 0, 0
                for pred_idx, gt_idx in assign_idx:
                    pos_loss += pos_neg_log_probs[pred_idx, gt_idx]
                    rot_loss += rot_dists[pred_idx, gt_idx]
                    # vel_loss += vel_dists[pred_idx, gt_idx]
                    # rot_loss += torch.norm(pred_rot[pred_idx] - gt_rots[i][j][gt_idx])
                    count += 1
                pos_loss /= count
                rot_loss /= count
                pos_loss = pos_loss * self.pos_loss_weight
                # pos_loss = F.relu(pos_loss)
                # if pos_loss < 0:
                    # import ipdb; ipdb.set_trace() # noqa
                rot_loss = rot_loss #* 0.1
                losses['pos_loss'].append(pos_loss)
                losses['rot_loss'].append(rot_loss)
                # losses['vel_loss'].append(vel_loss)

        losses = {k: torch.stack(v).mean() for k, v in losses.items()}
        return losses

    def _forward_single(self, data, return_unscaled=False, **kwargs):
        inter_embeds = []
        for key in data.keys():
            mod, node = key
            if mod == 'mocap':
                continue
            if mod not in self.backbones.keys():
                continue
            backbone = self.backbones[mod]
            try:
                feats = backbone(data[key]['img'])
            except:
                feats = backbone([data[key]['img']])
            
            # model = self.models[mod + '_' + node]
            # embeds = model(feats)
            embeds = feats[0]
            inter_embeds.append(embeds)

        if len(inter_embeds) == 0:
            import ipdb; ipdb.set_trace() # noqa
        num_mods = len(inter_embeds)
        inter_embeds = torch.stack(inter_embeds, dim=1)
        inter_embeds = self.mod_dropout(inter_embeds)
        inter_embeds = [inter_embeds[:, i] for i in range(num_mods)]
        return inter_embeds
        B, Nmod, L, D = inter_embeds.shape
        inter_embeds = inter_embeds.reshape(B, Nmod*L, D)
        # inter_embeds = torch.cat(inter_embeds, dim=-2)
        return inter_embeds

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
        
        num_samples = len(data[0][('mocap', 'mocap')]['gt_positions'])

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
