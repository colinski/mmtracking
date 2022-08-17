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
from mmcv.runner import BaseModule, auto_fp16
from ..builder import MODELS, build_tracker
from mmdet.core import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy, reduce_mean
from .base import BaseMocapModel
from collections import defaultdict

@MODELS.register_module()
class KNNMocapModel(BaseMocapModel):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.dummy_param = nn.Parameter(torch.zeros(1))

        self.buckets = {}
        for x in range(100):
            for y in range(100):
                for z in range(100):
                    key = (x / 100, y / 100, z / 100)
                    self.buckets[key] = defaultdict(list)

    
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

    def find_nearest(self, name, img):
        results = []
        for k, v in self.buckets.items():
            if len(v.keys()) == 0 or name not in v.keys():
                continue
            try:
                imgs = torch.cat(v[name])
                diffs = (img - imgs)**2
            except:
                continue
            diffs = diffs.flatten(1)
            mse = diffs.mean(dim=-1).mean()
            results.append([k[0], k[1], k[2], mse])
        results = torch.tensor(results)
        min_idx = torch.argmin(results[:, -1])
        return results[min_idx]

    def forward_test(self, data, **kwargs):
        results = []
        for key, val in data.items():
            if key == 'mocap' or key == 'ind':
                continue
            img = val['img'].data.cpu()
            res = self.find_nearest(key, img)
            results.append(res)
        if len(results) == 0:
            return {
                'pred_position': np.zeros((1, 3)),
                'gt_position': data['mocap']['gt_positions'][0][-2].cpu().numpy()
            }         

        results = torch.stack(results)
        pred = results.mean(dim=0)[0:3]
        return {
            'pred_position': pred.unsqueeze(0).cpu().detach().numpy(),
            'gt_position': data['mocap']['gt_positions'][0][-2].cpu().numpy()
        }         

            # for k, v in self.buckets.items():
                # if len(v.keys()) == 0 or key not in v.keys():
                    # continue
                # imgs = torch.cat(v[key])
                # diffs = (img - imgs)**2
                # diffs = diffs.flatten(1)
                # mse = diffs.mean(dim=-1).mean()
                # import ipdb; ipdb.set_trace() # noqa
        # if 'azimuth_static' not in data.keys():
            # return {
                # 'pred_position': np.zeros((1, 3)),
                # 'gt_position': data['mocap']['gt_positions'][0][-2].cpu().numpy()
            # }         
   
        img = data['azimuth_static']['img'].data.cpu()
        for k, v in self.buckets.items():
            if len(v) == 0:
                continue
            imgs = torch.cat(v)
            diffs = (imgs - img)**2
            diffs = diffs.flatten(1)
            mse = diffs.mean(axis=-1).mean()
            results.append([k[0], k[1], k[2], mse])
        results = torch.tensor(results)
        min_idx = torch.argmin(results[:, -1])
        return {
            'pred_position': results[min_idx][0:3].unsqueeze(0).cpu().detach().numpy(),
            'gt_position': data['mocap']['gt_positions'][0][-2].cpu().numpy()
        }         

    def forward_train(self, data, **kwargs):
        x, y, z = data['mocap']['gt_positions'][0][-2]
        x, y, z = round(x.item(), 2), round(y.item(), 2), round(z.item(), 2)
        for key, val in data.items():
            if key == 'mocap' or key == 'ind':
                continue
            img = val['img'].data.cpu()
            self.buckets[(x, y, z)][key].append(img)
        # if 'azimuth_static' in data.keys():
            # img = data['azimuth_static']['img'].data.cpu()
            # self.buckets[(x, y, z)].append(img)
        return {'dummy_loss': self.dummy_param}

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

