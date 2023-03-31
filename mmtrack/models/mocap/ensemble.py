import numpy as np
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import lap 
from mmdet.models import build_detector, build_head, build_backbone, build_neck
from collections import OrderedDict
import torch.distributed as dist
#from mmtrack.core import outs2results, results2outs
from mmcv.runner import BaseModule, auto_fp16
from ..builder import MODELS, build_tracker, build_model
#from mmdet.core import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy, reduce_mean
import copy
from .base import BaseMocapModel
from mmdet.models import build_loss
# from cad.pos import AnchorEncoding
# from cad.attn import ResCrossAttn, ResSelfAttn
# from cad.models.detr import DETRDecoder
from collections import defaultdict
#from mmtrack.models.mot.kalman_track import MocapTrack
import time
from mmcv.cnn.bricks.registry import FEEDFORWARD_NETWORK
from mmcv import build_from_cfg
#from pyro.contrib.tracking.measurements import PositionMeasurement
from mmtrack.models.mocap.tracker import Tracker, MultiTracker

class Delist(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[0]

def linear_assignment(cost_matrix):
    cost_matrix = cost_matrix.cpu().detach().numpy()
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    assign_idx = np.array([[y[i], i] for i in x if i >= 0])
    sort_idx = np.argsort(assign_idx[:, 1])
    assign_idx = assign_idx[sort_idx]
    assign_idx = torch.from_numpy(assign_idx)
    return assign_idx.long()

@MODELS.register_module()
class DetectorEnsemble(BaseMocapModel):
    def __init__(self,
                 backbone_cfgs=None,
                 adapter_cfgs=None,
                 output_head_cfg=None,
                 dim=256,
                 track_eval=False,
                 pos_loss_weight=0.1,
                 num_queries=1,
                 match_by_id=False,
                 global_ca_layers=1,
                 mod_dropout_rate=0.0,
                 loss_type='nll',
                 freeze_backbone=False,
                 kf_train=False,
                 cov_only_train=False,
                 init_cfg={},
                 is_audio=False,
                 entropy_loss_weight=0.0,
                 *args,
                 **kwargs):
        super().__init__(init_cfg, *args, **kwargs)
        # config_file = '/home/csamplawski/src/mmdetection/configs/detr/detr_r50_8x2_150e_coco.py'
        # checkpoint_file = '/home/csamplawski/src/mmtracking/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'
        # self.detr = init_detector(config_file, checkpoint_file, device='cuda')  # or device='cuda:0'
        # self.detr = self.detr.eval()
        self.num_classes = 2
        self.tracks = None
        self.frame_count = 0 
        self.track_eval = track_eval
        self.pos_loss_weight = pos_loss_weight
        self.prev_frame = None
        self.dim = dim
        self.match_by_id = match_by_id
        self.is_audio = is_audio
        self.entropy_loss_weight = entropy_loss_weight
        #self.mod_dropout = nn.Dropout2d(mod_dropout_rate)
        #self.tracker = MultiTracker(mode='kf')
        self.loss_type = loss_type
        self.cov_only_train = cov_only_train
        
        output_head_cfg['cov_only_train'] = cov_only_train
        self.output_head = build_model(output_head_cfg)
        self.times = []
        
        self.backbones = nn.ModuleDict()
        for key, cfg in backbone_cfgs.items():
            self.backbones[key] = build_backbone(cfg)
        
        self.adapters = nn.ModuleDict()
        for key, cfg in adapter_cfgs.items():
            mod, node = key
            self.adapters[mod + '_' + node] = build_model(cfg)
        
        self.num_queries = num_queries
        # self.global_pos_encoding = nn.Embedding(self.num_queries, self.dim)
        
        # self.global_ca_layers = global_ca_layers
        # if global_ca_layers > 1:
            # self.global_cross_attn = nn.ModuleList([ResCrossAttn(cross_attn_cfg)]*global_ca_layers)
        # else:
            # self.global_cross_attn = ResCrossAttn(cross_attn_cfg)
        
        # if 'init_cfg' in kwargs:
            # self.init_weights()
        self.freeze_backbone = freeze_backbone
        self.kf_train = kf_train
        self.sessions = None
        # if init_cfg != {}:
            # self.init_weights()

        self.audio_network = nn.Sequential(
            nn.Linear(5,5),
            nn.ReLU()
        )
        
    def forward(self, data, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if self.is_audio:
            return self.forward_train_audio(data, **kwargs)

        if return_loss:
            if self.kf_train:
                return self.forward_train_track(data, **kwargs)
            return self.forward_train(data, **kwargs)
        else:
            if self.track_eval:
                return self.forward_track(data, **kwargs)
            else:
                return self.forward_test(data, **kwargs)

    def forward_track(self, datas, return_unscaled=False, **kwargs):
        preds = {}
        # mocaps = [d[('mocap', 'mocap')] for d in datas]
        # mocaps = mmcv.parallel.collate(mocaps)
        
        # num_timesteps x batch_size x num_objects x (2 or 3)
        #gt_positions = mocaps['gt_positions']
        
        #num_timesteps x batch_size x num_views x D x H x W
        all_outputs = [self._forward_single(data) for data in datas]
        assert len(all_outputs) == 1
        for t, output in enumerate(all_outputs):
            for key, embeds in output.items():
                loss_key = '_'.join(key)
                assert len(embeds) == 1
                for b, embed in enumerate(embeds): 
                    #gt_pos = gt_positions[t, b]
                    embed = embed.unsqueeze(0)
                    output_dict = self.output_head(embed)
                    dist = output_dict['dist']
                    H, W = output_dict['grid_size']
                    # comp_dist = dist.componet_distribution
                    comp_dist = dist.component_distribution
                    mean, cov = comp_dist.loc, comp_dist.covariance_matrix
                    weights = dist.mixture_distribution.probs
                    preds[loss_key] = {
                        'mean': mean.reshape(H,W,2).cpu(),
                        'cov': cov.reshape(H,W,2,2).cpu(),
                        'weights': weights.reshape(H,W).cpu()
                    }
        return preds
    
    def forward_export(self, datas, path, return_unscaled=False, **kwargs):
        from onnxsim import simplify
        import onnx
        import onnxruntime as ort

        for key, val in datas[0].items():
            mod, node = key
            if mod == 'mocap':
                continue
            img = val['img'].data.unsqueeze(0).cuda()
            name = mod + '_' + node
            backbone = self.backbones[mod]
            adapter = self.adapters[name]
            output_head = self.output_head.eval()
            output_head.return_raw = True
            export_model = nn.Sequential(backbone.eval(), Delist(), adapter.eval(), output_head.eval())
            export_model = export_model.cuda().eval()
            onnx_fname = "%s/%s.onnx" % (path,name)
            torch.onnx.export(export_model, img, onnx_fname, verbose=True,
                input_names=['img'], output_names=['mean', 'cov', 'mix_weights'])
            model = onnx.load(onnx_fname)
            model_simp, check = simplify(model)
            onnx.save(model, onnx_fname)

            sess = ort.InferenceSession(onnx_fname, providers=['CPUExecutionProvider'])
            outputs = sess.run(None, {'img': img.cpu().numpy()})
            print(datas[0][('mocap', 'mocap')]['gt_positions'])
            print(outputs)
        return None
    
    def forward_train_audio(self, datas, return_unscaled=False, **kwargs):
        losses = defaultdict(list)
        mocaps = [d[('mocap', 'mocap')] for d in datas]
        mocaps = mmcv.parallel.collate(mocaps)
        
        # num_timesteps x batch_size x num_objects x (2 or 3)
        gt_positions = mocaps['gt_positions']

        #concat and compute RMS on datas
        #apply network
        #compute loss
        #return {'mse_loss': loss}
        
        losses['mse_loss'] = 5
        return losses

    def forward_train(self, datas, return_unscaled=False, **kwargs):
        losses = defaultdict(list)
        mocaps = [d[('mocap', 'mocap')] for d in datas]
        mocaps = mmcv.parallel.collate(mocaps)
        
        # num_timesteps x batch_size x num_objects x (2 or 3)
        gt_positions = mocaps['gt_positions']
        
        #num_timesteps x batch_size x num_views x D x H x W
        with torch.set_grad_enabled(not self.cov_only_train):
            all_outputs = [self._forward_single(data) for data in datas]
        for t, output in enumerate(all_outputs):
            for key, embeds in output.items():
                loss_key = '_'.join(key + ('loss',))
                for b, embed in enumerate(embeds): 
                    gt_pos = gt_positions[t, b]
                    dist = self.output_head(embed.unsqueeze(0))['dist']
                    nll = -dist.log_prob(gt_pos)
                    losses[loss_key].append(nll.mean()) 
                    if self.entropy_loss_weight > 0:
                        num_objs = len(gt_pos)
                        entropy_target = np.log(num_objs)
                        loss_key = '_'.join(key + ('entropy_los',))
                        dist_entropy = dist.mixture_distribution.entropy()
                        entropy_loss = (dist_entropy - entropy_target).abs()
                        losses[loss_key].append(entropy_loss * self.entropy_loss_weight)
                        loss_key = '_'.join(key + ('dist_entropy',))
                        losses[loss_key].append(dist_entropy.detach())


        losses = {k: torch.stack(v).mean() for k, v in losses.items()}
        return losses
      
    def _forward_single(self, data, return_unscaled=False, **kwargs):
        inter_embeds = []
        outputs = {}
        for key in data.keys():
            mod, node = key
            if mod == 'mocap':
                continue
            if mod not in self.backbones.keys():
                continue
            backbone = self.backbones[mod]
            model = self.adapters[mod + '_' + node]
            if self.freeze_backbone:
                with torch.no_grad():
                    try:
                        feats = backbone(data[key]['img'])
                    except:
                        feats = backbone([data[key]['img']])
            else:
                try:
                    feats = backbone(data[key]['img'])
                except:
                    feats = backbone([data[key]['img']])

            feats = feats[0]
            embeds = model(feats)
            outputs[key] = embeds
            inter_embeds.append(embeds)

        if len(inter_embeds) == 0:
            import ipdb; ipdb.set_trace() # noqa
        
        inter_embeds = torch.stack(inter_embeds, dim=1)
        return outputs
   


    #REMAINING FUNCTIONS JUST TO COMPLY WITH MMCV API
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
