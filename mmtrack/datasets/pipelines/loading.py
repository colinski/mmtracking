# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile

from mmtrack.core import results2outs
import numpy as np
import torch
import torchaudio
import cv2

@PIPELINES.register_module()
class DecodeJPEG(object):
    def __init__(self):
        pass

    def __call__(self, code):
        img = cv2.imdecode(code, 1)
        return img

@PIPELINES.register_module()
class LoadAudio(object):
    def __init__(self, n_fft=400):
        self.spectro = torchaudio.transforms.Spectrogram(n_fft=n_fft)

    def __call__(self, array):
        array = array[:, 1:5]
        array = torch.from_numpy(array)
        array = array.unsqueeze(0)
        array = array.permute(0, 2, 1)
        sgram = self.spectro(array)
        sgram = sgram.permute(0, 2, 3, 1).squeeze()
        sgram = sgram.numpy()
        results = {
            'img': sgram, 
            'img_shape': sgram.shape,
            'ori_shape': sgram.shape, 
            'img_fields': ['img'],
            'filename': 'placeholder.jpg',
            'ori_filename': 'placeholder.jpg'
        }
        return results

        # sgram = sgram.squeeze()
        # sgram = sgram.permute(1, 2, 0)
        # return sgram.numpy()

@PIPELINES.register_module()
class LoadFromNumpyArray(object):
    def __init__(self, force_float32=False, transpose=False, force_rgb=False):
        self.force_float32 = force_float32
        self.transpose = transpose
        self.force_rgb = force_rgb

    def __call__(self, array):
        if self.force_float32:
            array = array.astype(np.float32)
        if self.transpose:
            array = array.T
        if len(array.shape) == 2: #add channel dimesion
            array = array[:, :, np.newaxis]
        if self.force_rgb:
            array = np.concatenate([array, array, array], axis=-1)
        array = np.nan_to_num(array, nan=0.0)
        results = {
            'img': array, 
            'img_shape': array.shape,
            'ori_shape': array.shape, 
            'img_fields': ['img'],
            'filename': 'placeholder.jpg',
            'ori_filename': 'placeholder.jpg'
        }
        return results


@PIPELINES.register_module()
class LoadMultiImagesFromFile(LoadImageFromFile):
    """Load multi images from file.

    Please refer to `mmdet.datasets.pipelines.loading.py:LoadImageFromFile`
    for detailed docstring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function.

        For each dict in `results`, call the call function of
        `LoadImageFromFile` to load image.

        Args:
            results (list[dict]): List of dict from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains loaded image.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class SeqLoadAnnotations(LoadAnnotations):
    """Sequence load annotations.

    Please refer to `mmdet.datasets.pipelines.loading.py:LoadAnnotations`
    for detailed docstring.

    Args:
        with_track (bool): If True, load instance ids of bboxes.
    """

    def __init__(self, with_track=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_track = with_track

    def _load_track(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_instance_ids'] = results['ann_info']['instance_ids'].copy()

        return results

    def __call__(self, results):
        """Call function.

        For each dict in results, call the call function of `LoadAnnotations`
        to load annotation.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains loaded annotations, such as
            bounding boxes, labels, instance ids, masks and semantic
            segmentation annotations.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            if self.with_track:
                _results = self._load_track(_results)
            outs.append(_results)
        return outs


@PIPELINES.register_module()
class LoadDetections(object):
    """Load public detections from MOT benchmark.

    Args:
        results (dict): Result dict from :obj:`mmtrack.CocoVideoDataset`.
    """

    def __call__(self, results):
        outs_det = results2outs(bbox_results=results['detections'])
        bboxes = outs_det['bboxes']
        labels = outs_det['labels']

        results['public_bboxes'] = bboxes[:, :4]
        if bboxes.shape[1] > 4:
            results['public_scores'] = bboxes[:, -1]
        results['public_labels'] = labels
        results['bbox_fields'].append('public_bboxes')
        return results
