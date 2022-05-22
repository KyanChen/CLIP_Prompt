import copy
import inspect
import math
import warnings

import cv2
import mmcv
import numpy as np
from ...datasets.builder import PIPELINES


@PIPELINES.register_module()
class ScaleCrop:
    """
    crop img according to boxes
    """

    def __init__(self, scale_range=[0.1, 0.4]):
        self.scale_range = scale_range

    def _get_crop_size(self, instance_size):
        h, w = instance_size
        random_scale = self.scale_range[0] + np.random.rand() * (self.scale_range[1] - self.scale_range[0])
        random_scale += 1
        return int(h * random_scale + 0.5), int(w * random_scale + 0.5)

    def __call__(self, results):
        x, y, w, h = results['instance_bbox']
        instance_size = [h, w]
        crop_size = self._get_crop_size(instance_size)
        # cv2.imshow('1', results['img'][y:y+h, x:x+w, :])
        # cv2.waitKey()
        cx = x + w/2.
        cy = y + h/2.
        for key in results.get('img_fields', ['img']):
            img = results[key]
            y0 = cy - crop_size[0]/2
            y1 = cy + crop_size[0]/2
            x0 = cx - crop_size[1]/2
            x1 = cx + crop_size[1]/2
            # print(img.shape)
            # print(x0, ' ', y0, ' ', x1, ' ', y1)
            y0, y1 = np.clip([y0, y1], 0, img.shape[0]).astype(np.int)
            x0, x1 = np.clip([x0, x1], 0, img.shape[1]).astype(np.int)
            # print(x0, ' ', y0, ' ', x1, ' ', y1)
            img = img[y0:y1, x0:x1, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape
        # print(results['img'].shape)
        return results



