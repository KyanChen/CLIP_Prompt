import contextlib
import io
import itertools
import logging
import os.path
import os.path as osp
import pickle
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from mmdet.utils.api_wrappers import COCO, COCOeval
from ..datasets.builder import DATASETS
from ..datasets.custom import CustomDataset
from ..datasets.pipelines import Compose
import re
from string import punctuation


@DATASETS.register_module()
class CocoCLIPAnnDataset(CustomDataset):
    def __init__(self,
                 attributes_file,
                 annotations_file,
                 pipeline,
                 attribute_id_map=None,
                 img_prefix='',
                 test_mode=False,
                 file_client_args=dict(backend='disk')
                 ):
        self.file_client = mmcv.FileClient(**file_client_args)

        self.attributes_dataset = pickle.load(attributes_file)
        self.coco = COCO(annotations_file)
        self.img_prefix = img_prefix
        self.test_mode = test_mode

        self.pipeline = Compose(pipeline)

        self.patch_ids = []
        split = 'val2014' if self.test_mode else 'train2014'
        # get all attribute vectors for this split
        for patch_id, _ in self.attributes_dataset['ann_vecs'].items():
            if self.attributes_dataset['split'][patch_id] == split:
                self.patch_ids.append(patch_id)

        # list of attribute names
        self.attributes = sorted(
            self.attributes_dataset['attributes'], key=lambda x: x['id'])
        self.attribute_id_map = mmcv.load(attribute_id_map)

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, index):
        patch_id = self.patch_ids[index]

        attrs = self.attributes_dataset['ann_vecs'][patch_id]
        attrs = (attrs >= 0.5).astype(np.float)

        ann_id = self.attributes_dataset['patch_id_to_ann_id'][patch_id]
        # coco.loadImgs returns a list
        ann_info = self.coco.load_anns(ann_id)[0]
        img_info = self.coco.load_imgs(ann_info['image_id'])[0]

        x, y, width, height = ann_info["bbox"]
        results = dict(img_info=img_info)
        results['img_prefix'] = self.img_prefix

        img = Image.open(osp.join(self.dataset_root, self.split,
                                  image["file_name"])).convert('RGB')

        # Crop out the object with context padding.
        img = get_image_crop(img, x, y, width, height, self.crop_size)


        return img, attrs


def get_image_crop(img, x, y, width, height, crop_size=224, padding=16):
    """
    Get the image crop for the object specified in the COCO annotations.
    We crop in such a way that in the final resized image, there is `context padding` amount of image data around the object.
    This is the same as is used in RCNN to allow for additional image context.
    :param img: The image ndarray
    :param x: The x coordinate for the start of the bounding box
    :param y: The y coordinate for the start of the bounding box
    :param width: The width of the bounding box
    :param height: The height of the bounding box
    :param crop_size: The final size of the cropped image. Needed to calculate the amount of context padding.
    :param padding: The amount of context padding needed in the image.
    :return:
    """
    # Scale used to compute the new bbox for the image such that there is surrounding context.
    # The way it works is that we find what is the scaling factor between the crop and the crop without the padding
    # (which would be the original tight bounding box).
    # `crop_size` is the size of the crop with context padding.
    # The denominator is the size of the crop if we applied the same transform with the original tight bounding box.
    scale = crop_size / (crop_size - padding * 2)

    # Calculate semi-width and semi-height
    semi_width = width / 2
    semi_height = height / 2

    # Calculate the center of the crop
    centerx = x + semi_width
    centery = y + semi_height

    img_width, img_height = img.size

    # We get the crop using the semi- height and width from the center of the crop.
    # The semi- height and width are scaled accordingly.
    # We also ensure the numbers are valid
    upper = max(0, centery - (semi_height * scale))
    lower = min(img_height, centery + (semi_height * scale))
    left = max(0, centerx - (semi_width * scale))
    right = min(img_width, centerx + (semi_width * scale))

    crop_img = img.crop((left, upper, right, lower))

    if 0 in crop_img.size:
        print(img.size)
        print("lowx {0}\nlowy {1}\nhighx {2}\nhighy {3}".format(
            left, upper, right, lower))

    return crop_img