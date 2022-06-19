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

        self.attributes_dataset = pickle.load(open(attributes_file, 'rb'))
        self.coco = COCO(annotations_file)
        self.img_prefix = img_prefix
        self.test_mode = test_mode

        self.pipeline = Compose(pipeline)

        self.patch_ids = []
        split = 'val2014' if self.test_mode else 'train2014'
        # get all attribute vectors for this split
        for patch_id in self.attributes_dataset['ann_vecs'].keys():
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
        if len(ann_info['bboxes']) != 0:
            print('ann_info bboxes is not 1')

        results = dict(img_info=img_info, ann_info=ann_info, attrs=attrs)
        results['img_prefix'] = self.img_prefix
        results = self.pipeline(results)

        return results

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        pass