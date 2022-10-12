import json
import logging
import math
import os
import os.path as osp
import pickle
import random
import tempfile
import warnings
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from collections import OrderedDict, defaultdict
import imagesize
import cv2
import mmcv
import numpy as np
import torch
from mmcv import tensor2imgs
from mmcv.parallel import DataContainer
from sklearn import metrics
from torchmetrics.detection import MeanAveragePrecision

from ..core import eval_recalls, eval_map
from ..datasets.builder import DATASETS
from torch.utils.data import Dataset
from ..datasets.pipelines import Compose
from .evaluate_tools import cal_metrics
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


@DATASETS.register_module()
class RPNAttributeDataset(Dataset):

    CLASSES = None

    def __init__(self,
                 data_root,
                 pipeline,
                 dataset_split='train',
                 attribute_index_file=None,
                 dataset_names=[],
                 dataset_balance=False,
                 kd_pipeline=None,
                 test_mode=False,
                 test_content='box_given',
                 mult_proposal_score=False,
                 file_client_args=dict(backend='disk')
                 ):
        super(RPNAttributeDataset, self).__init__()
        assert dataset_split in ['train', 'val', 'test']
        self.dataset_split = dataset_split
        self.test_mode = test_mode
        self.mult_proposal_score = mult_proposal_score
        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        self.dataset_names = dataset_names

        self.kd_pipeline = kd_pipeline
        if kd_pipeline:
            self.kd_pipeline = Compose(kd_pipeline)

        self.attribute_index_file = attribute_index_file
        self.att2id = {}
        self.att_seen_unseen = {}
        if 'att_file' in attribute_index_file.keys():
            file = attribute_index_file['att_file']
            att2id = json.load(open(file, 'r'))
            att_group = attribute_index_file['att_group']
            if att_group in ['common1', 'common2', 'common', 'rare']:
                self.att2id = att2id[att_group]
            elif att_group == 'common1+common2':
                self.att2id.update(att2id['common1'])
                self.att2id.update(att2id['common2'])
                self.att_seen_unseen['seen'] = list(att2id['common1'].keys())
                self.att_seen_unseen['unseen'] = list(att2id['common2'].keys())
            elif att_group == 'common+rare':
                self.att2id.update(att2id['common'])
                self.att2id.update(att2id['rare'])
                self.att_seen_unseen['seen'] = list(att2id['common'].keys())
                self.att_seen_unseen['unseen'] = list(att2id['rare'].keys())
            else:
                raise NameError
        self.category2id = {}
        self.category_seen_unseen = {}
        if 'category_file' in attribute_index_file.keys():
            file = attribute_index_file['category_file']
            category2id = json.load(open(file, 'r'))
            att_group = attribute_index_file['category_group']
            if att_group in ['common1', 'common2', 'common', 'rare']:
                self.category2id = category2id[att_group]
            elif att_group == 'common1+common2':
                self.category2id.update(category2id['common1'])
                self.category2id.update(category2id['common2'])
                self.category_seen_unseen['seen'] = list(category2id['common1'].keys())
                self.category_seen_unseen['unseen'] = list(category2id['common2'].keys())
            elif att_group == 'common+rare':
                self.category2id.update(category2id['common'])
                self.category2id.update(category2id['rare'])
            else:
                raise NameError
        self.att2id = {k: v - min(self.att2id.values()) for k, v in self.att2id.items()}
        self.category2id = {k: v - min(self.category2id.values()) for k, v in self.category2id.items()}

        self.id2images = {}
        self.id2instances = {}

        if 'coco' in self.dataset_names:
            id2images_coco, id2instances_coco = self.read_data_coco(dataset_split)
            self.id2images.update(id2images_coco)
            self.id2instances.update(id2instances_coco)
            self.id2instances.pop('coco_200365', None)
            self.id2instances.pop('coco_183338', None)
            self.id2instances.pop('coco_550395', None)
            self.id2instances.pop('coco_77039', None)
            self.id2instances.pop('coco_340038', None)
            # self.id2instances.pop('coco_147195', None)
            # self.id2instances.pop('coco_247306', None)
            self.id2instances.pop('coco_438629', None)
            self.id2instances.pop('coco_284932', None)

        if 'vaw' in self.dataset_names:
            assert dataset_split in ['train', 'test']
            id2images_vaw, id2instances_vaw = self.read_data_vaw(dataset_split)
            self.id2images.update(id2images_vaw)
            self.id2instances.update(id2instances_vaw)
            self.id2instances.pop('vaw_713545', None)
            self.id2instances.pop('vaw_2369080', None)

        if 'ovadcate' in self.dataset_names:
            if dataset_split == 'test':
                dataset_split == 'val'
            id2images_ovad, id2instances_ovad = self.read_data_ovad('cate')
            self.id2images.update(id2images_ovad)
            self.id2instances.update(id2instances_ovad)

        if 'ovadattr' in self.dataset_names:
            if dataset_split == 'test':
                dataset_split == 'val'
            id2images_ovad, id2instances_ovad = self.read_data_ovad('attr')
            self.id2images.update(id2images_ovad)
            self.id2instances.update(id2instances_ovad)

        if 'ovadgen' in self.dataset_names:
            id2images_ovadgen, id2instances_ovadgen = self.read_data_ovadgen(dataset_split)
            self.id2images.update(id2images_ovadgen)
            self.id2instances.update(id2instances_ovadgen)

        if not self.test_mode:
            # filter images too small and containing no annotations
            self.img_ids = self._filter_imgs()
            self._set_group_flag()
        else:
            self.img_ids = list(self.id2instances.keys())[:320] + list(self.id2instances.keys())[-320:]

        img_ids_per_dataset = {}
        for x in self.img_ids:
            img_ids_per_dataset[x.split('_')[0]] = img_ids_per_dataset.get(x.split('_')[0], []) + [x]

        rank, world_size = get_dist_info()
        if rank == 0:
            for k, v in img_ids_per_dataset.items():
                print('dataset: ', k, ' len - ', len(v))
        if dataset_balance and not test_mode:
            balance_frac = round(len(img_ids_per_dataset['coco']) / len(img_ids_per_dataset['vaw']))
            self.img_ids = img_ids_per_dataset['coco'] + balance_frac * img_ids_per_dataset['vaw']
            if rank == 0:
                print('balance dataset fractor = ', balance_frac)
                data_len = {}
                for x in self.img_ids:
                    data_len[x.split('_')[0]] = data_len.get(x.split('_')[0], 0) + 1
                for k, v in data_len.items():
                    print('data len: ', k, ' - ', v)

            flag_dataset = [x.split('_')[0] for x in self.img_ids]
            dataset_types = {'coco': 0, 'vaw': 1}
            flag_dataset = [dataset_types[x] for x in flag_dataset]
            self.flag_dataset = np.array(flag_dataset, dtype=np.int)
        self.error_list = set()
        self.test_content = test_content
        assert self.test_content in ['box_given', 'box_free']

    def read_data_coco(self, pattern):
        if pattern == 'test':
            pattern = 'val'
        json_file = 'instances_train2017' if pattern == 'train' else 'instances_val2017'
        # json_file = 'lvis_v1_train' if pattern == 'train' else 'instances_val2017'
        json_data = json.load(open(self.data_root + f'/COCO/annotations/{json_file}.json', 'r'))
        id2name = {x['id']: x['name'] for x in json_data['categories']}
        id2images = {}
        id2instances = {}
        for data in json_data['images']:
            img_id = 'coco_' + str(data['id'])
            data['file_name'] = f'{data["id"]:012d}.jpg'
            id2images[img_id] = data
        for data in json_data['annotations']:
            img_id = 'coco_' + str(data['image_id'])
            data['name'] = id2name[data['category_id']]
            id2instances[img_id] = id2instances.get(img_id, []) + [data]
        return id2images, id2instances

    def read_data_vaw(self, pattern):
        json_files = ['train_part1', 'train_part2'] if pattern == 'train' else [f'{pattern}']
        json_data = [json.load(open(self.data_root + '/VAW/' + f'{x}.json', 'r')) for x in json_files]
        instances = []
        [instances.extend(x) for x in json_data]
        id2images = {}
        id2instances = {}
        for instance in instances:
            img_id = 'vaw_' + str(instance['image_id'])
            id2instances[img_id] = id2instances.get(img_id, []) + [instance]
        for img_id in id2instances.keys():
            img_info = {'file_name': f'{img_id.split("_")[-1] + ".jpg"}'}
            img_path = os.path.abspath(self.data_root) + '/VG/VG_100K/' + img_info['file_name']
            w, h = imagesize.get(img_path)
            img_info['width'] = w
            img_info['height'] = h
            id2images[img_id] = img_info
        return id2images, id2instances

    def _filter_imgs(self, min_wh_size=32, min_box_wh_size=4):
        valid_img_ids = []
        for img_id, img_info in self.id2images.items():
            if min(img_info['width'], img_info['height']) < min_wh_size:
                continue
            instances = self.id2instances.get(img_id, [])
            instances_tmp = []

            data_set = img_id.split('_')[0]
            key = 'bbox' if data_set == 'coco' else 'instance_bbox'
            for instance in instances:
                x, y, w, h = instance[key]
                if w < min_box_wh_size or h < min_box_wh_size:
                    continue
                if data_set == 'coco':
                    category = instance['name']
                    category_id = self.category2id.get(category, None)  # 未标注的该类别的应该去除
                    if category_id is not None:
                        instances_tmp.append(instance)
                elif data_set == 'vaw':
                    instances_tmp.append(instance)
                elif data_set == 'ovadgen':
                    instances_tmp.append(instance)

            self.id2instances[img_id] = instances_tmp
            if len(instances_tmp) == 0:
                continue
            valid_img_ids.append(img_id)
        return valid_img_ids

    def get_instances(self):
        proposals = json.load(open('/data/kyanchen/prompt1/tools/results/EXP20220628_0/FasterRCNN_R50_OpenImages.proposal.json', 'r'))
        img_proposal_pair = {}
        for instance in proposals:
            img_id = instance['image_id']
            img_proposal_pair[img_id] = img_proposal_pair.get(img_id, []) + [instance]

        instances = []
        for img_id in self.img_ids:
            gt_bboxes = [instance['instance_bbox'] for instance in self.img_instances_pair[img_id]]
            gt_bboxes = np.array(gt_bboxes).reshape(-1, 4)
            gt_bboxes[:, 2:] = gt_bboxes[:, :2] + gt_bboxes[:, 2:]
            for proposal in img_proposal_pair[img_id]:
                if proposal['score'] < 0.55:
                    continue
                box = np.array(proposal['bbox']).reshape(-1, 4)
                iou = bbox_overlaps(box, gt_bboxes)[0]
                box_ind = np.argmax(iou)
                if iou[box_ind] < 0.6:
                    continue
                # import pdb
                # pdb.set_trace()
                instance = self.img_instances_pair[img_id][box_ind]
                instance['instance_bbox'] = box[0].tolist()
                instances.append(instance)
        return instances

    def __len__(self):
        return len(self.img_ids)

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for idx, img_id in enumerate(self.img_ids):
            img_info = self.id2images[img_id]
            w, h = img_info['width'], img_info['height']
            if w / h > 1:
                self.flag[idx] = 1

    def get_test_data(self, idx):
        results = self.instances[idx].copy()
        results['img_prefix'] = os.path.abspath(self.data_root) + '/VG/VG_100K'
        results['img_info'] = {}
        results['img_info']['filename'] = f'{results["image_id"]}.jpg'
        x, y, w, h = results["instance_bbox"]
        results['proposals'] = np.array([x, y, x + w, y + h], dtype=np.float32).reshape(1, 4)
        results['bbox_fields'] = ['proposals']
        results = self.pipeline(results)
        return results

    def get_img_instances(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.id2images[img_id]
        instances = self.id2instances[img_id]
        data_set = img_id.split('_')[0]
        if data_set == 'coco':
            data_set_type = 0
            if self.dataset_split == 'test':
                dataset_split = 'val'
            else:
                dataset_split = self.dataset_split
            prefix_path = f'/COCO/{dataset_split}2017'
        elif data_set == 'vaw':
            prefix_path = f'/VG/VG_100K'
            data_set_type = 1
        elif data_set in ['ovadcate', 'ovadattr']:
            if self.dataset_split == 'test':
                dataset_split = 'val'
            else:
                dataset_split = self.dataset_split
            prefix_path = f'/COCO/{dataset_split}2017'
            if data_set == 'ovadcate':
                data_set_type = 0
            elif data_set == 'ovadattr':
                data_set_type = 1
            else:
                raise NameError
        elif data_set == 'ovadgen':
            data_set_type = 1
            prefix_path = f'/ovadgen'
        else:
            raise NameError

        results = {}
        results['data_set_type'] = data_set_type
        results['img_prefix'] = os.path.abspath(self.data_root) + prefix_path
        results['img_info'] = {}
        results['img_info']['filename'] = img_info['file_name']

        bbox_list = []
        attr_label_list = []
        for instance in instances:
            key = 'bbox' if data_set == 'coco' else 'instance_bbox'
            x, y, w, h = instance[key]
            bbox_list.append([x, y, x + w, y + h])

            labels = np.ones(len(self.att2id)+len(self.category2id)) * 2
            labels[len(self.att2id):] = 0
            if data_set == 'vaw' or data_set == 'ovadgen':
                positive_attributes = instance["positive_attributes"]
                negative_attributes = instance["negative_attributes"]
                for att in positive_attributes:
                    att_id = self.att2id.get(att, None)
                    if att_id is not None:
                        labels[att_id] = 1
                for att in negative_attributes:
                    att_id = self.att2id.get(att, None)
                    if att_id is not None:
                        labels[att_id] = 0
            if data_set == 'coco':
                category = instance['name']
                category_id = self.category2id.get(category, None)
                if category_id is not None:
                    labels[category_id+len(self.att2id)] = 1
            attr_label_list.append(labels)

        gt_bboxes = np.array(bbox_list, dtype=np.float32)
        gt_labels = np.stack(attr_label_list, axis=0).astype(np.int)
        results['gt_bboxes'] = gt_bboxes
        results['bbox_fields'] = ['gt_bboxes']
        results['gt_labels'] = gt_labels
        results['dataset_type'] = data_set_type
        assert len(gt_labels) == len(gt_bboxes)
        if self.kd_pipeline:
            kd_results = results.copy()
            kd_results.pop('gt_labels')
            kd_results.pop('bbox_fields')
        try:
            results = self.pipeline(results)
            if self.kd_pipeline:
                kd_results = self.kd_pipeline(kd_results, 0)
                img_crops = []
                for gt_bboxes in kd_results['gt_bboxes']:
                    kd_results_tmp = kd_results.copy()
                    kd_results_tmp['crop_box'] = gt_bboxes
                    kd_results_tmp = self.kd_pipeline(kd_results_tmp, (1, ':'))
                    img_crops.append(kd_results_tmp['img'])
                img_crops = torch.stack(img_crops, dim=0)
                results['img_crops'] = img_crops

            results['gt_bboxes'] = DataContainer(results['gt_bboxes'], stack=False)
            results['gt_labels'] = DataContainer(results['gt_labels'], stack=False)
            results['img'] = DataContainer(results['img'], padding_value=0, stack=True)
            if self.kd_pipeline:
                results['img_crops'] = DataContainer(results['img_crops'], stack=False)
        except Exception as e:
            print(e)
            self.error_list.add(idx)
            self.error_list.add(img_id)
            print(self.error_list)
            if not self.test_mode:
                results = self.__getitem__(np.random.randint(0, len(self)))
        return results

    def get_test_box_given(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.id2images[img_id]
        instances = self.id2instances[img_id]
        data_set = img_id.split('_')[0]
        if data_set == 'coco':
            data_set_type = 0
            if self.dataset_split == 'test':
                dataset_split = 'val'
            else:
                dataset_split = self.dataset_split
            prefix_path = f'/COCO/{dataset_split}2017'
        elif data_set == 'vaw':
            prefix_path = f'/VG/VG_100K'
            data_set_type = 1
        elif data_set in ['ovadcate', 'ovadattr']:
            if self.dataset_split == 'test':
                dataset_split = 'val'
            else:
                dataset_split = self.dataset_split
            prefix_path = f'/COCO/{dataset_split}2017'
            if data_set == 'ovadcate':
                data_set_type = 0
            elif data_set == 'ovadattr':
                data_set_type = 1
            else:
                raise NameError
        elif data_set == 'ovadgen':
            data_set_type = 1
            prefix_path = f'/ovadgen'
        else:
            raise NameError

        results = {}
        results['img_prefix'] = os.path.abspath(self.data_root) + prefix_path
        results['img_info'] = {}
        results['img_info']['filename'] = img_info['file_name']

        bbox_list = []
        for instance in instances:
            key = 'bbox' if data_set == 'coco' else 'instance_bbox'
            x, y, w, h = instance[key]
            bbox_list.append([x, y, x + w, y + h])

        gt_bboxes = np.array(bbox_list, dtype=np.float32).reshape(-1, 4)
        results['gt_bboxes'] = gt_bboxes
        results['bbox_fields'] = ['gt_bboxes']
        results = self.pipeline(results)

        return results

    def get_test_box_free(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.id2images[img_id]

        data_set = img_id.split('_')[0]
        if data_set == 'coco':
            data_set_type = 0
            if self.dataset_split == 'test':
                dataset_split = 'val'
            else:
                dataset_split = self.dataset_split
            prefix_path = f'/COCO/{dataset_split}2017'
        elif data_set == 'vaw':
            prefix_path = f'/VG/VG_100K'
            data_set_type = 1
        elif data_set in ['ovadcate', 'ovadattr']:
            if self.dataset_split == 'test':
                dataset_split = 'val'
            else:
                dataset_split = self.dataset_split
            prefix_path = f'/COCO/{dataset_split}2017'
            if data_set == 'ovadcate':
                data_set_type = 0
            elif data_set == 'ovadattr':
                data_set_type = 1
            else:
                raise NameError
        elif data_set == 'ovadgen':
            data_set_type = 1
            prefix_path = f'/ovadgen'
        else:
            raise NameError
        results = {}
        results['img_prefix'] = os.path.abspath(self.data_root) + prefix_path
        results['img_info'] = {}
        results['img_info']['filename'] = img_info['file_name']
        results = self.pipeline(results)
        return results

    def __getitem__(self, idx):
        if self.test_mode:
            if self.test_content == 'box_free':
                return self.get_test_box_free(idx)
            elif self.test_content == 'box_given':
                return self.get_test_box_given(idx)
            else:
                assert NotImplementedError
        if idx in self.error_list and not self.test_mode:
            idx = np.random.randint(0, len(self))
        return self.get_img_instances(idx)

        if self.test_mode:
            return self.get_test_data(idx)

        results = self.instances[idx].copy()
        '''
        "image_id": "2373241",
        "instance_id": "2373241004",
        "instance_bbox": [0.0, 182.5, 500.16666666666663, 148.5],
        "instance_polygon": [[[432.5, 214.16666666666669], [425.8333333333333, 194.16666666666666], [447.5, 190.0], [461.6666666666667, 187.5], [464.1666666666667, 182.5], [499.16666666666663, 183.33333333333331], [499.16666666666663, 330.0], [3.3333333333333335, 330.0], [0.0, 253.33333333333334], [43.333333333333336, 245.0], [60.833333333333336, 273.3333333333333], [80.0, 293.3333333333333], [107.5, 307.5], [133.33333333333334, 309.16666666666663], [169.16666666666666, 295.8333333333333], [190.83333333333331, 274.1666666666667], [203.33333333333334, 252.5], [225.0, 260.0], [236.66666666666666, 254.16666666666666], [260.0, 254.16666666666666], [288.3333333333333, 253.33333333333334], [287.5, 257.5], [271.6666666666667, 265.0], [324.1666666666667, 281.6666666666667], [369.16666666666663, 274.1666666666667], [337.5, 261.6666666666667], [338.3333333333333, 257.5], [355.0, 261.6666666666667], [357.5, 257.5], [339.1666666666667, 255.0], [337.5, 240.83333333333334], [348.3333333333333, 238.33333333333334], [359.1666666666667, 248.33333333333331], [377.5, 251.66666666666666], [397.5, 248.33333333333331], [408.3333333333333, 236.66666666666666], [418.3333333333333, 220.83333333333331], [427.5, 217.5], [434.16666666666663, 215.0]]],
        "object_name": "floor",
        "positive_attributes": ["tiled", "gray", "light colored"],
        "negative_attributes": ["multicolored", "maroon", "weathered", "speckled", "carpeted"]
        '''
        results['img_prefix'] = os.path.abspath(self.data_root) + '/VG/VG_100K'
        results['img_info'] = {}
        results['img_info']['filename'] = f'{results["image_id"]}.jpg'
        x, y, w, h = results["instance_bbox"]

        # filename = os.path.abspath(self.data_root) + '/VG/VG_100K' + f'/{results["image_id"]}.jpg'
        # img = cv2.imread(filename, cv2.IMREAD_COLOR)
        # # import pdb
        # # pdb.set_trace()
        # # x1, y1, x2, y2 = int(x-w/2.), int(y-h/2), int(x+w/2), int(y+h/2)
        # x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        # img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)
        # os.makedirs('results/tmp', exist_ok=True)
        # cv2.imwrite('results/tmp' + f'/{idx}.jpg', img)

        results['proposals'] = np.array([x, y, x+w, y+h], dtype=np.float32).reshape(1, 4)
        results['bbox_fields'] = ['proposals']
        positive_attributes = results["positive_attributes"]
        negative_attributes = results["negative_attributes"]
        labels = np.ones(len(self.classname_maps.keys())) * 2
        for att in positive_attributes:
            att_id = self.att2id.get(att, None)
            if att_id is not None:
                labels[att_id] = 1
        for att in negative_attributes:
            att_id = self.att2id.get(att, None)
            if att_id is not None:
                labels[att_id] = 0
        results['gt_labels'] = labels.astype(np.int)

        try:
            results = self.pipeline(results)
        except Exception as e:
            print(e)
            self.error_list.add(idx)
            self.error_list.add(results['img_info']['filename'])
            print(self.error_list)
            if len(self.error_list) > 20:
                return
            if not self.test_mode:
                results = self.__getitem__(np.random.randint(0, len(self)))

        # img = results['img']
        # img_metas = results['img_metas'].data
        #
        # img = img.cpu().numpy().transpose(1, 2, 0)
        # mean, std = img_metas['img_norm_cfg']['mean'], img_metas['img_norm_cfg']['std']
        # img = (255*mmcv.imdenormalize(img, mean, std, to_bgr=True)).astype(np.uint8)
        # # import pdb
        # # pdb.set_trace()
        # box = results['proposals'].numpy()[0]
        # # print(box)
        # x1, y1, x2, y2 = box.astype(np.int)
        # img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)
        # os.makedirs('results/tmp', exist_ok=True)
        # cv2.imwrite('results/tmp' + f'/x{idx}.jpg', img)
        return results

    def get_labels(self):
        total_gt_list = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            img_info = self.id2images[img_id]
            instances = self.id2instances[img_id]
            data_set = img_id.split('_')[0]

            if data_set == 'coco':
                data_set_type = 0
            elif data_set == 'vaw':
                data_set_type = 1
            elif data_set in ['ovadcate', 'ovadattr']:
                if data_set == 'ovadcate':
                    data_set_type = 0
                elif data_set == 'ovadattr':
                    data_set_type = 1
                else:
                    raise NameError
            elif data_set == 'ovadgen':
                data_set_type = 1
            else:
                raise NameError

            bbox_list = []
            attr_label_list = []
            for instance in instances:
                key = 'bbox' if data_set == 'coco' else 'instance_bbox'
                x, y, w, h = instance[key]
                bbox_list.append([data_set_type, x, y, x + w, y + h])

                labels = np.ones(len(self.att2id) + len(self.category2id)) * 2
                labels[len(self.att2id):] = 0
                if data_set == 'vaw' or data_set == 'ovadgen':
                    positive_attributes = instance["positive_attributes"]
                    negative_attributes = instance["negative_attributes"]
                    for att in positive_attributes:
                        att_id = self.att2id.get(att, None)
                        if att_id is not None:
                            labels[att_id] = 1
                    for att in negative_attributes:
                        att_id = self.att2id.get(att, None)
                        if att_id is not None:
                            labels[att_id] = 0
                if data_set == 'coco':
                    category = instance['name']
                    category_id = self.category2id.get(category, None)
                    if category_id is not None:
                        labels[category_id + len(self.att2id)] = 1
                attr_label_list.append(labels)

            gt_bboxes = np.array(bbox_list, dtype=np.float32).reshape(-1, 5)
            gt_labels = np.stack(attr_label_list, axis=0).reshape(-1, len(self.att2id) + len(self.category2id))
            total_gt_list.append(np.concatenate([gt_bboxes, gt_labels], axis=1))

        return total_gt_list

    def get_img_instance_labels(self):
        attr_label_list = []
        for img_id in self.img_ids:
            instances = self.id2instances[img_id]
            for instance in instances:
                x, y, w, h = instance["instance_bbox"]
                positive_attributes = instance.get("positive_attributes", [])
                negative_attributes = instance.get("negative_attributes", [])

                labels = np.ones(len(self.att2id.keys())) * 2

                for att in positive_attributes:
                    att_id = self.att2id.get(att, None)
                    if att_id is not None:
                        labels[att_id] = 1
                for att in negative_attributes:
                    att_id = self.att2id.get(att, None)
                    if att_id is not None:
                        labels[att_id] = 0

                attr_label_list.append(labels)
        gt_labels = np.stack(attr_label_list, axis=0)
        return gt_labels

    def get_rpn_img_instance_labels(self):
        gt_labels = []
        for img_id in self.img_ids:
            instances = self.id2instances[img_id]
            gt_labels_tmp = []
            for instance in instances:
                x, y, w, h = instance["instance_bbox"]
                positive_attributes = instance.get("positive_attributes", [])
                negative_attributes = instance.get("negative_attributes", [])

                labels = [2] * len(self.att2id.keys())

                for att in positive_attributes:
                    labels[self.att2id[att]] = 1
                for att in negative_attributes:
                    labels[self.att2id[att]] = 0

                gt_labels_tmp.append([x, y, x+w, y+h]+ labels)
            gt_labels_tmp = torch.tensor(gt_labels_tmp)
            gt_labels.append(gt_labels_tmp)
        return gt_labels

    def evaluate_box_free(self, results):
        # results List[Tensor] N, Nx(4+1+620)
        # gt_labels List[Tensor] N, Nx(1+4+620)
        result_metrics = OrderedDict()

        gt_labels = self.get_labels()

        print('Computing cate RPN recall:')
        gt_bboxes = [gt[:, 1:5] for gt in gt_labels if gt[0, 0] == 0]
        proposals = [x[:, :5].numpy() for idx, x in enumerate(results) if gt_labels[idx][0, 0] == 0]
        proposal_nums = [100, 300, 1000]
        iou_thrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        recalls = eval_recalls(
            gt_bboxes, proposals, proposal_nums, iou_thrs)
        print(recalls[:, 0])
        ar = recalls.mean(axis=1)
        log_msg = []
        for i, num in enumerate(proposal_nums):
            log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
        log_msg = ''.join(log_msg)
        print(log_msg)

        print('Computing att RPN recall:')
        gt_bboxes = [gt[:, 1:5] for gt in gt_labels if gt[0, 0] == 1]
        proposals = [x[:, :5].numpy() for idx, x in enumerate(results) if gt_labels[idx][0, 0] == 1]
        proposal_nums = [100, 300, 1000]
        iou_thrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        recalls = eval_recalls(
            gt_bboxes, proposals, proposal_nums, iou_thrs)
        print(recalls[:, 0])
        ar = recalls.mean(axis=1)
        log_msg = []
        for i, num in enumerate(proposal_nums):
            log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
        log_msg = ''.join(log_msg)
        print(log_msg)

        print('Computing all RPN recall:')
        gt_bboxes = [gt[:, 1:5] for gt in gt_labels]
        proposals = [x[:, :5].numpy() for idx, x in enumerate(results)]
        proposal_nums = [100, 300, 1000]
        iou_thrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        recalls = eval_recalls(
            gt_bboxes, proposals, proposal_nums, iou_thrs)
        print(recalls[:, 0])
        ar = recalls.mean(axis=1)
        log_msg = []
        for i, num in enumerate(proposal_nums):
            log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
        log_msg = ''.join(log_msg)
        print(log_msg)

        print('Computing att mAP:')
        gt_bboxes = [gt for gt in gt_labels if gt[0, 0] == 1]
        predictions = [x.numpy() for idx, x in enumerate(results) if gt_labels[idx][0, 0] == 1]

        prediction_attributes = []
        gt_attributes = []
        # predictions List[Tensor] N, Nx(4+1+620)
        # gts List[Tensor] N, Nx(1+4+620)
        for prediction, gt in zip(predictions, gt_bboxes):
            IoUs = bbox_overlaps(prediction[:, :4], gt[:, 1:5])
            inds = np.argmax(IoUs, axis=0)
            pred_att = prediction[inds][:, 5:5+len(self.att2id)]
            gt_att = gt[:, 5:5 + len(self.att2id)]

            # thres = 0.5
            # IoU_thres = IoUs[inds, range(IoUs.shape[1])]
            # pred_att = pred_att[IoU_thres > thres]
            # gt_att = gt_att[IoU_thres > thres]
            #
            prediction_attributes.append(pred_att)
            gt_attributes.append(gt_att)
        prediction_attributes = np.concatenate(prediction_attributes, axis=0)
        gt_attributes = np.concatenate(gt_attributes, axis=0)

        dataset_name = self.attribute_index_file['att_file'].split('/')[-2]
        top_k = 15 if dataset_name == 'VAW' else 8

        pred_att_logits = torch.from_numpy(prediction_attributes).float().sigmoid().numpy()  # Nx620
        gt_att = gt_attributes

        prs = []
        for i_att in range(pred_att_logits.shape[1]):
            y = gt_att[:, i_att]
            pred = pred_att_logits[:, i_att]
            gt_y = y[~(y == 2)]
            pred = pred[~(y == 2)]
            if len(pred) != 0:
                import pdb
                pdb.set_trace()
                pr = metrics.average_precision_score(pred, gt_y.astype(np.int))
                if torch.isnan(pr):
                    continue
                prs.append(pr)
        print('map: ', np.mean(prs))
        import pdb
        pdb.set_trace()

        output = cal_metrics(
            self.att2id,
            dataset_name,
            prefix_path=f'../attributes/{dataset_name}',
            pred=pred_att_logits,
            gt_label=gt_att,
            top_k=top_k,
            save_result=True,
            att_seen_unseen=self.att_seen_unseen
        )
        result_metrics['att_att_all'] = output['PC_ap/all']

        print('Computing cate mAP:')
        gt_bboxes = [gt for gt in gt_labels if gt[0, 0] == 0]
        predictions = [x for idx, x in enumerate(results) if gt_labels[idx][0, 0] == 0]
        # predictions List[Tensor] N, Nx(4+1+620)
        # gts List[Tensor] N, Nx(1+4+620)

        det_results = []
        annotations = []
        for pred, gt in zip(predictions, gt_bboxes):
            if pred.shape[0] == 0:
                pred_boxes = [np.zeros((0, 5), dtype=np.float32) for i in range(len(self.category2id))]
            else:
                # pred_cate_logits = pred_cate_logits.detach().sigmoid().cpu()
                pred_cate_logits = pred[:, 5+len(self.att2id):].float().softmax(dim=-1).cpu()
                if self.mult_proposal_score:
                    proposal_scores = pred[:, 4]
                    pred_cate_logits = (pred_cate_logits * proposal_scores) ** 0.5

                max_v, max_ind = torch.max(pred_cate_logits, dim=-1)
                max_v = max_v.view(-1, 1)
                pred_boxes = pred[:, :4]
                pred_boxes = [torch.cat([pred_boxes[max_ind == i], max_v[max_ind == i]], dim=-1).numpy()
                              for i in range(len(self.category2id))]
            det_results.append(pred_boxes)
            annotation = {
                'bboxes': gt[:, 1:5],
                'labels': np.argmax(gt[:, 5 + len(self.att2id):], axis=-1)}
            annotations.append(annotation)
        mean_ap, eval_results = eval_map(
            det_results,
            annotations,
            scale_ranges=[(0, 32), (32, 96), (32, 1e3)],
            iou_thr=0.5,
            ioa_thr=None,
            dataset=None,
            logger=None,
            tpfp_fn=None,
            nproc=8,
            use_legacy_coordinate=False,
            use_group_of=False)

        return result_metrics

    def evaluate_box_given(self, results):
        result_metrics = OrderedDict()

        if isinstance(results[0], type(np.array(0))):
            results = np.concatenate(results, axis=0)
        else:
            results = np.array(results)

        preds = torch.from_numpy(results)
        gt_labels = self.get_labels()
        gt_labels = np.concatenate(gt_labels, axis=0)
        data_set_type = gt_labels[:, 0].astype(np.int)
        gt_labels = torch.from_numpy(gt_labels[:, 5:].astype(np.int))
        assert preds.shape[-1] == gt_labels.shape[-1]
        assert preds.shape[0] == gt_labels.shape[0]

        data_set_type = torch.from_numpy(data_set_type)

        cate_mask = data_set_type == 0
        att_mask = data_set_type == 1
        pred_att_logits = preds[att_mask][:, :len(self.att2id)]
        pred_cate_logits = preds[cate_mask][:, len(self.att2id):]
        gt_att = gt_labels[att_mask][:, :len(self.att2id)]
        gt_cate = gt_labels[cate_mask][:, len(self.att2id):]
        # import pdb
        # pdb.set_trace()
        if len(pred_cate_logits):
            dataset_name = self.attribute_index_file['category_file'].split('/')[-2]
            top_k = 1 if dataset_name == 'COCO' else -1
            print('dataset_name: ', dataset_name, 'top k: ', top_k)

            # pred_cate_logits = pred_cate_logits.detach().sigmoid().cpu()
            pred_cate_logits = pred_cate_logits.float().softmax(dim=-1).cpu()
            #         if self.mult_proposal_score:
            #             proposal_scores = [p.get('objectness_logits') for p in proposals]
            #             scores = [(s * ps[:, None]) ** 0.5 \
            #                 for s, ps in zip(scores, proposal_scores)]
            pred_cate_logits = pred_cate_logits * (pred_cate_logits == pred_cate_logits.max(dim=-1)[0][:, None])
            gt_cate = gt_cate.detach().cpu()

            # values, indices = torch.max(pred_cate_logits, dim=-1)
            # row_indices = torch.arange(len(values))[values > 0.5]
            # col_indices = indices[values > 0.5]
            # pred_cate_logits[row_indices, col_indices] = 1
            # pred_cate_logits[pred_cate_logits < 1] = 0

            pred_cate_logits = pred_cate_logits.numpy()
            gt_cate = gt_cate.numpy()

            output = cal_metrics(
                self.category2id,
                dataset_name,
                prefix_path=f'../attributes/{dataset_name}',
                pred=pred_cate_logits,
                gt_label=gt_cate,
                top_k=top_k,
                save_result=True,
                att_seen_unseen=self.category_seen_unseen
            )
            # import pdb
            # pdb.set_trace()
            # print(output)
            result_metrics['cate_ap_all'] = output['PC_ap/all']

        assert pred_att_logits.shape[-1] == gt_att.shape[-1]

        if not len(self.att2id):
            return result_metrics

        dataset_name = self.attribute_index_file['att_file'].split('/')[-2]
        top_k = 15 if dataset_name == 'VAW' else 8

        pred_att_logits = pred_att_logits.data.cpu().float().sigmoid().numpy()  # Nx620
        gt_att = gt_att.data.cpu().float().numpy()  # Nx620

        prs = []
        for i_att in range(pred_att_logits.shape[1]):
            y = gt_att[:, i_att]
            pred = pred_att_logits[:, i_att]
            gt_y = y[~(y == 2)]
            pred = pred[~(y == 2)]
            pr = metrics.average_precision_score(gt_y, pred)
            prs.append(pr)
        print('map: ', np.mean(prs))

        output = cal_metrics(
            self.att2id,
            dataset_name,
            prefix_path=f'../attributes/{dataset_name}',
            pred=pred_att_logits,
            gt_label=gt_att,
            top_k=top_k,
            save_result=True,
            att_seen_unseen=self.att_seen_unseen
        )
        result_metrics['att_ap_all'] = output['PC_ap/all']
        return result_metrics

    def evaluate(self,
                 results,
                 logger=None,
                 metric='mAP',
                 per_class_out_file=None,
                 is_logit=True
                 ):
        if self.test_content == 'box_free':
            return self.evaluate_box_free(results)
        elif self.test_content == 'box_given':
            return self.evaluate_box_given(results)
        else:
            raise NotImplementedError

        if isinstance(results[0], type(np.array(0))):
            results = np.concatenate(results, axis=0)
        else:
            results = np.array(results)

        preds = torch.from_numpy(results)
        gts = self.get_img_instance_labels()
        gts = torch.from_numpy(gts)
        assert preds.shape[-1] == gts.shape[-1]

        output = cal_metrics(self.data_root + '/VAW',
                             preds, gts,
                             fpath_attribute_index=self.attribute_index_file,
                             return_all=True,
                             return_evaluator=per_class_out_file,
                             is_logit=is_logit)

        if per_class_out_file:
            scores_overall, scores_per_class, scores_overall_topk, scores_per_class_topk, evaluator = output
        else:
            scores_overall, scores_per_class, scores_overall_topk, scores_per_class_topk = output

        # CATEGORIES = ['all', 'head', 'medium', 'tail'] + \
        # list(evaluator.attribute_parent_type.keys())
        results = OrderedDict()
        CATEGORIES = ['all']

        for category in CATEGORIES:
            print(f"----------{category.upper()}----------")
            print(f"mAP: {scores_per_class[category]['ap']:.4f}")
            results['all_mAP'] = scores_per_class['all']['ap']

            print("Per-class (threshold 0.5):")
            for metric in ['recall', 'precision', 'f1', 'bacc']:
                if metric in scores_per_class[category]:
                    print(f"- {metric}: {scores_per_class[category][metric]:.4f}")

            print("Per-class (top 15):")
            for metric in ['recall', 'precision', 'f1']:
                if metric in scores_per_class_topk[category]:
                    print(f"- {metric}: {scores_per_class_topk[category][metric]:.4f}")

            print("Overall (threshold 0.5):")
            for metric in ['recall', 'precision', 'f1', 'bacc']:
                if metric in scores_overall[category]:
                    print(f"- {metric}: {scores_overall[category][metric]:.4f}")
            print("Overall (top 15):")
            for metric in ['recall', 'precision', 'f1']:
                if metric in scores_overall_topk[category]:
                    print(f"- {metric}: {scores_overall_topk[category][metric]:.4f}")

            if per_class_out_file:
                mmcv.mkdir_or_exist(osp.basename(per_class_out_file))
                with open(per_class_out_file, 'w') as f:
                    f.write('| {:<18}| AP\t\t| Recall@K\t| B.Accuracy\t| N_Pos\t| N_Neg\t|\n'.format('Name'))
                    f.write('-----------------------------------------------------------------------------------------------------\n')
                    for i_class in range(evaluator.n_class):
                        att = evaluator.idx2attr[i_class]
                        f.write('| {:<18}| {:.4f}\t| {:.4f}\t| {:.4f}\t\t| {:<6}| {:<6}|\n'.format(
                            att,
                            evaluator.get_score_class(i_class).ap,
                            evaluator.get_score_class(i_class, threshold_type='topk').get_recall(),
                            evaluator.get_score_class(i_class).get_bacc(),
                            evaluator.get_score_class(i_class).n_pos,
                            evaluator.get_score_class(i_class).n_neg))

        return results

