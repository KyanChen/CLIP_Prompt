import json
import logging
import os
import os.path as osp
import pickle
import random
import tempfile
import warnings
from collections import OrderedDict, defaultdict

import cv2
import imagesize
import mmcv
import numpy as np
import torch

from ..datasets.builder import DATASETS
from torch.utils.data import Dataset
from ..datasets.pipelines import Compose
from .evaluate_tools import cal_metrics


@DATASETS.register_module()
class VAWCropDataset(Dataset):

    CLASSES = None

    def __init__(self,
                 data_root,
                 pipeline,
                 dataset_split='train',
                 attribute_index_file=None,
                 test_mode=False,
                 open_category=True,
                 test_all_instances=False,
                 file_client_args=dict(backend='disk')
                 ):

        assert dataset_split in ['train', 'val', 'test']
        self.dataset_split = dataset_split
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        self.test_all_instances = test_all_instances
        if open_category:
            print('open_category: ', open_category)
            self.instances, self.img_instances_pair = self.read_data(["train_part1.json", "train_part2.json", 'val.json', 'test.json'])
            self.instances = self.split_instance_by_category(pattern=pattern)
        else:
            if dataset_split == 'train':
                if self.test_all_instances:
                    id2images_coco, id2instances_coco = self.read_data_coco(dataset_split)
                    id2images_vaw, id2instances_vaw = self.read_data_vaw(dataset_split)
                    self.id2images = {}
                    self.id2images.update(id2images_coco)
                    self.id2images.update(id2images_vaw)

                    self.id2instances = {}
                    self.id2instances.update(id2instances_coco)
                    self.id2instances.update(id2instances_vaw)
                    self.instances = []
                    for k, v in self.id2instances.items():
                        for item in v:
                            item['img_id'] = k
                            self.instances.append(item)
                    # self.instances.pop(74197)
                    # self.instances.pop(171246)
                    # self.instances.pop(171245)
                else:
                    self.instances, self.img_instances_pair = self.read_data(["train_part1.json", "train_part2.json"])
                    self.instances.pop(74197)
                    self.instances.pop(171246)
                    self.instances.pop(171245)
            elif dataset_split == 'val':
                self.instances, self.img_instances_pair = self.read_data(['val.json'])
            elif dataset_split == 'test':
                self.instances, self.id2instances = self.read_data(['test.json'])

        # self.instances = self.instances[:20]

        print('num instances: ', len(self.instances))
        print('data len: ', len(self.instances))
        self.error_list = set()
        self.img_ids = list(self.id2instances.keys())

        self.attribute_index_file = attribute_index_file
        if isinstance(attribute_index_file, dict):
            file = attribute_index_file['file']
            att2id = json.load(open(file, 'r'))
            att_group = attribute_index_file['att_group']
            if 'common2common' in file:
                if att_group in ['common1', 'common2']:
                    self.att2id = att2id[att_group]
                elif att_group == 'all':
                    self.att2id = {}
                    self.att2id.update(att2id['common1'])
                    self.att2id.update(att2id['common2'])
            elif 'common2rare' in file:
                if att_group in ['common', 'rare']:
                    self.att2id = att2id[att_group]
                elif att_group == 'all':
                    self.att2id = {}
                    self.att2id.update(att2id['common'])
                    self.att2id.update(att2id['rare'])
        else:
            self.att2id = json.load(open(attribute_index_file, 'r'))
        self.att2id = {k: v-min(self.att2id.values()) for k, v in self.att2id.items()}

        self.flag = np.zeros(len(self), dtype=int)
        print('num_att: ', len(self.att2id))

    def read_data_coco(self, pattern):
        json_file = 'instances_train2017' if pattern == 'train' else 'instances_val2017'
        # json_file = 'lvis_v1_train' if pattern == 'train' else 'instances_val2017'
        json_data = json.load(open(self.data_root + f'/COCO/annotations/{json_file}.json', 'r'))
        id2images = {}
        id2instances = {}
        for data in json_data['images']:
            img_id = 'coco_' + str(data['id'])
            data['file_name'] = f'{data["id"]:012d}.jpg'
            id2images[img_id] = data
        for data in json_data['annotations']:
            img_id = 'coco_' + str(data['image_id'])
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

    def read_data(self, json_file_list):
        json_data = [json.load(open(self.data_root + '/VAW/' + x)) for x in json_file_list]
        instances = []
        [instances.extend(x) for x in json_data]
        img_instances_pair = {}
        for instance in instances:
            img_id = instance['image_id']
            img_instances_pair[img_id] = img_instances_pair.get(img_id, []) + [instance]
        return instances, img_instances_pair

    def split_instance_by_category(self, pattern='train'):
        categories = json.load(open(self.data_root + '/VAW/' + 'category_instances_split.json'))[f'{pattern}_category']
        categories = [x[0] for x in categories]
        instances = []
        for instance in self.instances:
            if instance['object_name'] in categories:
                instances.append(instance)
        return instances

    def __len__(self):
        return len(self.instances)

    def get_test_instance(self, idx):
        instance = self.instances[idx]
        img_id = instance['img_id']
        img_info = self.id2images[img_id]

        data_set = img_id.split('_')[0]
        if data_set == 'coco':
            prefix_path = f'/COCO/{self.dataset_split}2017'
        elif data_set == 'vaw':
            prefix_path = f'/VG/VG_100K'
        else:
            raise NameError
        results = {}
        results['img_prefix'] = os.path.abspath(self.data_root) + prefix_path
        results['img_info'] = {}
        results['img_info']['filename'] = img_info['file_name']
        key = 'bbox' if data_set == 'coco' else 'instance_bbox'
        x, y, w, h = instance[key]
        results['crop_box'] = np.array([x, y, x + w, y + h])
        try:
            results = self.pipeline(results)
        except Exception as e:
            print(e)
            print(idx)
        return results

    def __getitem__(self, idx):
        if self.test_all_instances:
            return self.get_test_instance(idx)
        if idx in self.error_list and not self.test_mode:
            idx = np.random.randint(0, len(self))
        instance = self.instances[idx]
        results = {}
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
        results['img_info']['filename'] = f'{instance["image_id"]}.jpg'

        x, y, w, h = instance["instance_bbox"]
        results['crop_box'] = np.array([x, y, x + w, y + h])
        if self.test_mode:
            try:
                results = self.pipeline(results)
            except Exception as e:
                print(e)
                print(idx)
        else:
            try:
                positive_attributes = instance["positive_attributes"]
                negative_attributes = instance["negative_attributes"]
                labels = np.ones(len(self.att2id.keys())) * 2
                for att in positive_attributes:
                    att_id = self.att2id.get(att, None)
                    if att_id is not None:
                        labels[att_id] = 1
                for att in negative_attributes:
                    att_id = self.att2id.get(att, None)
                    if att_id is not None:
                        labels[att_id] = 0
                results['gt_labels'] = labels.astype(np.int)
                results = self.pipeline(results)
            except Exception as e:
                print(e)
                self.error_list.add(idx)
                self.error_list.add(results['img_info']['filename'])
                print(self.error_list)
                if not self.test_mode:
                    results = self.__getitem__(np.random.randint(0, len(self)))

        # img = results['img']
        # img_metas = results['img_metas'].data
        #
        # img = img.cpu().numpy().transpose(1, 2, 0)
        # mean, std = img_metas['img_norm_cfg']['mean'], img_metas['img_norm_cfg']['std']
        # img = (255*mmcv.imdenormalize(img, mean, std, to_bgr=True)).astype(np.uint8)
        #
        # os.makedirs('results/tmp', exist_ok=True)
        # cv2.imwrite('results/tmp' + f'/x{idx}.jpg', img)
        import pdb
        pdb.set_trace()
        return results

    def get_labels(self):
        np_gt_labels = []
        for results in self.instances:
            positive_attributes = results['positive_attributes']
            negative_attributes = results['negative_attributes']
            labels = np.ones(len(self.att2id.keys())) * 2
            for att in positive_attributes:
                att_id = self.att2id.get(att, None)
                if att_id is not None:
                    labels[att_id] = 1
            for att in negative_attributes:
                att_id = self.att2id.get(att, None)
                if att_id is not None:
                    labels[att_id] = 0
            np_gt_labels.append(labels.astype(np.int))
        return np.stack(np_gt_labels, axis=0)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 per_class_out_file=None,
                 is_logit=True
                 ):
        results = np.array(results)
        if self.test_all_instances:
            np.save('x.npy', results)
            return None
        preds = torch.from_numpy(results)
        gts = self.get_labels()
        gts = torch.from_numpy(gts)
        assert preds.shape[-1] == gts.shape[-1]
        # import pdb
        # pdb.set_trace()

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


