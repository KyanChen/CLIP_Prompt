import json
import logging
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
from torchmetrics.detection import MeanAveragePrecision

from ..core import eval_recalls
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
                 dataset_balance=False,
                 kd_pipeline=None,
                 test_rpn=False,
                 test_mode=False,
                 file_client_args=dict(backend='disk')
                 ):
        super(RPNAttributeDataset, self).__init__()
        assert dataset_split in ['train', 'val', 'test']
        self.pattern = dataset_split
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)

        if kd_pipeline:
            self.kd_pipeline = Compose(kd_pipeline)
        else:
            self.kd_pipeline = kd_pipeline

        self.data_root = data_root
        if test_mode:
            id2images_vaw, id2instances_vaw = self.read_data_vaw(dataset_split)
            self.id2images = id2images_vaw
            self.id2instances = id2instances_vaw
            self.img_ids = list(self.id2images.keys())
        else:
            id2images_coco, id2instances_coco = self.read_data_coco(dataset_split)
            id2images_vaw, id2instances_vaw = self.read_data_vaw(dataset_split)
            self.id2images = {}
            self.id2images.update(id2images_coco)
            self.id2images.update(id2images_vaw)

            self.id2instances = {}
            self.id2instances.update(id2instances_coco)
            self.id2instances.update(id2instances_vaw)

            # filter images too small and containing no annotations
            self.img_ids = self._filter_imgs()
            self._set_group_flag()

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
        self.att2id = {k: v - min(self.att2id.values()) for k, v in self.att2id.items()}

        img_ids_per_dataset = {}
        for x in self.img_ids:
            img_ids_per_dataset[x.split('_')[0]] = img_ids_per_dataset.get(x.split('_')[0], []) + [x]

        print()
        for k, v in img_ids_per_dataset.items():
            print(k, ': ', len(v))
        if dataset_balance and not test_mode:
            self.img_ids = img_ids_per_dataset['coco'] + 2*img_ids_per_dataset['vaw']
            print('dataset_balance: ', True)
            print()
            for k, v in img_ids_per_dataset.items():
                if 'coco' == k:
                    print(k, ': ', len(v))
                else:
                    print(k, ': ', 2*len(v))
            flag_dataset = [x.split('_')[0] for x in self.img_ids]
            dataset_types = {'coco': 0, 'vaw': 1}
            flag_dataset = [dataset_types[x] for x in flag_dataset]
            self.flag_dataset = np.array(flag_dataset, dtype=np.int)

        print('data len: ', len(self))
        self.test_rpn = test_rpn
        self.error_list = set()

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

    def _filter_imgs(self, min_wh_size=32, min_box_wh_size=4):
        valid_img_ids = []
        for img_id, img_info in self.id2images.items():
            if min(img_info['width'], img_info['height']) < min_wh_size:
                continue
            instances = self.id2instances.get(img_id, [])
            instances_tmp = []
            for instance in instances:
                key = 'bbox' if img_id.split('_')[0] == 'coco' else 'instance_bbox'
                x, y, w, h = instance[key]
                if w < min_box_wh_size or h < min_box_wh_size:
                    continue
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
            prefix_path = f'/COCO/{self.pattern}2017'
            dataset_type = 0
        elif data_set == 'vaw':
            prefix_path = f'/VG/VG_100K'
            dataset_type = 1
        else:
            raise NameError
        results = {}
        results['img_prefix'] = os.path.abspath(self.data_root) + prefix_path
        results['img_info'] = {}
        results['img_info']['filename'] = img_info['file_name']

        bbox_list = []
        attr_label_list = []
        for instance in instances:
            key = 'bbox' if data_set == 'coco' else 'instance_bbox'
            x, y, w, h = instance[key]
            bbox_list.append([x, y, x + w, y + h])
            positive_attributes = instance.get("positive_attributes", [])
            negative_attributes = instance.get("negative_attributes", [])
            labels = np.ones(len(self.att2id.keys())) * 2
            for att in positive_attributes:
                labels[self.att2id[att]] = 1
            for att in negative_attributes:
                labels[self.att2id[att]] = 0
            attr_label_list.append(labels)

        gt_bboxes = np.array(bbox_list, dtype=np.float32)
        gt_labels = np.stack(attr_label_list, axis=0)
        results['gt_bboxes'] = gt_bboxes
        results['bbox_fields'] = ['gt_bboxes']
        results['gt_labels'] = gt_labels.astype(np.int)
        results['dataset_type'] = dataset_type
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
            if len(self.error_list) > 20:
               raise UnboundLocalError
            if not self.test_mode:
                results = self.__getitem__(np.random.randint(0, len(self)))
        return results

    def get_test_img_instances(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.id2images[img_id]
        instances = self.id2instances[img_id]

        data_set = img_id.split('_')[0]
        if data_set == 'coco':
            raise NameError
        elif data_set == 'vaw':
            prefix_path = f'/VG/VG_100K'
            dataset_type = 1
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

    def get_test_rpn_img_instances(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.id2images[img_id]

        data_set = img_id.split('_')[0]
        if data_set == 'coco':
            raise NameError
        elif data_set == 'vaw':
            prefix_path = f'/VG/VG_100K'
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
            if self.test_rpn:
                self.get_test_rpn_img_instances(idx)
            return self.get_test_img_instances(idx)
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
            labels[self.classname_maps[att]] = 1
        for att in negative_attributes:
            labels[self.classname_maps[att]] = 0

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
        np_gt_labels = []
        for results in self.instances:
            positive_attributes = results['positive_attributes']
            negative_attributes = results['negative_attributes']
            label = np.ones(len(self.classname_maps.keys())) * 2
            for att in positive_attributes:
                label[self.classname_maps[att]] = 1
            for att in negative_attributes:
                label[self.classname_maps[att]] = 0

            gt_labels = label.astype(np.int)

            np_gt_labels.append(gt_labels.astype(np.int))
        return np.stack(np_gt_labels, axis=0)

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
                    labels[self.att2id[att]] = 1
                for att in negative_attributes:
                    labels[self.att2id[att]] = 0

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

    def evaluate_rpn(self, results):
        # results List[Tensor] N, Nx(4+1+620)
        # gt_labels List[Tensor] N, Nx(4+620)
        gt_labels = self.get_rpn_img_instance_labels()

        print('Computing!')
        gt_bboxes = [gt[:, :4].numpy() for gt in gt_labels]
        proposals = [x[:, :5].numpy() for x in results]
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
        return ar[0]

    def evaluate(self,
                 results,
                 logger=None,
                 metric='mAP',
                 per_class_out_file=None,
                 is_logit=True
                 ):
        if self.test_rpn:
            return self.evaluate_rpn(results)

        if isinstance(results[0], type(np.array(0))):
            results = np.concatenate(results, axis=0)
        else:
            results = np.array(results)

        preds = torch.from_numpy(results)
        gts = self.get_img_instance_labels()
        gts = torch.from_numpy(gts)
        assert len(preds) == len(gts)

        output = cal_metrics(self.data_root + '/VAW',
                             preds, gts,
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

