import glob
import json
import logging
import os
import os.path as osp
import pickle
import random
import tempfile
import warnings
from collections import OrderedDict, defaultdict
from pprint import pprint

import cv2
import imagesize
import mmcv
import numpy as np
import torch
from mmcv.runner import get_dist_info

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
                 dataset_names='vaw',
                 load_label=None,
                 save_label=False,
                 file_client_args=dict(backend='disk')
                 ):

        assert dataset_split in ['train', 'val', 'test']
        self.dataset_split = dataset_split
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        self.dataset_names = dataset_names

        self.attribute_index_file = attribute_index_file
        self.att2id = {}
        if 'att_file' in attribute_index_file.keys():
            file = attribute_index_file['att_file']
            att2id = json.load(open(file, 'r'))
            att_group = attribute_index_file['att_group']
            if att_group in ['common1', 'common2', 'common', 'rare']:
                self.att2id = att2id[att_group]
            elif att_group == 'common1+common2':
                self.att2id.update(att2id['common1'])
                self.att2id.update(att2id['common2'])
            elif att_group == 'common+rare':
                self.att2id.update(att2id['common'])
                self.att2id.update(att2id['rare'])
            else:
                raise NameError
        self.category2id = {}
        if 'category_file' in attribute_index_file.keys():
            file = attribute_index_file['category_file']
            category2id = json.load(open(file, 'r'))
            att_group = attribute_index_file['category_group']
            if att_group in ['common1', 'common2', 'common', 'rare']:
                self.category2id = category2id[att_group]
            elif att_group == 'common1+common2':
                self.category2id.update(category2id['common1'])
                self.category2id.update(category2id['common2'])
            elif att_group == 'common+rare':
                self.category2id.update(category2id['common'])
                self.category2id.update(category2id['rare'])
            else:
                raise NameError
        self.att2id = {k: v - min(self.att2id.values()) for k, v in self.att2id.items()}
        self.category2id = {k: v - min(self.category2id.values()) for k, v in self.category2id.items()}

        if open_category:
            print('open_category: ', open_category)
            self.instances, self.img_instances_pair = self.read_data(["train_part1.json", "train_part2.json", 'val.json', 'test.json'])
            self.instances = self.split_instance_by_category(pattern=pattern)
        else:
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

            self.instances = []
            for k, v in self.id2instances.items():
                for item in v:
                    item['img_id'] = k
                    self.instances.append(item)

        rank, world_size = get_dist_info()

        if not test_mode:
            self.instances = self.filter_instance(self.instances)
            flag_dataset = [x['img_id'].split('_')[0] for x in self.instances]
            dataset_types = {'coco': 0, 'vaw': 1, 'ovadgen': 1}
            flag_dataset = [dataset_types[x] for x in flag_dataset]
            self.flag_dataset = np.array(flag_dataset, dtype=np.int)

        self.flag = np.zeros(len(self), dtype=int)

        if rank == 0:
            print('data len: ', len(self))
            print('num_att: ', len(self.att2id))
            print('num_category: ', len(self.category2id))
        self.error_list = set()

        self.save_label = save_label
        if load_label:
            self.pred_labels = np.load(load_label)
            assert len(self) == len(self.pred_labels)

    def filter_instance(self, instances):
        return_instances = []
        for instance in instances:
            img_id = instance['img_id']
            data_set = img_id.split('_')[0]
            if data_set == 'coco':
                category = instance['name']
                category_id = self.category2id.get(category, None)  # 未标注的该类别的应该去除
                if category_id is not None:
                    return_instances.append(instance)
            elif data_set == 'vaw':
                return_instances.append(instance)
            elif data_set == 'ovadgen':
                return_instances.append(instance)
        return return_instances

    def read_data_ovadgen(self, pattern):
        instances = glob.glob(self.data_root + '/ovadgen/*.jpg')
        instances = [x for x in instances if os.path.getsize(x) > 25 * 1024]

        id2images = {}
        id2instances = {}
        for idx, data in enumerate(instances):
            img_id = 'ovadgen_' + str(idx)
            id2images[img_id] = {}
            id2images[img_id]['file_name'] = os.path.basename(data)
        id2att = {v: k for k, v in self.att2id.items()}
        for idx, data in enumerate(instances):
            img_id = 'ovadgen_' + str(idx)
            instance = {}
            img_name = os.path.basename(data)
            id_att = int(img_name.split('_')[-2])
            att_name = '_'.join(img_name.split('_')[:-2])
            target_att = id2att[id_att]

            att_type, att_names = target_att.split(':')
            target_atts = [x + ' ' + att_type for x in att_names.split('/')]
            target_atts = [x.replace(' ', '_') for x in target_atts]
            assert att_name in target_atts
            instance['positive_attributes'] = [target_att]
            instance['negative_attributes'] = []
            id2instances[img_id] = id2instances.get(img_id, []) + [instance]

        return id2images, id2instances

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
        # instances = instances[:1024]
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

    def read_data_ovad(self, pattern):
        json_data = json.load(open('../attributes/OVAD/ovad1200_licence.json', 'r'))
        instances = json_data['annotations']
        categoryid2name = {x['id']: x['name'] for x in json_data['categories']}
        attributeid2name = {x['id']: x for x in json_data['attributes']}
        id2images = {}
        id2instances = {}
        for instance in instances:
            img_id = f'ovad{pattern}_' + str(instance['image_id'])
            instance['name'] = categoryid2name[instance['category_id']]
            instance['positive_attributes'] = []
            instance['negative_attributes'] = []
            for idx, att_ann in enumerate(instance['att_vec']):
                '''
                % 1 = positive attribute
                % 0 = negative attribute
                % -1 = ignore attribute
                '''
                if att_ann in [0, 1]:
                    att = attributeid2name[idx]['name']
                    if att_ann == 1:
                        instance['positive_attributes'] += [att]
                    else:
                        instance['negative_attributes'] += [att]
            id2instances[img_id] = id2instances.get(img_id, []) + [instance]

        for data in json_data['images']:
            img_id = f'ovad{pattern}_' + str(data['id'])
            id2images[img_id] = data

        return id2images, id2instances

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

    def get_generated_sample(self, idx):
        instance = self.instances[idx]
        results = {}
        results['img_prefix'] = ''
        results['img_info'] = {}
        results['img_info']['filename'] = instance
        labels = np.ones(len(self.att2id.keys())) * 2
        att = ' '.join(os.path.basename(instance).split('_')[:-2])
        att_id = self.att2id.get(att, None)
        labels[att_id] = 1
        if hasattr(self, 'pred_labels'):
            thresh_low = 0.1
            thresh_high = 0.5
            thresh_topk = 3
            pred_label = torch.from_numpy(self.pred_labels[idx])
            idx_tmp = torch.nonzero(pred_label < thresh_low)[:, 0]
            labels[idx_tmp] = 0
            # values, idx_tmp = torch.topk(-pred_label, k=thresh_topk)
            # labels[idx_tmp] = 0
            # idx_tmp = torch.nonzero(pred_label > thresh_high)[:, 0]
            # labels[idx_tmp] = 1
            # values, idx_tmp = torch.topk(pred_label, k=thresh_topk)
            # labels[idx_tmp] = 1

        results['gt_labels'] = labels.astype(np.int)
        results = self.pipeline(results)
        return results

    def __getitem__(self, idx):
        if idx in self.error_list and not self.test_mode:
            idx = np.random.randint(0, len(self))
        instance = self.instances[idx]
        img_id = instance['img_id']
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
        results['data_set_type'] = data_set_type
        results['img_prefix'] = os.path.abspath(self.data_root) + prefix_path
        results['img_info'] = {}
        results['img_info']['filename'] = img_info['file_name']
        if 'gen' in data_set:
            pass
        else:
            key = 'bbox' if data_set in ['coco', 'ovadattr', 'ovadcate'] else 'instance_bbox'
            x, y, w, h = instance[key]
            results['crop_box'] = np.array([x, y, x + w, y + h])
        if self.test_mode:
            results = self.pipeline(results)
            # try:
            #
            # except Exception as e:
            #     print(f'idx: {idx}')
            #     print(f'img_id: {img_id}')
        else:
            try:
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
                results['gt_labels'] = labels.astype(np.int)
                if 'gen' in data_set:
                    results = self.pipeline(results, 0)
                    results = self.pipeline(results, (2, ':'))
                else:
                    results = self.pipeline(results)
            except Exception as e:
                self.error_list.add(idx)
                self.error_list.add(img_id)
                print(self.error_list)
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
        return results

    def get_labels(self):
        np_gt_labels = []
        for instance in self.instances:
            labels = np.ones(len(self.att2id) + len(self.category2id)) * 2
            labels[-len(self.category2id):] = 0
            img_id = instance['img_id']
            img_info = self.id2images[img_id]
            data_set = img_id.split('_')[0]
            # import pdb
            # pdb.set_trace()
            if data_set in ['vaw', 'ovadattr']:
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
            if data_set in ['coco', 'ovadcate']:
                category = instance['name']
                category_id = self.category2id.get(category, None)
                if category_id is not None:
                    labels[category_id + len(self.att2id)] = 1
            np_gt_labels.append(labels.astype(np.int))
        return np.stack(np_gt_labels, axis=0)

    def get_data_set_type(self):
        data_set_types = []
        for instance in self.instances:
            img_id = instance['img_id']
            img_info = self.id2images[img_id]
            data_set = img_id.split('_')[0]
            if data_set == 'coco':
                data_set_type = 0
            elif data_set == 'vaw':
                data_set_type = 1
            elif data_set == 'ovadcate':
                data_set_type = 0
            elif data_set == 'ovadattr':
                data_set_type = 1
            else:
                raise NameError
            data_set_types.append(data_set_type)
        return np.array(data_set_types)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 per_class_out_file=None,
                 is_logit=True
                 ):
        result_metrics = OrderedDict()

        results = np.array(results)
        preds = torch.from_numpy(results)
        gt_labels = self.get_labels()
        gt_labels = torch.from_numpy(gt_labels)

        data_set_type = self.get_data_set_type()
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

            pred_cate_logits = pred_cate_logits.detach().sigmoid().cpu()
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
                save_result=True
            )
            # import pdb
            # pdb.set_trace()
            # print(output)
            result_metrics['cate_ap_all'] = output['PC_ap/all']

            # pred_pos_mask = pred_prob > 0.5
            # pred_neg_mask = pred_prob <= 0.5
            #
            # tp = torch.sum(gt_cate[pred_pos_mask][torch.arange(len(gt_cate[pred_pos_mask])), pred_label[pred_pos_mask]] == 1)
            # tn = torch.sum(torch.sum(gt_cate[pred_neg_mask], dim=-1) == 0)
            # fp = torch.sum(gt_cate[pred_pos_mask][torch.arange(len(gt_cate[pred_pos_mask])), pred_label[pred_pos_mask]] == 0)
            # fn = torch.sum(torch.sum(gt_cate[pred_neg_mask], dim=-1) == 1)
            #
            # result_metrics['cate_precision'] = tp / torch.sum(pred_pos_mask)
            # result_metrics['cate_recall'] = tp / (tp + fn)
            # result_metrics['cate_acc'] = (tp + tn) / (tp + tn + fp + fn)
            # result_metrics['cate_f1'] = 2 * result_metrics['cate_precision'] * result_metrics['cate_recall'] / (result_metrics['cate_precision'] + result_metrics['cate_recall'])
            # result_metrics['cate_num_tp'] = tp
            # result_metrics['cate_num_tn'] = tn
            # result_metrics['cate_num_fp'] = fp
            # result_metrics['cate_num_fn'] = fn
            # print()
            # for k, v in result_metrics.items():
            #     value = f'{v.item():.4f}' if 'num' not in k else f'{v.item()}'
            #     print(k, '\t', value)
            #     result_metrics[k] = v.item()

        if self.save_label:
            np.save(self.save_label, preds.data.cpu().float().sigmoid().numpy())
        assert pred_att_logits.shape[-1] == gt_att.shape[-1]

        if not len(self.att2id):
            return result_metrics

        dataset_name = self.attribute_index_file['att_file'].split('/')[-2]
        top_k = 15 if dataset_name == 'VAW' else 8

        pred_att_logits = pred_att_logits.data.cpu().float().sigmoid().numpy()  # Nx620
        gt_att = gt_att.data.cpu().float().numpy()  # Nx620

        output = cal_metrics(
            self.att2id,
            dataset_name,
            prefix_path=f'../attributes/{dataset_name}',
            pred=pred_att_logits,
            gt_label=gt_att,
            top_k=top_k,
            save_result=True
        )
        result_metrics['att_ap_all'] = output['PC_ap/all']
        return result_metrics

        # output = cal_metrics(
        #     f'../attributes/{dataset_name}',
        #     preds[:, :len(self.att2id)], gts[:, :len(self.att2id)],
        #     fpath_attribute_index=self.attribute_index_file,
        #     return_all=True,
        #     return_evaluator=per_class_out_file,
        #     is_logit=is_logit
        # )
        if per_class_out_file:
            scores_overall, scores_per_class, scores_overall_topk, scores_per_class_topk, evaluator = output
        else:
            scores_overall, scores_per_class, scores_overall_topk, scores_per_class_topk = output

        # CATEGORIES = ['all', 'head', 'medium', 'tail'] + \
        # list(evaluator.attribute_parent_type.keys())

        CATEGORIES = ['all']

        for category in CATEGORIES:
            print(f"----------{category.upper()}----------")
            print(f"mAP: {scores_per_class[category]['ap']:.4f}")
            result_metrics['all_mAP'] = scores_per_class['all']['ap']

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

        return result_metrics


