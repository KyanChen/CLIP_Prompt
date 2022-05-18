import io
import itertools
import json
import logging
import os
import os.path as osp
import pickle
import random
import tempfile
import warnings
from collections import OrderedDict, defaultdict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from ..datasets.builder import DATASETS
from torch.utils.data import Dataset
from ..datasets.pipelines import Compose


@DATASETS.register_module()
class VAWDataset(Dataset):

    CLASSES = None

    def __init__(self,
                 data_root,
                 pipeline,
                 num_shots,
                 pattern,
                 seed=1,
                 test_mode=False,
                 file_client_args=dict(backend='disk')
                 ):
        assert pattern in ['train', 'val', 'test']
        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        self.image_dir = os.path.join(self.data_root, "/VG/images")
        self.preprocessed = os.path.join(self.data_root, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.data_root, "split_fewshot")
        mmcv.mkdir_or_exist(self.split_fewshot_dir)

        text_file = os.path.join(self.data_root, "/VAW/attribute_index.json")
        classname_maps = json.load(open(text_file))

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
        else:
            train = self.read_data(classname_maps, ["train_part1.json", "train_part2.json"])
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            val = self.read_data(classname_maps, ['val.json'])
            test = self.read_data(classname_maps, ['test.json'])

            preprocessed = {"train": train, "val": val, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(10 * '*', 'All Dataset', 10 * '*')
        [print(k, ': ', len(v)) for k, v in preprocessed.items()]
        print(10 * '*', 'All Dataset', 10 * '*')

        if num_shots >= 1:
            split_path = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(split_path):
                print(f"Loading preprocessed few-shot data from {split_path}")
                with open(split_path, "rb") as file:
                    data = pickle.load(file)
            else:
                train = self.generate_fewshot_dataset(preprocessed['train'], num_shots=num_shots)
                val = self.generate_fewshot_dataset(preprocessed['val'], num_shots=int(0.01 * len(val)))
                test = self.generate_fewshot_dataset(preprocessed['test'], num_shots=int(0.01 * len(test)))

                data = {"train": train, "val": val, "test": test}

                print(f"Saving preprocessed few-shot data to {split_path}")

                with open(split_path, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            data = preprocessed

        print(10 * '*', 'Split Dataset', 10 * '*')
        [print(k, ': ', len(v)) for k, v in data.items()]
        print(10 * '*', 'Split Dataset', 10 * '*')
        self.data = data[pattern]

        self.num_classes = max(classname_maps.values()) + 1
        self.lab2cname = {v: k for k, v in classname_maps.items()}
        self.classnames = list(self._lab2cname.values())
        self.CLASSES = self.classnames

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            idxes = np.nonzero(item.label + 1)
            for idx in idxes[0]:
                output[idx].append(item)

        return output

    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=False):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed (default: False).
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f"Creating a {num_shots}-shot dataset")

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def read_data(self, classname_maps, json_file_list):
        json_data = [json.load(open(self.dataset_dir + '/' + x)) for x in json_file_list]
        datas = []
        [datas.extend(x) for x in json_data]
        items = []
        for data in datas:
            # for x in data['positive_attributes']:
            item = DataItem(
                image_id=data['image_id'],
                instance_id=data['instance_id'],
                instance_bbox=data['instance_bbox'],
                object_name=data['object_name'],
                positive_attributes=data['positive_attributes'],
                negative_attributes=data['negative_attributes'],
                label=None)
            item.set_label(classname_maps=classname_maps)

            items.append(item)
        # print(json_file_list[0], ' instances: ', len(items))
        return items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item = self.data[item]
        results = item.__dict__.copy()
        results['img_prefix'] = os.path.abspath(self.data_root) + '/VG/VG_100K'
        results['img_info'] = {}
        results['img_info']['filename'] = f'{item.image_id}.jpg'
        results['instance_bbox'] = item.instance_bbox
        results['gt_labels'] = item.label
        results = self.pipeline(results)
        return results

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = self.evaluate_det_segm(results, result_files, coco_gt,
                                              metrics, logger, classwise,
                                              proposal_nums, iou_thrs,
                                              metric_items)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results



#     "image_id": "2373241",
#     "instance_id": "2373241004",
#     "instance_bbox": [0.0, 182.5, 500.16666666666663, 148.5],
#     "instance_polygon": [[[432.5, 214.16666666666669], [425.8333333333333, 194.16666666666666], [447.5, 190.0], [461.6666666666667, 187.5], [464.1666666666667, 182.5], [499.16666666666663, 183.33333333333331], [499.16666666666663, 330.0], [3.3333333333333335, 330.0], [0.0, 253.33333333333334], [43.333333333333336, 245.0], [60.833333333333336, 273.3333333333333], [80.0, 293.3333333333333], [107.5, 307.5], [133.33333333333334, 309.16666666666663], [169.16666666666666, 295.8333333333333], [190.83333333333331, 274.1666666666667], [203.33333333333334, 252.5], [225.0, 260.0], [236.66666666666666, 254.16666666666666], [260.0, 254.16666666666666], [288.3333333333333, 253.33333333333334], [287.5, 257.5], [271.6666666666667, 265.0], [324.1666666666667, 281.6666666666667], [369.16666666666663, 274.1666666666667], [337.5, 261.6666666666667], [338.3333333333333, 257.5], [355.0, 261.6666666666667], [357.5, 257.5], [339.1666666666667, 255.0], [337.5, 240.83333333333334], [348.3333333333333, 238.33333333333334], [359.1666666666667, 248.33333333333331], [377.5, 251.66666666666666], [397.5, 248.33333333333331], [408.3333333333333, 236.66666666666666], [418.3333333333333, 220.83333333333331], [427.5, 217.5], [434.16666666666663, 215.0]]],
#     "object_name": "floor",
#     "positive_attributes": ["tiled", "gray", "light colored"],
#     "negative_attributes": ["multicolored", "maroon", "weathered", "speckled", "carpeted"]
# }

class DataItem:

    def __init__(
            self, image_id, instance_id, instance_bbox,
            object_name, positive_attributes, negative_attributes,
            label
    ) -> None:
        self.image_id = image_id
        self.instance_id = instance_id
        self.instance_bbox = instance_bbox
        self.object_name = object_name
        self.positive_attributes = positive_attributes
        self.negative_attributes = negative_attributes
        self.label = label

    def set_label(self, classname_maps):
        self.label = np.ones(len(classname_maps.keys())) * 2
        for att in self.positive_attributes:
            self.label[classname_maps[att]] = 1
        for att in self.negative_attributes:
            self.label[classname_maps[att]] = 0
