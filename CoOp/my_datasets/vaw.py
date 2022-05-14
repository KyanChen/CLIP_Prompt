import json
import os
import pickle
from collections import OrderedDict, defaultdict
import random

import numpy as np

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing


@DATASET_REGISTRY.register()
class VAW(DatasetBase):

    dataset_dir = "vaw"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        text_file = os.path.join(self.dataset_dir, "attribute_index.json")
        classname_maps = self.read_classname_maps(text_file)
        
        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                val = preprocessed["val"]
                test = preprocessed["test"]
                [print(k, ': ', len(v)) for k,v in preprocessed.items()]
        else:
            
            train = self.read_data(classname_maps, ["train_part1.json", "train_part2.json"])
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            val = self.read_data(classname_maps, ['val.json'])
            test = self.read_data(classname_maps, ['test.json'])

            preprocessed = {"train": train, "val": val, "test": test}
            
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"][:2]
                    val = data["val"]
                    test = data["test"]
                    [print(k, ': ', len(v)) for k,v in data.items()]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)

                val = self.generate_fewshot_dataset(val, num_shots=int(0.01*len(val)))
                test = self.generate_fewshot_dataset(test, num_shots=int(0.01*len(test)))
                data = {"train": train, "val": val, "test": test}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        # subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        # train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        self._train_x = train  # labeled training data
        self._val = val  # validation data (optional)
        self._train_u = None
        self._test = test  # test data
        self._num_classes = max(classname_maps.values())
        self._lab2cname = {v:k for k,v in classname_maps.items()}
        self._classnames = list(self._lab2cname.values())

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            idxes = np.nonzero(item.label+1)
            for idx in idxes[0]:
                output[idx].append(item)

        return output
    
    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=False
    ):
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

    @staticmethod
    def read_classname_maps(text_file):
        """Return a dictionary containing
        key-value pairs of "amber": 0
        """
        classname_maps = json.load(open(text_file))
        return classname_maps

    def read_data(self, classname_maps, json_file_list):
        # split_dir = os.path.join(self.image_dir, split_dir)
        # folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        # items = []

        # for label, folder in enumerate(folders):
        #     imnames = listdir_nohidden(os.path.join(split_dir, folder))
        #     classname = classnames[folder]
        #     for imname in imnames:
        #         impath = os.path.join(split_dir, folder, imname)
        #         item = Datum(impath=impath, label=label, classname=classname)
        #         items.append(item)
        json_data = [json.load(open(self.dataset_dir+'/'+x)) for x in json_file_list]
        datas = []
        [datas.extend(x) for x in json_data]
        print(json_file_list[0], ' samples : ', len(datas))
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
        print(json_file_list[0], ' instances: ', len(items))
        return items 

# {
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
        self.label = np.ones(len(classname_maps.keys()))*-1
        for att in self.positive_attributes:
            self.label[classname_maps[att]] = 1
        for att in self.negative_attributes:
            self.label[classname_maps[att]] = 0