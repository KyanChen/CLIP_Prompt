import json
from collections import OrderedDict

import tqdm

data_root = '/data1/kyanchen/prompt/data/VAW/'
attr_dict = json.load(open(data_root + 'attribute_index.json'))


json_file_list = ["train_part1.json", "train_part2.json"]
json_data = [json.load(open(data_root + x)) for x in json_file_list]
instances = []
[instances.extend(x) for x in json_data]
freq_attr = OrderedDict()
for key, value in attr_dict.items():
    freq_attr[key] = 0
for instance in tqdm.tqdm(instances):
    positive_attributes = instance['positive_attributes']
    negative_attributes = instance['negative_attributes']
    for item in positive_attributes:
        freq_attr[item] = freq_attr[item] + 1
    for item in negative_attributes:
        freq_attr[item] = freq_attr[item] + 1
json.dump(freq_attr, open(data_root + 'attr_freq.json', 'w'), indent=4)
