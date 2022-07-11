import json
from collections import OrderedDict

import tqdm

data_root = '/data/kyanchen/prompt/data/VAW/'
attr_dict = json.load(open(data_root + 'attribute_index.json'))

json_file_list = ["train_part1.json", "train_part2.json", 'test.json', 'val.json']
json_data = [json.load(open(data_root + x)) for x in json_file_list]
instances = []
[instances.extend(x) for x in json_data]
freq_attr = OrderedDict()

for instance in tqdm.tqdm(instances):
    positive_attributes = instance['positive_attributes']
    negative_attributes = instance['negative_attributes']
    category = instance['object_name']
    freq_attr[category] = freq_attr.get(category, {})
    freq_attr[category]['n_instance'] = freq_attr[category].get('n_instance', 0) + 1
    freq_attr[category]['pos'] = freq_attr[category].get('pos', {})
    freq_attr[category]['neg'] = freq_attr[category].get('neg', {})
    for item in positive_attributes:
        freq_attr[category]['pos'][item] = freq_attr[category]['pos'].get(item, 0) + 1

    for item in negative_attributes:
        freq_attr[category]['neg'][item] = freq_attr[category]['neg'].get(item, 0) + 1

category_instances = OrderedDict()
for k, v in freq_attr.items():
    category_instances[k] = v['n_instance']
json.dump(category_instances, open(data_root + 'category_instances.json', 'w'), indent=4)
json.dump(freq_attr, open(data_root + 'category_attr_pair.json', 'w'), indent=4)

