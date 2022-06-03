import glob
import os
import json




import json
import os

import tqdm
from textblob import TextBlob
from textblob import Word
from textblob.taggers import NLTKTagger
import multiprocessing

nltk_tagger = NLTKTagger()


def get_key_freq(src_keys, target_data, kv_list, pid):

    for key in tqdm.tqdm(src_keys):
        key = key.lower()
        count_num = 0
        for tgt_text in target_data:
            count_num += tgt_text.lower().count(key)
        kv_list.append({key: count_num})


def split_json_data(text_list, split_num, path):
    os.makedirs(path, exist_ok=True)
    n_item_per_slice = len(text_list) // split_num
    for i in range(split_num):
        start = i * n_item_per_slice
        end = min(start + n_item_per_slice, len(json_data))
        json.dump(text_list[start: end], open(path+f'/split_{i}.json', 'w'))


def gather_all(path, split_num):
    return_data = {'num_atts': 0, 'num_categories': 0, 'atts': {}, 'categories': {}}
    for i in range(split_num):
        data = json.load(open(path + f'/split_{i}_atts_categories.json', 'r'))
        for k, v in data['atts'].items():
            return_data['atts'][k] = return_data['atts'].get(k, 0) + v
        for k, v in data['categories'].items():
            return_data['categories'][k] = return_data['categories'].get(k, 0) + v

    return_data['atts'] = sorted(return_data['atts'].items(), key=lambda kv: kv[1], reverse=True)
    return_data['categories'] = sorted(return_data['categories'].items(), key=lambda kv: kv[1], reverse=True)
    return_data['num_atts'] = len(return_data['atts'])
    return_data['num_categories'] = len(return_data['categories'])
    return return_data


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    n_process = 32

    src_data = json.load(open('attribute_all/all_attributes.json', 'r'))['attributes']
    data_slice_list = []
    n_item_per_slice = len(src_data) // n_process
    for i in range(n_process):
        start = i * n_item_per_slice
        end = min(start + n_item_per_slice, len(src_data))
        data_slice_list.append(src_data[start: end])

    target_data = json.load(open('../captions/caption_all/caption_seg_word.json', 'r'))['captions']

    kv_list = multiprocessing.Manager().list()
    process_list = []
    for pid in range(n_process):
        print('pid {}'.format(pid))
        process_list.append(
            multiprocessing.Process(target=get_key_freq, args=(data_slice_list[pid], target_data, kv_list, pid))
        )
    [p.start() for p in process_list]
    [p.join() for p in process_list]

    kv_list = list(kv_list)
    kv_dict = {}
    for kv in kv_list:
        kv_dict.update(kv)
    kv_dict = sorted(kv_dict.items(), key=lambda kv: kv[1], reverse=True)
    json_data = {'num': len(kv_dict), 'attributes': kv_dict}
    json.dump(json_data, open('attribute_all/all_attributes_with_freq.json', 'w'), indent=4)
