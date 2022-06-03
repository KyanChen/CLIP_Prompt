import json
import os

import pandas
import tqdm
import multiprocessing


def get_key_freq(src_keys, target_data, path, pid):
    kv_dict = {}
    for key in tqdm.tqdm(src_keys):
        key = key.lower()
        count_num = 0
        for tgt_text in target_data:
            if pandas.isna(tgt_text):
                continue
            try:
                count_num += tgt_text.lower().count(key)
            except Exception as e:
                print(e)
        kv_dict[key] = count_num
    json.dump(kv_dict, open(path + f'/split_{pid}_with_freq.json', 'w'), indent=4)


def split_json_data(text_list, split_num, path):
    os.makedirs(path, exist_ok=True)
    n_item_per_slice = len(text_list) // split_num
    for i in range(split_num):
        start = i * n_item_per_slice
        end = min(start + n_item_per_slice, len(json_data))
        json.dump(text_list[start: end], open(path+f'/split_{i}.json', 'w'))


def gather_all(path, split_num):
    return_data = {'num_atts': 0, 'atts': {}}
    for i in range(split_num):
        data = json.load(open(path + f'/split_{i}_with_freq.json', 'r'))
        return_data['atts'].update(data)
    return_data['atts'] = sorted(return_data['atts'].items(), key=lambda kv: kv[1], reverse=True)
    return_data['num_atts'] = len(return_data['atts'])
    return return_data


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    n_process = 16

    src_data = json.load(open('../gather_infos/all_objects.json', 'r'))['objects']
    data_slice_list = []
    n_item_per_slice = len(src_data) // n_process
    for i in range(n_process):
        start = i * n_item_per_slice
        end = start + n_item_per_slice
        if i == n_process - 1:
            end = len(src_data)
        data_slice_list.append(src_data[start: end])

    target_data = json.load(open('../captions/caption_all/caption_seg_word.json', 'r'))['captions']

    tmp_path = 'object_all/tmp'
    os.makedirs(tmp_path, exist_ok=True)
    process_list = []
    for pid in range(n_process):
        print('pid {}'.format(pid))
        process_list.append(
            multiprocessing.Process(target=get_key_freq, args=(data_slice_list[pid], target_data, tmp_path, pid))
        )
    [p.start() for p in process_list]
    [p.join() for p in process_list]

    json_data = gather_all(tmp_path, split_num=n_process)
    json.dump(json_data, open('../gather_infos/all_objects_with_freq.json', 'w'), indent=4)
