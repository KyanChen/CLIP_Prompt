import json
import os

import tqdm
from textblob import TextBlob
from textblob import Word
from textblob.taggers import NLTKTagger
import multiprocessing

nltk_tagger = NLTKTagger()


def get_att_categories(pid, path):
    data = json.load(open(path + f'/split_{pid}.json', 'r'))
    return_data = {'atts': {}, 'categories': {}}
    for text_str in tqdm.tqdm(data):
        blob = TextBlob(text_str, pos_tagger=nltk_tagger)
        # print(blob.pos_tags)
        for word, tag in blob.pos_tags:
            if tag == 'JJ':
                return_data['atts'][word] = return_data['atts'].get(word, 0) + 1
            elif tag == 'NN':
                return_data['categories'][word] = return_data['categories'].get(word, 0) + 1
            elif tag == 'NNS':
                word = Word(word).singularize()
                return_data['categories'][word] = return_data['categories'].get(word, 0) + 1

    json.dump(return_data, open(path + f'/split_{pid}_atts_categories.json', 'w'), indent=4)


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

    json_data = json.load(open('caption_all/caption_seg_word.json', 'r'))['captions']
    split_json_data(json_data, split_num=n_process, path='caption_all/tmp')

    process_list = []
    for pid in range(n_process):
        print('pid {}'.format(pid))
        process_list.append(
            multiprocessing.Process(target=get_att_categories, args=(pid, 'caption_all/tmp'))
        )
    [p.start() for p in process_list]
    [p.join() for p in process_list]

    return_data = gather_all(path='caption_all/tmp', split_num=n_process)
    json.dump(return_data, open(f'caption_all/extracted_atts_categories.json', 'w'), indent=4)
