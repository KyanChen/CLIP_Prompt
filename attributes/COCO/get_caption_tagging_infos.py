import json
import copy
import re
from textblob import TextBlob
from tqdm import tqdm

parent_folder = '../../data/COCO/annotations'
# parent_folder = '/Users/kyanchen/Documents/COCO/annotations'
json_file = parent_folder+'/captions_train2017.json'
json_data = json.load(open(json_file, 'r'))
caption_anns = json_data['annotations']
extracted_data = {}
for ann in caption_anns:
    image_id = ann['image_id']
    caption = ann['caption']
    if caption[-1] not in ['.', '。', '?', '!', ';']:
        caption += '.'
    extracted_data[image_id] = extracted_data.get(image_id, {})
    extracted_data[image_id]['caption'] = extracted_data[image_id].get('caption', []) + [caption]

categories = json.load(open('common2common_category2id_48_32.json', 'r'))
categories = list(categories['common1'].keys()) + list(categories['common2'].keys())
print('len category: ', len(categories))

atts = json.load(open('../VAW/common2rare_att2id.json', 'r'))
atts = list(atts['common'].keys()) + list(atts['rare'].keys())
print('len att: ', len(atts))


def punc_filter(text):
    # rule = re.compile(r'[^\-a-zA-Z0-9]')
    rule = re.compile('^\w+$')  # 由数字、26个英文字母或者下划线组成的字符串
    text = rule.sub(' ', text)
    text = ' '.join([x.strip() for x in text.split(' ') if len(x.strip()) > 0])
    return text

extracted_data_tmp = extracted_data.copy()
for img_id, item in tqdm(extracted_data_tmp.items()):
    extracted_data[img_id]['category'] = []
    extracted_data[img_id]['attribute'] = []
    captions = item['caption']
    # for caption in captions:
    caption = ' '.join(captions)
    caption = punc_filter(caption)
    caption = caption.lower()
    for category in categories:
        rex = re.search(rf'\b{category}\b', caption)
        if rex is not None:
            extracted_data[img_id]['category'] += [category]
    for att in atts:
        rex = re.search(rf'\b{att}\b', caption)
        if rex is not None:
            extracted_data[img_id]['attribute'] += [att]

    speech = TextBlob(caption)
    noun_phrases = [str(x) for x in speech.noun_phrases]
    # extracted_data[img_id]['phase'] += noun_phrases
    extracted_data[img_id]['phase'] = list(set(noun_phrases))
    extracted_data[img_id]['category'] = list(set(extracted_data[img_id]['category']))
    extracted_data[img_id]['attribute'] = list(set(extracted_data[img_id]['attribute']))

json.dump(extracted_data, open(parent_folder+'/train_2017_caption_tagging.json', 'w'), indent=4)
