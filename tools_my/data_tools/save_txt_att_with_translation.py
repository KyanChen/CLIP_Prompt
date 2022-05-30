import time
from googletrans import Translator
import json

# 设置Google翻译服务地址
dataset = 'VAW'
pattern = 'attributes'

translator = Translator(service_urls=['translate.google.cn'])

# txt_path = f'../../attributes/{dataset}/{pattern}.txt'
# txt_data_lines = open(txt_path, 'r').readlines()
# att_data = {}


json_path = f'../../attributes/{dataset}/{pattern}.json'
json_data_lines = json.load(open(json_path, 'r'))
att_data = json_data_lines

# for data in txt_data_lines:
#     att, sub_att = data.split(':')
#
#     att = att.strip().split('_')[-1].replace('_', ' ').lower()
#     sub_att = sub_att.strip().replace('_', ' ').lower().split(',')
#     sub_att = [x.strip() for x in sub_att]
#     att_data[att] = att_data.get(att, [])
#     att_data[att] += sub_att


# txt_data_lines = txt_data_lines[0].strip().split(' ')
# for data in txt_data_lines:
#     # att, sub_att = data.split('::')
#     sub_att = data.split(":")[-1].replace(',', '/')
#     att = sub_att
#
#     att = att.strip().replace('_', ' ').lower()
#     sub_att = sub_att.strip().replace('_', ' ').lower().split('(')[0].strip()
#     att_data[att] = att_data.get(att, [])
#     att_data[att] += [sub_att]


# for data in txt_data_lines:
#     # att, sub_att = data.split('::')
#     sub_att = data.split(":")[-1].replace(',', '/')
#     att = sub_att
#
#     att = att.strip().split('_')[-1].replace('_', ' ').lower()
#     sub_att = sub_att.strip().replace('_', ' ').lower().split('(')[0].strip()
#     att_data[att] = att_data.get(att, [])
#     att_data[att] += [sub_att]

att_data = {k: list(set(v)) for k, v in att_data.items()}
all_att = {'att': att_data, 'levels': 2}

json_data = {}
for k, v in all_att.items():
    if k == 'att':
        json_data['l1_attributes'] = list(v.keys())
    else:
        json_data[k] = v

json_data['attribute_tree'] = []
for att_k, att_v in att_data.items():
    if json_data['levels'] == 2:
        sub_atts = []
        for sub_att in att_v:
            while True:
                try:
                    translation = translator.translate(sub_att, src='en', dest='zh-CN')
                    break
                except:
                    pass
            sub_atts.append(sub_att + ',' + translation.text)
    else:
        sub_atts = None
    while True:
        try:
            translation = translator.translate(att_k, src='en', dest='zh-CN')
            break
        except:
            pass
    json_data['attribute_tree'].append({att_k + ',' + translation.text: sub_atts})
    time.sleep(0.1)

json.dump(json_data, open(f'../../attributes/{dataset}/{pattern}.json', 'w'), indent=4, ensure_ascii=False)

