import json
import os.path

file_path = '../../../attributes/OVAD/ovad1200_licence.json'
save_path = '../../../attributes/OVAD'
data = json.load(open(file_path))['categories']
common2common = {'common1': {}, 'common2': {}}
for idx, item in enumerate(data):
    if item["ov_set"] == "base":
        common2common['common1'].update({item['name']: idx})
    elif item["ov_set"] == "novel":
        common2common['common2'].update({item['name']: idx})
common2common['common1_len'] = len(common2common['common1'])
common2common['common2_len'] = len(common2common['common2'])

json.dump(common2common, open(save_path+'/common2common_category2id.json', 'w'), indent=4)


data = json.load(open(file_path))['attributes']
common2common = {'common1': {}, 'common2': {}}
for idx, item in enumerate(data):
    common2common['common1'].update({item["name"]: idx})
common2common['common1_len'] = len(common2common['common1'])
common2common['common2_len'] = len(common2common['common2'])

json.dump(common2common, open(save_path+'/common2common_att2id.json', 'w'), indent=4)
