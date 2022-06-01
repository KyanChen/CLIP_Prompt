import time
import tqdm
import json
# import joblib

pattern = 'attributes'
# json_data_lines = joblib.load('cocottributes_eccv_version.jbl', 'rb')
# json.dump(json_data_lines['attributes'], open('att_extracted.json', 'w'))

json_data_lines = json.load(open('att_extracted.json', 'r'))
all_attributes = []
for data in tqdm.tqdm(json_data_lines):
    attribute = data['name']
    all_attributes += [attribute]

all_attributes = list(set(all_attributes))
json.dump(all_attributes, open(f'{pattern}_extracted.json', 'w'))


