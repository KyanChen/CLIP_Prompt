import json
import copy
import re

import numpy as np
from textblob import TextBlob
from tqdm import tqdm

parent_folder = '../../data/COCO/annotations'
json_file = parent_folder+'/train_2017_caption_tagging.json'
json_data = json.load(open(json_file, 'r'))

proposal_data = json.load(open(parent_folder+'/train_faster_rcnn.proposal.json', 'r'))

proposal_data_dict = {}
for proposal in proposal_data:
    image_id = proposal['image_id']
    box = proposal['bbox']
    bbox = [box[0], box[1], box[0]+box[2], box[1]+box[3], proposal['score'], proposal['category_id']]
    proposal_data_dict[image_id] = proposal_data_dict.get(image_id, []) + [bbox]

for k, v in proposal_data_dict.items():
    v = np.array(v).reshape(-1, 6)
    v_area = (v[:, 2] - v[:, 0]) * (v[:, 3] - v[:, 1])
    ind = np.argmax(v_area)
    biggest_box = v[ind].tolist()
    sorted(v, key=lambda x: x[-2], reverse=True)
    box_sort_by_score = v[:50].tolist()
    json_data[k]['biggest_proposal'] = biggest_box
    json_data[k]['proposals'] = box_sort_by_score

json.dump(json_data, open(parent_folder+'/train_2017_caption_tagging_with_proposals.json', 'w'), indent=4)
