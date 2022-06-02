import json

import pandas as pd

data = json.load(open('train.json', 'r'))
data = data['annotations']
data = [x['caption'] for x in data]
json.dump(list(data), open('captions_extracted.json', 'w'), indent=4)
pass