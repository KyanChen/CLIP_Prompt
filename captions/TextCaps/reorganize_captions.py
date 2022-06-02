import json

import pandas as pd

data = json.load(open('TextCaps_0.1_val.json', 'r'))
data = data['data']
data = [t for x in data for t in x['reference_strs']]
json.dump(list(data), open('captions_extracted.json', 'w'), indent=4)
pass