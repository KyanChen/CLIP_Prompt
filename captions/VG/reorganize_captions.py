import json

import pandas as pd

data = json.load(open('region_descriptions.json', 'r'))
data = [x['regions'] for x in data]
data = [m['phrase'] for x in data for m in x]
json.dump(list(data)[:1000], open('captions_extracted_split.json', 'w'), indent=4)
pass