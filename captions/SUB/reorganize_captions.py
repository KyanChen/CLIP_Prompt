import json

import pandas as pd

data = json.load(open('sbu-captions-all.json', 'r'))

json.dump(list(data['captions']), open('captions_extracted.json', 'w'), indent=4)
pass