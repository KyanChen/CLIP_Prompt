import json

import pandas as pd

data = pd.read_csv('results.csv', sep='|').iloc[:, -1]

json.dump(list(data), open('captions_extracted.json', 'w'), indent=4)
pass