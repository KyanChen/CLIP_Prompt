import json

import pandas as pd

data = pd.read_csv('cc12m.tsv', header=None, delimiter='\t')

json.dump(list(data.iloc[:, 1])[:1000], open('captions_extracted_split.json', 'w'), indent=4)
pass