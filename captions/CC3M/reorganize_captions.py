import json

import pandas as pd

data = pd.read_csv('Train_GCC-training.tsv', header=None, delimiter='\t')

json.dump(list(data.iloc[:, 0]), open('captions_extracted.json', 'w'), indent=4)
pass