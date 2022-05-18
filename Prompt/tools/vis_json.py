import json
import os


# VAW
dataDir = r'D:\Dataset\VAW\data'

attribute_parent_types = json.load(open(dataDir+'/attribute_parent_types.json'))
attribute_types = json.load(open(dataDir+'/attribute_types.json'))

num_atts = 0
for key, value in attribute_parent_types.items():
    print(f'{key}: {value} {len(value)}')
    for sub_key in value:
        if attribute_types.get(sub_key, None):
            num_atts += len(attribute_types[sub_key])
            print(len(attribute_types[sub_key]), end=' ')
        else:
            print('No: ', sub_key)
    print()
print('Total atts:', num_atts)



attribute_index = json.load(open(dataDir+'/attribute_index.json'))
print(len(list(attribute_index.values())))



