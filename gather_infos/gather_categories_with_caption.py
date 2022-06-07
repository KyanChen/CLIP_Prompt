import json

data_caption = json.load(open('infos/all_caption_extracted_categories.json', 'r'))['categories']
data_categories = json.load(open('infos/all_objects_with_freq_filtered.json', 'r'))['categories']
categories = {}

all_data = [data_caption, data_categories]
for data_split in all_data:
    for data in data_split:
        key, value = data
        key = key.lower()
        if len(key) < 2:
            continue
        categories[key] = categories.get(key, 0) + value

categories = sorted(categories.items(), key=lambda kv: kv[1], reverse=True)
return_categories = {'num_categories': len(categories), 'categories': categories}
json.dump(return_categories, open(f'infos/all_gather_categories.json', 'w'), indent=4)

