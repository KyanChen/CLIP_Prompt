import json


data = json.load(open(r"D:\Dataset\COCO\captions_val2017.json", 'r'))
data = [x['caption'] for x in data['annotations']]
json_data = {'num': len(data), 'captions': list(data)}
json.dump(json_data, open('../caption_all/COCO_val_extracted.json', 'w'), indent=4)