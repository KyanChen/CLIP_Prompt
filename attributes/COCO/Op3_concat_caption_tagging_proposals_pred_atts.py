import json

# parent_folder = '../../data/COCO/annotations'
import numpy as np

# from mmdet.core import bbox_overlaps

parent_folder = '/Users/kyanchen/Documents/COCO/annotations'
json_file = parent_folder+'/val_2017_caption_tagging_with_proposals.json'
ori_data = json.load(open(json_file, 'r'))

pred_atts = parent_folder+'/pred_proposal_att.json'
pred_atts = json.load(open(pred_atts, 'r'))

flag_id_start = 0
ori_data_tmp = ori_data.copy()
for img_id, data in ori_data_tmp.items():
    img_id_coco = 'coco_' + str(img_id)
    proposals = np.array(data['proposals'])
    flag_id_end = flag_id_start + len(proposals)
    proposals_atts = pred_atts[flag_id_start: flag_id_end]
    proposals_imgids = [x['img_id'] for x in proposals_atts]
    assert set(img_id_coco) == set(proposals_imgids)
    pred_proposals = [x['bbox'] for x in proposals_atts]
    pred_proposals = np.array(pred_proposals)
    iou = bbox_overlaps(proposals, pred_proposals)
    assert np.all(iou[:, np.arange(len(iou))] > 0.99)
    all_atts = np.array([x['pred_att'] for x in proposals_atts])
    proposals = np.concatenate((proposals, all_atts), axis=-1)  # 6 [xywh,conf,class]+ 606
    ori_data[img_id]['proposals'] = proposals.tolist()

json.dump(ori_data, open(parent_folder+'/train_2017_caption_tagging_with_proposals_predatts.json', 'w'), indent=4)



