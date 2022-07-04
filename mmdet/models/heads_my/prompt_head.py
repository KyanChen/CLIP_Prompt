import json
import warnings
from abc import abstractmethod

import numpy as np
import torch
from einops import rearrange, repeat
from mmcv.runner import force_fp32

from ..builder import HEADS, build_loss
from mmcv.runner import BaseModule
from mmdet.datasets_my.evaluate_tools import cal_metrics
import torch.nn.functional as F

@HEADS.register_module()
class PromptHead(BaseModule):
    def __init__(self,
                 data_root='',
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 re_weight_alpha=2,
                 ):
        super(PromptHead, self).__init__(init_cfg)
        self.data_root = data_root
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        attr_freq = json.load(open(data_root + '/VAW/attr_freq_wo_sort.json', 'r'))
        self.re_weight_alpha = re_weight_alpha
        self.reweight_att_frac = self.reweight_att(attr_freq)

    def reweight_att(self, attr_freq):
        pos_rew = torch.from_numpy(np.array([v['pos'] for k, v in attr_freq.items()], dtype=np.float32))
        neg_rew = torch.from_numpy(np.array([v['neg'] for k, v in attr_freq.items()], dtype=np.float32))
        total_rew = torch.from_numpy(np.array([v['total'] for k, v in attr_freq.items()], dtype=np.float32))
        total_rew = torch.pow(1 / total_rew, self.re_weight_alpha)
        total_rew = total_rew / total_rew.sum()
        return total_rew


    def get_classify_loss(self, cls_scores, gt_labels):
        # cls_scores: BxN
        # gt_labels: BxN
        BS = cls_scores.size(0)
        cls_scores_flatten = rearrange(cls_scores, 'B N -> (B N)')
        gt_labels_flatten = rearrange(gt_labels, 'B N -> (B N)')
        total_rew = self.reweight_att_frac.to(gt_labels_flatten.device)
        total_rew = repeat(total_rew, 'N -> (B N)', B=BS)

        pos_neg_mask = gt_labels_flatten < 2
        bce_loss_pos_neg = F.binary_cross_entropy_with_logits(cls_scores_flatten[pos_neg_mask],
                                                              gt_labels_flatten[pos_neg_mask].float(),
                                                              weight=total_rew[pos_neg_mask],
                                                              reduction='mean')
        pred_unk = cls_scores_flatten[~pos_neg_mask]
        gt_labels_unk = pred_unk.new_zeros(pred_unk.size())
        weight_fac = total_rew[~pos_neg_mask]
        bce_loss_unk = F.binary_cross_entropy_with_logits(pred_unk, gt_labels_unk, weight=weight_fac, reduction='mean')

        bce_loss = bce_loss_pos_neg + 0.1 * bce_loss_unk
        return bce_loss

    def loss(self,
             cls_scores,
             gt_labels,
             img_metas,
             **kwargs
             ):

        # tmp_output = cls_scores.view(-1)
        # tmp_label = gt_labels.view(-1)
        # import pdb
        # pdb.set_trace()
        loss = self.get_classify_loss(cls_scores, gt_labels)
        # tmp_mask = (tmp_label >= 0)
        # loss = loss * tmp_mask
        # loss = loss.sum() / tmp_mask.sum()
        # import pdb
        # pdb.set_trace()
        try:
            acc = cal_metrics(f'{self.data_root}/VAW', cls_scores, gt_labels, is_logit=True).float()
        except Exception as e:
            print(e)
            acc = torch.tensor(0., dtype=torch.float32)
        acc = acc.to(loss.device)
        losses = {
            "loss": loss,
            "acc": acc
        }
        return losses

    def forward_train(self,
                      x,
                      img_metas,
                      gt_labels,
                      **kwargs):
        losses = self.loss(x, gt_labels, img_metas)

        return losses

    def forward(self, feats):
        return feats

