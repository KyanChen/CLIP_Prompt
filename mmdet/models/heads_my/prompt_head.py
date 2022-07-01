import warnings
from abc import abstractmethod

import torch
from einops import rearrange
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
                 init_cfg=None
                 ):
        super(PromptHead, self).__init__(init_cfg)
        self.data_root = data_root
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def get_classify_loss(self, cls_scores, gt_labels):
        # cls_scores: BxN
        # gt_labels: BxN
        cls_scores_flatten = rearrange(cls_scores, 'B N -> (B N)')
        gt_labels_flatten = rearrange(gt_labels, 'B N -> (B N)')
        pos_neg_mask = gt_labels_flatten < 2
        bce_loss = F.binary_cross_entropy_with_logits(cls_scores_flatten[pos_neg_mask], gt_labels_flatten[pos_neg_mask], reduction='mean')
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
            acc = cal_metrics(f'{self.data_root}/VAW', cls_scores, gt_labels, is_logit=True)
        except Exception as e:
            print(e)
            acc = torch.tensor(0.)
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

