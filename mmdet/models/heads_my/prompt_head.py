import warnings
from abc import abstractmethod

import torch
from mmcv.runner import force_fp32

from ..builder import HEADS, build_loss
from mmcv.runner import BaseModule
from mmdet.datasets_my.evaluate_tools import cal_metrics


@HEADS.register_module()
class PromptHead(BaseModule):
    def __init__(self,
                 data_root='',
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None
                 ):
        super(PromptHead, self).__init__(init_cfg)
        self.data_root = data_root
        self.loss_cls = build_loss(loss_cls)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss(self,
             cls_scores,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        # tmp_output = cls_scores.view(-1)
        # tmp_label = gt_labels.view(-1)
        # import pdb
        # pdb.set_trace()
        loss = self.loss_cls(cls_scores, gt_labels)
        # tmp_mask = (tmp_label >= 0)
        # loss = loss * tmp_mask
        # loss = loss.sum() / tmp_mask.sum()
        import pdb
        pdb.set_trace()
        try:
            acc = cal_metrics(f'{self.data_root}/VAW', cls_scores, gt_labels, is_logit=True)
        except Exception as e:
            print(e)
            acc = torch.tensor(0.)
        losses = {
            "loss": loss,
            "acc": acc
        }
        return losses

    def forward_train(self,
                      x,
                      img_metas,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        losses = self.loss(x, gt_labels, img_metas, gt_bboxes_ignore=gt_bboxes_ignore)

        return losses

    def forward(self, feats):
        return feats

