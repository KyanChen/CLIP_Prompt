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
                 re_weight_gamma=2,
                 re_weight_beta=0.995,  # 越小，加权越弱
                 ):
        super(PromptHead, self).__init__(init_cfg)
        self.data_root = data_root
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        attr_freq = json.load(open(data_root + '/VAW/attr_freq_wo_sort.json', 'r'))
        self.re_weight_gamma = re_weight_gamma
        self.re_weight_beta = re_weight_beta
        self.re_weight_alpha = re_weight_alpha
        self.reweight_att_frac = self.reweight_att(attr_freq)

    def reweight_att(self, attr_freq):
        pos_rew = torch.from_numpy(np.array([v['pos'] for k, v in attr_freq.items()], dtype=np.float32))
        neg_rew = torch.from_numpy(np.array([v['neg'] for k, v in attr_freq.items()], dtype=np.float32))
        total_rew_bak = torch.from_numpy(np.array([v['total'] for k, v in attr_freq.items()], dtype=np.float32))

        total_rew = 99 * (total_rew_bak - total_rew_bak.min()) / (total_rew_bak.max() - total_rew_bak.min()) + 1
        total_rew = 1 - torch.pow(self.re_weight_beta, total_rew)
        total_rew = (1 - self.re_weight_beta) / total_rew
        total_rew = 620 * total_rew / total_rew.sum()
        # import pdb
        # pdb.set_trace()
        # total_rew = 1 / torch.pow(total_rew, self.re_weight_alpha)
        # total_rew = 620 * total_rew / total_rew.sum()
        return total_rew


    def get_classify_loss(self, cls_scores, gt_labels):
        # cls_scores: BxN
        # gt_labels: BxN
        BS = cls_scores.size(0)
        cls_scores_flatten = rearrange(cls_scores, 'B N -> (B N)')
        gt_labels_flatten = rearrange(gt_labels, 'B N -> (B N)')
        total_rew = self.reweight_att_frac.to(gt_labels_flatten.device)
        total_rew = repeat(total_rew, 'N -> (B N)', B=BS)

        cls_scores_flatten = torch.sigmoid(cls_scores_flatten)
        pos_mask = gt_labels_flatten == 1
        neg_mask = gt_labels_flatten == 0
        unk_mask = gt_labels_flatten == 2
        pos_pred = torch.clamp(cls_scores_flatten[pos_mask], 1e-10, 1-1e-10)
        neg_pred = torch.clamp(1-cls_scores_flatten[neg_mask], 1e-10, 1-1e-10)
        loss_pos = - total_rew[pos_mask] * torch.pow(1-cls_scores_flatten[pos_mask], self.re_weight_gamma) * torch.log(pos_pred)
        loss_neg = - total_rew[neg_mask] * torch.pow(cls_scores_flatten[neg_mask], self.re_weight_gamma) * torch.log(neg_pred)
        # loss_pos = - total_rew[pos_mask] * torch.log(pos_pred)
        # loss_neg = - total_rew[neg_mask] * torch.log(neg_pred)
        loss_pos = loss_pos.mean()
        loss_neg = loss_neg.mean()

        pred_unk = cls_scores_flatten[unk_mask]
        gt_labels_unk = pred_unk.new_zeros(pred_unk.size())
        bce_loss_unk = F.binary_cross_entropy(pred_unk, gt_labels_unk, reduction='mean')

        bce_loss = loss_pos + loss_neg + 0.2 * bce_loss_unk
        return bce_loss

    def loss(self,
             cls_scores,
             gt_labels,
             img_metas,
             **kwargs
             ):

        # tmp_output = cls_scores.view(-1)
        # tmp_label = gt_labels.view(-1)
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

