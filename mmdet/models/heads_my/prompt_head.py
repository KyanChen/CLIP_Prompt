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
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm


@HEADS.register_module()
class PromptHead(BaseModule):
    def __init__(self,
                 data_root='',
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 re_weight_alpha=0.2,  # 0.2:68, 0.4:67
                 re_weight_gamma=2,
                 re_weight_beta=0.995,  # 越小，加权越弱
                 balance_unk=0.1,
                 kd_model_loss=None,
                 balance_kd=0.1
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
        self.balance_unk = balance_unk
        self.kd_model_loss = kd_model_loss
        self.balance_kd = balance_kd

    def reweight_att(self, attr_freq):
        pos_rew = torch.from_numpy(np.array([v['pos'] for k, v in attr_freq.items()], dtype=np.float32))
        neg_rew = torch.from_numpy(np.array([v['neg'] for k, v in attr_freq.items()], dtype=np.float32))
        total_rew_bak = torch.from_numpy(np.array([v['total'] for k, v in attr_freq.items()], dtype=np.float32))

        # total_rew = 99 * (total_rew_bak - total_rew_bak.min()) / (total_rew_bak.max() - total_rew_bak.min()) + 1
        # total_rew = 1 - torch.pow(self.re_weight_beta, total_rew)
        # total_rew = (1 - self.re_weight_beta) / total_rew
        # total_rew = 620 * total_rew / total_rew.sum()

        total_rew = 1 / torch.pow(total_rew_bak, self.re_weight_alpha)
        total_rew = 620 * total_rew / total_rew.sum()
        # import pdb
        # pdb.set_trace()
        return total_rew

    def get_classify_loss(self, cls_scores, gt_labels):
        # cls_scores: BxN
        # gt_labels: BxN
        BS = cls_scores.size(0)
        cls_scores_flatten = rearrange(cls_scores, 'B N -> (B N)')
        gt_labels_flatten = rearrange(gt_labels, 'B N -> (B N)')
        gt_labels_flatten = gt_labels_flatten.float()
        total_rew = self.reweight_att_frac.to(gt_labels_flatten.device)
        total_rew = repeat(total_rew, 'N -> (B N)', B=BS)

        pos_mask = gt_labels_flatten == 1
        neg_mask = gt_labels_flatten == 0
        unk_mask = gt_labels_flatten == 2

        # cls_scores_flatten = torch.sigmoid(cls_scores_flatten)
        # pos_pred = torch.clamp(cls_scores_flatten[pos_mask], 1e-10, 1-1e-10)
        # neg_pred = torch.clamp(1-cls_scores_flatten[neg_mask], 1e-10, 1-1e-10)
        # loss_pos = - total_rew[pos_mask] * torch.pow(1-cls_scores_flatten[pos_mask], self.re_weight_gamma) * torch.log(pos_pred)
        # loss_neg = - total_rew[neg_mask] * torch.pow(cls_scores_flatten[neg_mask], self.re_weight_gamma) * torch.log(neg_pred)
        # # loss_pos = - total_rew[pos_mask] * torch.log(pos_pred)
        # # loss_neg = - total_rew[neg_mask] * torch.log(neg_pred)
        # loss_pos = loss_pos.mean()
        # loss_neg = loss_neg.mean()
        loss_pos_neg = F.binary_cross_entropy_with_logits(
            cls_scores_flatten[~unk_mask], gt_labels_flatten[~unk_mask], weight=total_rew[~unk_mask], reduction='mean')

        pred_unk = cls_scores_flatten[unk_mask]
        gt_labels_unk = pred_unk.new_zeros(pred_unk.size())

        # bce_loss_unk = F.binary_cross_entropy(pred_unk, gt_labels_unk, reduction='mean')
        # bce_loss = loss_pos + loss_neg + self.balance_unk * bce_loss_unk

        bce_loss_unk = F.binary_cross_entropy_with_logits(pred_unk, gt_labels_unk, reduction='mean')
        bce_loss = loss_pos_neg + self.balance_unk * bce_loss_unk

        return bce_loss

    def loss(self,
             cls_scores,
             gt_labels,
             img_metas,
             **kwargs
             ):

        loss_s_ce = self.get_classify_loss(cls_scores, gt_labels)

        losses = {}
        losses['loss_s_ce'] = loss_s_ce

        if 'img_crop_features' in kwargs and self.kd_model_loss:
            img_crop_features = kwargs.get('img_crop_features', None)
            proposal_features = kwargs.get('boxes_feats', None)
            kd_logits = kwargs.get('kd_logits', None)

            # img_crop_sigmoid = torch.sigmoid(img_crop_features)
            # proposal_sigmoid = torch.sigmoid(proposal_features)

            # loss_kd = F.kl_div(img_crop_features, proposal_features) + F.kl_div(proposal_features, img_crop_features)
            # loss_kd = F.kl_div(proposal_features, img_crop_features, reduction='mean')

            # similarity = torch.cosine_similarity(img_crop_features, proposal_features, dim=-1)
            # loss = 1 - similarity
            if self.kd_model_loss == 'smooth-l1':
                loss_kd = F.smooth_l1_loss(proposal_features, img_crop_features, reduction='mean')
                loss_kd = self.balance_kd * loss_kd
            elif self.kd_model_loss == 'ce':
                proposal_features = torch.sigmoid(self.balance_kd * proposal_features)
                img_crop_features = torch.sigmoid(self.balance_kd * img_crop_features)
                loss_kd = F.binary_cross_entropy(proposal_features, img_crop_features, reduction='mean')
            elif self.kd_model_loss == 't_ce+ts_ce':


                # gt_labels_flatten = gt_labels.view(-1)
                # kd_logits_flatten = kd_logits.view(-1)
                # cls_scores_flatten = cls_scores.view(-1)
                # unk_mask = gt_labels_flatten == 2

                BS = gt_labels.size()
                total_rew = self.reweight_att_frac.to(gt_labels.device)
                # total_rew = repeat(total_rew, 'N -> (B N)', B=BS)

                loss_t_ce = self.get_classify_loss(kd_logits, gt_labels)
                loss_ts_ce = F.cross_entropy(cls_scores, (kd_logits.detach()).softmax(dim=-1), weight=total_rew)

                losses['loss_t_ce'] = self.balance_kd * 0.5 * loss_t_ce
                losses['loss_ts_ce'] = self.balance_kd * loss_ts_ce
            elif self.kd_model_loss == 't_ce':
                loss_t_ce = self.get_classify_loss(kd_logits, gt_labels)
                losses['loss_t_ce'] = loss_t_ce
            else:
                raise NotImplementedError

        try:
            acc = cal_metrics(f'{self.data_root}/VAW', cls_scores, gt_labels, is_logit=True).float()
            # acc = cal_metrics(f'{self.data_root}/VAW', kd_logits, gt_labels, is_logit=True).float()
        except Exception as e:
            print(e)
            acc = torch.tensor(0., dtype=torch.float32)
        # acc = acc.to(kd_logits.device)
        acc = acc.to(loss_s_ce.device)

        losses['acc'] = acc
        return losses

    def forward_train(self,
                      x,
                      img_metas,
                      gt_labels,
                      **kwargs):
        losses = self.loss(x, gt_labels, img_metas, **kwargs)

        return losses

    def forward(self, feats):
        return feats


@HEADS.register_module()
class TransformerEncoderHead(BaseModule):
    def __init__(self,
                 in_dim=1024,
                 embed_dim=256,
                 use_abs_pos_embed=False,
                 drop_rate=0.1,
                 class_token=False,
                 num_encoder_layers=3,
                 global_pool=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 ):
        super(TransformerEncoderHead, self).__init__(init_cfg)
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.use_abs_pos_embed = use_abs_pos_embed
        self.class_token = class_token
        self.global_pool = global_pool

        if self.class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if self.use_abs_pos_embed:
            self.absolute_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, self.in_channel) * .02)

            self.drop_after_pos = nn.Dropout(p=drop_rate)

        self.proj1 = nn.Linear(in_dim, embed_dim)
        self.transformer_decoder = self.build_transformer_decoder(
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=self.embed_dim * 2,
            drop_rate=drop_rate
        )
        self.proj2 = nn.Linear(embed_dim, in_dim)

    def build_transformer_decoder(
            self, num_encoder_layers=3, dim_feedforward=2048, drop_rate=0.1
    ):
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=dim_feedforward,
            dropout=drop_rate,
            activation='gelu',
            batch_first=True
        )
        encoder_norm = LayerNorm(self.embed_dim)
        encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        return encoder

    def forward(self, x):
        x = self.proj1(x)
        len_x_shape = len(x.shape)
        if len_x_shape == 2:
            x = x.unsqueeze(0)
        B, N, C = x.shape
        x = self.transformer_decoder(x)
        if len_x_shape == 2:
            x = x.squeeze(0)
        x = self.proj2(x)
        return x

