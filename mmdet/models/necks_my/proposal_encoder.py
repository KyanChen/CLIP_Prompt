import torch
from mmcv.cnn import ConvModule
from torch import nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import NECKS, build_neck, build_roi_extractor, build_shared_head
from mmcv.runner import BaseModule


@NECKS.register_module()
class ProposalEncoder(BaseModule):
    def __init__(
            self,
            bbox_roi_extractor=None,
            shared_head=None,
            in_channels=512,
            out_channels=512,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
            init_cfg=None
    ):
        super(ProposalEncoder, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if shared_head is not None:
            shared_head.pretrained = pretrained
            self.shared_head = build_shared_head(shared_head)

        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)

        self.bbox_head = self.init_bbox_head(in_channels, out_channels)

    def init_bbox_head(self, in_channels, out_channels):
        bbox_head = nn.Sequential(
            ConvModule(
                in_channels,
                in_channels,
                3,
                stride=2,
                padding=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            ),
            ConvModule(
                in_channels,
                in_channels*2,
                3,
                stride=2,
                padding=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            ),
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=2),
            nn.BatchNorm2d(in_channels * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1),
        )
        return bbox_head

    def forward(self, x, proposal_list, **kwargs):
        rois = bbox2roi(proposal_list)
        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)  # N 256 7 7
        if hasattr(self, 'shared_head'):
            bbox_feats = self.shared_head(bbox_feats)
        proposal_features = self.bbox_head(bbox_feats)
        return proposal_features, bbox_feats

    def forward_train(self, x, proposal_list, **kwargs):
        return self.forward(x, proposal_list, **kwargs)

    def simple_test(self, x, proposal_list, **kwargs):
        return self.forward(x, proposal_list, **kwargs)

