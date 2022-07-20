import torch
from einops import rearrange
from mmcv.cnn import ConvModule
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import NECKS, build_neck, build_roi_extractor, build_shared_head, HEADS, build_head
from mmcv.runner import BaseModule

from ..utils.transformer import PatchMerging


@NECKS.register_module()
class ProposalEncoder(BaseModule):
    def __init__(
            self,
            out_channels=1024,
            bbox_roi_extractor=None,
            shared_head=None,
            attribute_head=None,
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

        # self.bbox_head = self.init_bbox_head(in_channels, out_channels, roi_size)
        self.bbox_head = build_head(attribute_head)
        self.feature_proj = nn.Sequential(
            nn.Linear(self.bbox_head.embed_dim, self.bbox_head.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(self.bbox_head.embed_dim, out_channels)
        )

    def init_bbox_head(self, in_channels, out_channels, roi_size=7):
        if roi_size == 7:
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
                ConvModule(
                    in_channels * 2,
                    in_channels * 4,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')
                ),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels * 4, out_channels, kernel_size=1),
            )
        elif roi_size == 14:
            bbox_head = nn.Sequential(
                ConvModule(
                    in_channels,
                    in_channels*2,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')
                ),
                ConvModule(
                    in_channels*2,
                    in_channels*2,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')
                ),
                ConvModule(
                    in_channels * 2,
                    in_channels * 2,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')
                ),
                ConvModule(
                    in_channels * 2,
                    in_channels * 4,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')
                ),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels * 4, out_channels, kernel_size=1),
            )
        else:
            raise NotImplementedError
        # bbox_head = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Conv2d(in_channels, out_channels, 1)
        # )
        return bbox_head

    def forward(self, x, proposal_list, **kwargs):
        rois = bbox2roi(proposal_list)
        # x 由大到小
        bbox_feats = self.bbox_roi_extractor(x[-self.bbox_roi_extractor.num_inputs:], rois)  # N 256 7 7
        if hasattr(self, 'shared_head'):
            bbox_feats = self.shared_head(bbox_feats)
        proposal_features = self.bbox_head(bbox_feats)
        proposal_features = self.feature_proj(proposal_features)
        return proposal_features, bbox_feats

    def forward_train(self, x, proposal_list, **kwargs):
        return self.forward(x, proposal_list, **kwargs)

    def simple_test(self, x, proposal_list, **kwargs):
        return self.forward(x, proposal_list, **kwargs)


@HEADS.register_module()
class TransformerAttrHead(BaseModule):
    def __init__(self,
                 in_channel=256,
                 embed_dim=512,
                 num_patches=49,
                 use_abs_pos_embed=True,
                 drop_rate=0.,
                 class_token=True,
                 num_encoder_layers=3,
                 global_pool=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 ):
        super(TransformerAttrHead, self).__init__(init_cfg)
        self.embed_dim = embed_dim
        self.use_abs_pos_embed = use_abs_pos_embed
        self.num_patches = num_patches
        self.class_token = class_token
        self.global_pool = global_pool

        if self.class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if self.use_abs_pos_embed:
            embed_len = self.num_patches + 1 if self.cls_token is not None else self.num_patches
            self.absolute_pos_embed = nn.Parameter(torch.randn(1, embed_len, self.embed_dim) * .02)

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if num_patches == 14*14:
            self.patch_merger = PatchMerging(in_channel, self.embed_dim)
        elif num_patches == 7*7:
            self.patch_merger = nn.Conv2d(in_channel, self.embed_dim, kernel_size=1)
        else:
            raise NotImplementedError
        self.transformer_decoder = self.build_transformer_decoder(num_encoder_layers=num_encoder_layers, dim_feedforward=self.embed_dim * 2)

    def build_transformer_decoder(
            self, num_encoder_layers=3, dim_feedforward=2048
    ):
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        encoder_norm = LayerNorm(self.embed_dim)
        encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        return encoder

    def forward(self, x):
        B, C, H, W = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c')

        if self.class_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos_embed is not  None:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        if self.num_patches == 14*14:
            x, down_hw_shape = self.patch_merger(x, (14, 14))
        else:
            x = self.patch_merger(x)

        x = self.transformer_decoder(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        else:
            x = x[:, 0]

        return x

