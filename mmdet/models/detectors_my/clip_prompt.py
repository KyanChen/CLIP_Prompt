import json

import torch
from torch import nn

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from ..detectors.base import BaseDetector
import warnings


@DETECTORS.register_module()
class CLIP_Prompter(BaseDetector):
    def __init__(self,
                 attribute_index_file,
                 need_train_names,
                 backbone,
                 prompt_learner,
                 img_encoder=None,
                 prompt_learner_weights='',
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CLIP_Prompter, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        if isinstance(attribute_index_file, dict):
            file = attribute_index_file['file']
            att2id = json.load(open(file, 'r'))
            att_group = attribute_index_file['att_group']
            if 'common2common' in file:
                if att_group in ['common1', 'common2']:
                    self.att2id = att2id[att_group]
                elif att_group == 'all':
                    self.att2id = {}
                    self.att2id.update(att2id['common1'])
                    self.att2id.update(att2id['common2'])
        else:
            self.att2id = json.load(open(attribute_index_file, 'r'))
        atts = list(self.att2id.keys())

        clip_model = build_backbone(backbone).model
        if img_encoder is None:
            self.image_encoder = clip_model.visual
        else:
            self.image_encoder = build_backbone(img_encoder)
            self.img_proj_head = nn.Linear(768, 1024)
        self.logit_scale = clip_model.logit_scale

        self.text_encoder = build_backbone(
            dict(
                type='TextEncoder',
                clip_model=clip_model
            )
        )

        prompt_learner.update(dict(classnames=atts, clip_model=clip_model))
        self.prompt_learner = build_backbone(prompt_learner)

        if prompt_learner_weights:
            state_dict = torch.load(prompt_learner_weights, map_location="cpu")
            self.prompt_learner.load_state_dict(state_dict)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)

        bbox_head['attribute_index_file'] = attribute_index_file
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.need_train_names = need_train_names
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.named_parameters():
            flag = False
            for need_train_name in self.need_train_names:
                if need_train_name in name:
                    flag = True
            param.requires_grad_(flag)

    def extract_feat(self, img):
        return img

    def train(self, mode=True):
        self.training = mode
        for name, module in self.named_children():
            flag = False
            for need_train_name in self.need_train_names:
                if need_train_name in name:
                    flag = True
            if flag:
                module.train(mode)
            else:
                module.eval()
        return self

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer=None):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_labels,
                      gt_bboxes_ignore=None):
        image_features, last_f_map, f_maps = self.image_encoder(img.type(self.dtype))  # 2x1024

        prompts = self.prompt_learner()  # 620x77x512
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)  # 620x1024

        if hasattr(self, 'img_proj_head'):
            image_features = getattr(self, 'img_proj_head')(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()  # 2x620

        losses = self.bbox_head.forward_train(logits, img_metas, gt_labels)

        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    def simple_test(self, img, img_metas, rescale=False):
        image_features, last_f_map, f_maps = self.image_encoder(img.type(self.dtype))  # 2x1024

        prompts = self.prompt_learner()  # 620x77x512
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)  # 620x1024

        if hasattr(self, 'img_proj_head'):
            image_features = getattr(self, 'img_proj_head')(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()  # 2x620

        pred = list(logits.detach().cpu().numpy())
        return pred


    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

