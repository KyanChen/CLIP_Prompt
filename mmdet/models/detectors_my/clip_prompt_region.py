import json

import torch
from einops import rearrange

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from ..detectors.base import BaseDetector
import warnings


@DETECTORS.register_module()
class CLIP_Prompter_Region(BaseDetector):
    def __init__(self,
                 classname_path,
                 backbone,
                 roi_head,
                 prompt_learner,
                 prompt_learner_weights='',
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CLIP_Prompter_Region, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained

        classname_maps = json.load(open(classname_path))
        classnames = list(classname_maps.keys())

        clip_model = build_backbone(backbone).model
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.text_encoder = build_backbone(
            dict(
                type='TextEncoder',
                clip_model=clip_model
            )
        )

        prompt_learner.update(dict(classnames=classnames, clip_model=clip_model))
        self.prompt_learner = build_backbone(prompt_learner)

        if prompt_learner_weights:
            state_dict = torch.load(prompt_learner_weights, map_location="cpu")
            self.prompt_learner.load_state_dict(state_dict)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        if neck is not None:
            self.neck = build_neck(neck)
        if roi_head is not None:
            self.roi_head = build_head(roi_head)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.named_parameters():
            if 'prompt_learner' in name or 'neck' in name or 'roi_head' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    def extract_feat(self, img):
        return img

    def train(self, mode=True):
        for name, module in self.named_children():
            # import pdb
            # pdb.set_trace()
            if 'prompt_learner' in name or 'neck' in name or 'roi_head' in name:
                module.train(mode)
            else:
                module.eval()

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def train_step(self, data, optimizer):
        losses = self(**data)
        # for loss_name, loss_value in losses.items():
        #     if isinstance(loss_value, torch.LongTensor):
        #         import pdb
        #         pdb.set_trace()
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
                      proposals,
                      gt_labels,
                      **kwargs
                      ):
        with torch.no_grad():
            image_features, img_f_maps = self.image_encoder(img.type(self.dtype))  # 2x1024
        # import pdb
        # pdb.set_trace()

        # img_f_maps
        # torch.Size([256, 64, 112, 112])
        # torch.Size([256, 256, 56, 56])
        # torch.Size([256, 512, 28, 28])
        # torch.Size([256, 1024, 14, 14])
        # torch.Size([256, 2048, 7, 7])

        img_f_maps = tuple([x.float() for x in img_f_maps])
        # img_f_maps = self.neck(img_f_maps)
        proposal_features, bbox_feats = self.roi_head(img_f_maps, proposals)  # proposal_features: torch.Size([256, 1024, 1, 1])
        proposal_features = rearrange(proposal_features, 'B C H W -> B (C H W)')

        prompts = self.prompt_learner()  # 620x77x512
        tokenized_prompts = self.tokenized_prompts

        text_features = self.text_encoder(prompts, tokenized_prompts)  # torch.Size([620, 1024])

        proposal_features = proposal_features / proposal_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.float()

        logit_scale = self.logit_scale.exp().float()
        logits = logit_scale * proposal_features @ text_features.t()  # 2x620

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

    def simple_test(self,
                    img, img_metas, proposals,
                    rescale=False, **kwargs):
        # import pdb
        # pdb.set_trace()
        image_features, img_f_maps = self.image_encoder(img.type(self.dtype))  # 2x1024
        img_f_maps = tuple([x.float() for x in img_f_maps])
        # img_f_maps = self.neck(img_f_maps)
        proposal_features, bbox_feats = self.roi_head(img_f_maps,
                                                      proposals)  # proposal_features: torch.Size([256, 1024, 1, 1])
        proposal_features = rearrange(proposal_features, 'B C H W -> B (C H W)')

        prompts = self.prompt_learner()  # 620x77x512
        tokenized_prompts = self.tokenized_prompts

        text_features = self.text_encoder(prompts, tokenized_prompts)  # torch.Size([620, 1024])

        proposal_features = proposal_features / proposal_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.float()

        logit_scale = self.logit_scale.exp().float()
        logits = logit_scale * proposal_features @ text_features.t()  # 2x620

        pred = list(logits.detach().cpu().numpy())
        return pred


    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

