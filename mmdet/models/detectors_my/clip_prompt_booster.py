import json

import torch
from mmcv.runner import get_dist_info
from torch import nn

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from ..detectors.base import BaseDetector
import warnings


@DETECTORS.register_module()
class CLIP_Prompt_Booster(BaseDetector):
    def __init__(self,
                 attribute_index_file,
                 need_train_names,
                 backbone,
                 prompt_att_learner=None,
                 prompt_category_learner=None,
                 prompt_phase_learner=None,
                 prompt_caption_learner=None,
                 img_encoder=None,
                 shared_prompt_vectors=False,
                 load_prompt_weights='',
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CLIP_Prompt_Booster, self).__init__(init_cfg)

        self.att2id = {}
        if 'att_file' in attribute_index_file.keys():
            file = attribute_index_file['att_file']
            att2id = json.load(open(file, 'r'))
            att_group = attribute_index_file['att_group']
            if att_group in ['common1', 'common2', 'common', 'rare']:
                self.att2id = att2id[att_group]
            elif att_group == 'common1+common2':
                self.att2id.update(att2id['common1'])
                self.att2id.update(att2id['common2'])
            elif att_group == 'common+rare':
                self.att2id.update(att2id['common'])
                self.att2id.update(att2id['rare'])
            else:
                raise NameError
        self.category2id = {}
        if 'category_file' in attribute_index_file.keys():
            file = attribute_index_file['category_file']
            category2id = json.load(open(file, 'r'))
            att_group = attribute_index_file['category_group']
            if att_group in ['common1', 'common2', 'common', 'rare']:
                self.category2id = category2id[att_group]
            elif att_group == 'common1+common2':
                self.category2id.update(category2id['common1'])
                self.category2id.update(category2id['common2'])
            elif att_group == 'common+rare':
                self.category2id.update(category2id['common'])
                self.category2id.update(category2id['rare'])
            else:
                raise NameError
        self.att2id = {k: v - min(self.att2id.values()) for k, v in self.att2id.items()}
        self.category2id = {k: v - min(self.category2id.values()) for k, v in self.category2id.items()}

        clip_model = build_backbone(backbone).model
        if img_encoder is None:
            self.image_encoder = clip_model.visual
        else:
            self.image_encoder = build_backbone(img_encoder)
            self.img_proj_head = nn.Linear(768, 1024)
        self.logit_scale = nn.Parameter(clip_model.logit_scale.data)

        self.text_encoder = build_backbone(
            dict(
                type='TextEncoder',
                clip_model=clip_model
            )
        )

        if prompt_att_learner is not None:
            assert len(self.att2id)
            prompt_att_learner.update(
                dict(attribute_list=list(self.att2id.keys()), clip_model=clip_model)
            )
            self.prompt_att_learner = build_backbone(prompt_att_learner)

        if prompt_category_learner is not None:
            assert len(self.category2id)
            prompt_category_learner.update(
                dict(attribute_list=list(self.category2id.keys()), clip_model=clip_model)
            )
            if shared_prompt_vectors:
                prompt_category_learner.update(
                    dict(shared_prompt_vectors=self.prompt_att_learner.prompt_vectors)
                )
            self.prompt_category_learner = build_backbone(prompt_category_learner)

        if prompt_phase_learner is not None:
            prompt_phase_learner.update(
                dict(clip_model=clip_model)
            )
            self.prompt_phase_learner = build_backbone(prompt_phase_learner)

        if prompt_caption_learner is not None:
            prompt_caption_learner.update(
                dict(clip_model=clip_model)
            )
            self.prompt_caption_learner = build_backbone(prompt_caption_learner)

        # if load_prompt_weights:
        #     state_dict = torch.load(prompt_learner_weights, map_location="cpu")
        #     self.prompt_learner.load_state_dict(state_dict)

        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)

        bbox_head['attribute_index_file'] = attribute_index_file
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.need_train_names = need_train_names

        rank, world_size = get_dist_info()

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

    def forward_train(
            self,
            img,
            img_metas,
            gt_labels,
            gt_bboxes_ignore=None,
            data_set_type=None,
            **kwargs
    ):
        image_features, last_f_map, f_maps = self.image_encoder(img)  # 2x1024
        # prompts = self.prompt_learner()  # 620x77x512
        # tokenized_prompts = self.tokenized_prompts
        # text_features = self.text_encoder(prompts, tokenized_prompts)  # 620x1024
        text_features = []
        if hasattr(self, 'prompt_att_learner'):
            prompt_context, eot_index, att_group_member_num = self.prompt_att_learner()  # 620x77x512
            text_features_att = self.text_encoder(prompt_context, eot_index)
            text_features.append(text_features_att)

        if hasattr(self, 'prompt_category_learner'):
            prompt_context, eot_index, cate_group_member_num = self.prompt_category_learner()  # 620x77x512
            text_features_cate = self.text_encoder(prompt_context, eot_index)
            text_features.append(text_features_cate)
        text_features = torch.cat(text_features, dim=0)

        if hasattr(self, 'prompt_phase_learner') and hasattr(self, 'prompt_caption_learner'):
            phases = kwargs['phase']
            captions = kwargs['caption']
            num_phase_per_img = [len(x) for x in phases]
            num_caption_per_img = [len(x) for x in captions]
            phases = [t for x in phases for t in x]
            captions = [t for x in captions for t in x]
            phase_context, phase_eot_index, _ = self.prompt_phase_learner(phases, device=img.device)  # 620x77x512
            caption_context, caption_eot_index, _ = self.prompt_caption_learner(captions, device=img.device)
            prompt_context = torch.cat([phase_context, caption_context], dim=0)
            eot_index = torch.cat([phase_eot_index, caption_eot_index], dim=0)
            phase_cap_features = self.text_encoder(prompt_context, eot_index)

        if hasattr(self, 'img_proj_head'):
            image_features = getattr(self, 'img_proj_head')(image_features)
        if hasattr(self, 'text_proj_head'):
            text_features = getattr(self, 'text_proj_head')(text_features)

        logit_scale = self.logit_scale.exp()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if hasattr(self, 'prompt_phase_learner') and hasattr(self, 'prompt_caption_learner'):
            phase_cap_features = phase_cap_features / phase_cap_features.norm(dim=-1, keepdim=True)
            logits_phase_cap = logit_scale * image_features @ phase_cap_features.T
            # logits_per_text = logit_scale * phase_cap_features @ image_features.T
            label_phase_cap = torch.zeros_like(logits_phase_cap, device=img.device)
            flag_set_cursor = 0
            for idx_sample, num_content in enumerate(num_phase_per_img):
                label_phase_cap[idx_sample, flag_set_cursor:flag_set_cursor+num_content] = 1
                flag_set_cursor += num_content
            for idx_sample, num_content in enumerate(num_caption_per_img):
                label_phase_cap[idx_sample, flag_set_cursor:flag_set_cursor + num_content] = 1
                flag_set_cursor += num_content
            assert flag_set_cursor == len(logits_phase_cap.shape[-1]) - 1

        logits = logit_scale * image_features @ text_features.t()  # 2x620
        if hasattr(self, 'prompt_att_learner'):
            att_logit, cate_logit = logits[:, :len(text_features_att)], logits[:, len(text_features_att):]
            split_att_group_logits = att_logit.split(att_group_member_num, dim=-1)
            att_logit = [torch.mean(x, dim=-1, keepdim=True) for x in split_att_group_logits]
            att_logit = torch.cat(att_logit, dim=-1)
            logits = torch.cat((att_logit, cate_logit), dim=-1)

        losses = self.bbox_head.forward_train(logits, img_metas, data_set_type, gt_labels,
                                              logits_phase_cap=logits_phase_cap,
                                              label_phase_cap=label_phase_cap
                                              )

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
        image_features, last_f_map, f_maps = self.image_encoder(img)  # 2x1024
        text_features = []
        if hasattr(self, 'prompt_att_learner'):
            prompt_context, eot_index, att_group_member_num = self.prompt_att_learner()  # 620x77x512
            text_features_att = self.text_encoder(prompt_context, eot_index)
            text_features.append(text_features_att)

        if hasattr(self, 'prompt_category_learner'):
            prompt_context, eot_index, cate_group_member_num = self.prompt_category_learner()  # 620x77x512
            text_features_cate = self.text_encoder(prompt_context, eot_index)
            text_features.append(text_features_cate)
        text_features = torch.cat(text_features, dim=0)

        # prompt_context = self.prompt_learner()  # 620x77x512
        # text_features = self.text_encoder(prompt_context, self.tokenized_prompts)

        if hasattr(self, 'img_proj_head'):
            image_features = getattr(self, 'img_proj_head')(image_features)
        if hasattr(self, 'text_proj_head'):
            text_features = getattr(self, 'text_proj_head')(text_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.logit_scale.exp()
        logit_scale = 1e-1
        logits = logit_scale * image_features @ text_features.t()  # 2x620
        if hasattr(self, 'prompt_att_learner'):
            att_logit, cate_logit = logits[:, :len(text_features_att)], logits[:, len(text_features_att):]
            split_att_group_logits = att_logit.split(att_group_member_num, dim=-1)
            att_logit = [torch.max(x, dim=-1, keepdim=True)[0] for x in split_att_group_logits]
            att_logit = torch.cat(att_logit, dim=-1)
            logits = torch.cat((att_logit, cate_logit), dim=-1)

        pred = list(logits.detach().cpu().numpy())
        return pred


    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

