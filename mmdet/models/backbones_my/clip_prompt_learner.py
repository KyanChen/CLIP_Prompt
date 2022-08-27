import json

import numpy as np
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, get_dist_info
from ..builder import BACKBONES
from .clip import tokenize, _Tokenizer

_tokenizer = _Tokenizer()


@BACKBONES.register_module()
class PromptLearner(BaseModule):
    def __init__(self,
                 classnames,
                 clip_model,
                 n_ctx=16,
                 ctx_init='',
                 c_specific=False,
                 class_token_position='end',
                 load_ckpt_from=None
                 ):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if c_specific:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        rank, world_size = get_dist_info()
        if rank == 0:
            print(f'Initial context: "{prompt_prefix}"')
            print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = class_token_position
        if load_ckpt_from is not None:
            state_dict = torch.load(load_ckpt_from, map_location="cpu")['state_dict']
            ctx_data = state_dict['prompt_learner.ctx']
            self.ctx.data.copy_(ctx_data)

    def forward(self):
        import pdb
        pdb.set_trace()
        ctx = self.ctx  # 4x512
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # 620x4x512

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


@BACKBONES.register_module()
class PromptAttributes(BaseModule):
    def __init__(self,
                 attribute_list,
                 clip_model,
                 prompt_config=dict(
                     n_prompt=16,
                     is_att_specific=False,
                     att_position='mid',
                     with_att_type=False,
                     context_length=77,
                     n_prompt_type=8,
                 ),
                 load_ckpt_from=None
                 ):
        super(PromptAttributes, self).__init__()
        self.prompt_config = prompt_config
        n_att = len(attribute_list)
        word_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        n_prompt_vec = prompt_config.get('n_prompt', 16)
        is_att_specific = prompt_config.get('is_att_specific', False)
        with_att_type = prompt_config.get('with_att_type', False)
        n_prompt_type = prompt_config.get('n_prompt_type', None)

        if is_att_specific:
            print("Initializing att-specific contexts")
            prompt_vectors = torch.empty(n_att, n_prompt_vec, word_dim, dtype=torch.float32)
        else:
            prompt_vectors = torch.empty(n_prompt_vec, word_dim, dtype=torch.float32)
            nn.init.normal_(prompt_vectors, std=0.02)
        if n_prompt_type:
            import pdb
            pdb.set_trace()
            assert n_prompt_type == n_prompt_vec
            file = '/data/kyanchen/prompt/data/VAW/att2types.json'
            att2types = json.load(open(file, 'r'))
            id2type = att2types['id2type']
            prompt_vectors = torch.empty(1+len(id2type), n_prompt_vec, word_dim, dtype=torch.float32)
            nn.init.normal_(prompt_vectors, std=0.02)

            att2typeid = att2types['att2typeid']
            self.att_type_id = [att2typeid[attribute] for attribute in attribute_list]

        # prompt_prefix = " ".join(["X"] * n_ctx)
        # print(f'Initial context: "{prompt_prefix}"')
        rank, world_size = get_dist_info()
        if rank == 0:
            print(f"Number of all-shared prompt (tokens): {n_prompt_vec}")
            print(f"Number of type-shared prompt (tokens): {n_prompt_type}")

        self.prompt_vectors = nn.Parameter(prompt_vectors)  # to be optimized

        sot_token = torch.tensor([_tokenizer.encoder["<|startoftext|>"]], dtype=torch.long)
        eot_token = torch.tensor([_tokenizer.encoder["<|endoftext|>"]], dtype=torch.long)
        pad_token = torch.tensor([0], dtype=torch.long)
        if with_att_type:
            file = '/data/kyanchen/prompt/data/VAW/att2types.json'
            att2types = json.load(open(file, 'r'))
            id2type = att2types['id2type']
            att2typeid = att2types['att2typeid']
            att_type_list = [id2type[str(att2typeid[attribute])] for attribute in attribute_list]
            att_type_list = [att_type.replace("_", " ") for att_type in att_type_list]
            type_tokens = [torch.tensor(_tokenizer.encode(att_type)) for att_type in att_type_list]
            self.type_embeddings = [clip_model.token_embedding(x).detach() for x in type_tokens]

        attribute_list = [attribute.replace("_", " ") for attribute in attribute_list]
        attribute_tokens = [torch.tensor(_tokenizer.encode(attribute)) for attribute in attribute_list]
        self.register_buffer('sot_embedding', clip_model.token_embedding(sot_token).detach())
        self.register_buffer('eot_embedding', clip_model.token_embedding(eot_token).detach())
        self.register_buffer('pad_embedding', clip_model.token_embedding(pad_token).detach())
        self.attribute_embeddings = [clip_model.token_embedding(x).detach() for x in attribute_tokens]

        if load_ckpt_from is not None:
            state_dict = torch.load(load_ckpt_from, map_location="cpu")['state_dict']
            ctx_data = state_dict['prompt_learner.ctx']
            self.ctx.data.copy_(ctx_data)

        # prompt_context, eot_index = self.rearrange_context(**prompt_config)
        # self.register_buffer("prompt_context", prompt_context)  # SOS
        # self.register_buffer("eot_index", eot_index)  # CLS, EOS

        # prompts = [prompt_prefix + " " + name + "." for name in classnames]
        # tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        #
        # self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def rearrange_context(
            self,
            context_length=77,
            is_att_specific=False,
            att_position='mid',
            with_att_type=False,
            n_prompt_type=None,
            *args,
            **kwargs
    ):
        import pdb
        pdb.set_trace()
        if with_att_type:
            self.type_embeddings = [x.to(self.prompt_vectors.device) for x in self.type_embeddings]
            assert att_position == 'mid'

        self.attribute_embeddings = [x.to(self.prompt_vectors.device) for x in self.attribute_embeddings]
        rearranged_context = []
        eot_index = []
        for i in range(len(self.attribute_embeddings)):
            rearranged_context_tmp = [self.sot_embedding]

            if is_att_specific:
                prompt_vectors = self.prompt_vectors[i]
            else:
                prompt_vectors = self.prompt_vectors

            if att_position == 'end':
                rearranged_context_tmp.append(prompt_vectors)
                rearranged_context_tmp.append(self.attribute_embeddings[i])
                rearranged_context_tmp.append(self.eot_embedding)
            elif att_position == 'front':
                rearranged_context_tmp.append(self.attribute_embeddings[i])
                rearranged_context_tmp.append(prompt_vectors)
                rearranged_context_tmp.append(self.eot_embedding)
            elif att_position == 'mid':
                if with_att_type:
                    n_part = prompt_vectors.size(1) // 2
                    all_shared_part_1 = prompt_vectors[0, :n_part]
                    all_shared_part_2 = prompt_vectors[0, n_part:]
                    type_shared_vec = prompt_vectors[self.att_type_id[i]+1]
                    type_shared_part_1 = type_shared_vec[0, :n_part]
                    type_shared_part_2 = type_shared_vec[0, n_part:]
                    rearranged_context_tmp.append(all_shared_part_1)
                    rearranged_context_tmp.append(type_shared_part_1)
                    rearranged_context_tmp.append(self.attribute_embeddings[i])
                    rearranged_context_tmp.append(type_shared_part_2)
                    rearranged_context_tmp.append(all_shared_part_2)
                    rearranged_context_tmp.append(self.eot_embedding)
                elif n_prompt_type:
                    n_part = len(prompt_vectors) // 3
                    part_1 = prompt_vectors[:n_part]
                    part_2 = prompt_vectors[n_part:n_part * 2]
                    part_3 = prompt_vectors[n_part * 2:]
                    rearranged_context_tmp.append(part_1)
                    rearranged_context_tmp.append(self.attribute_embeddings[i])
                    rearranged_context_tmp.append(part_2)
                    rearranged_context_tmp.append(self.type_embeddings[i])
                    rearranged_context_tmp.append(part_3)
                    rearranged_context_tmp.append(self.eot_embedding)
                else:
                    n_part = len(prompt_vectors) // 2
                    part_1 = prompt_vectors[:n_part]
                    part_2 = prompt_vectors[n_part:]
                    rearranged_context_tmp.append(part_1)
                    rearranged_context_tmp.append(self.attribute_embeddings[i])
                    rearranged_context_tmp.append(part_2)
                    rearranged_context_tmp.append(self.eot_embedding)
            else:
                raise NotImplementedError
            rearranged_context_tmp = torch.cat(rearranged_context_tmp, dim=0)
            eot_index.append(len(rearranged_context_tmp) - 1)
            rearranged_context_tmp = [rearranged_context_tmp] + \
                                     [self.pad_embedding] * (context_length - len(rearranged_context_tmp))
            rearranged_context_tmp = torch.cat(rearranged_context_tmp, dim=0)
            rearranged_context.append(rearranged_context_tmp)
        return torch.stack(rearranged_context, dim=0), torch.tensor(eot_index, dtype=torch.long)

    def forward(self):
        prompt_context, eot_index = self.rearrange_context(**self.prompt_config)
        return prompt_context, eot_index
