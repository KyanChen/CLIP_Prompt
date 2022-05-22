from mmcv.runner import BaseModule
from ..builder import BACKBONES
from mmcv import ConfigDict
import torch
from typing import Optional

from .ofa import (ofa_tiny_architecture,
                  ofa_medium_architecture,
                  ofa_huge_architecture,
                  ofa_base_architecture,
                  ofa_large_architecture, OFAModel)
from fairseq.data import Dictionary
from fairseq.data.encoders import build_bpe
import json
from utils_my import Trie


@BACKBONES.register_module()
class OFA(BaseModule):
    __arch_func = {
        'ofa_tiny': ofa_tiny_architecture,
        'ofa_medium': ofa_medium_architecture,
        'ofa_huge': ofa_huge_architecture,
        'ofa_base': ofa_base_architecture,
        'ofa_large': ofa_large_architecture
    }

    def __init__(self,
                 ofa_name,
                 task,
                 model_config=dict()
                 ):
        super(OFA).__init__()
        default_args = self.get_default_args()
        cfg = default_args.update(ConfigDict(model_config))
        cfg = self.__arch[ofa_name](cfg)

        self.model = OFAModel.build_model(cfg, task)


    def get_default_args(self):
        args = ConfigDict()
        args.bpe_dir = None
        args.max_source_positions = 1024
        args.max_target_positions = 1024
        args.max_src_length = 128
        args.max_tgt_length = 30
        args.code_dict_size = 8192
        args.patch_image_size = 480
        args.num_bins = 1000
        args.constraint_range = None
        return args

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        patch_images: Optional[torch.Tensor] = None,
        patch_images_2: Optional[torch.Tensor] = None,
        patch_masks: Optional[torch.Tensor] = None,
        code_masks: Optional[torch.Tensor] = None,
        sample_patch_num: Optional[int] = None,
        features_only: bool = False,
        classification_head_name: Optional[str] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        if classification_head_name is not None:
            features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            patch_images=patch_images,
            patch_masks=patch_masks,
            patch_images_2=patch_images_2,
            token_embeddings=token_embeddings,
            return_all_hiddens=return_all_hiddens,
            sample_patch_num=sample_patch_num
        )
        x, extra = self.decoder(
            prev_output_tokens,
            code_masks=code_masks,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        pad = self.encoder.padding_idx
        if classification_head_name is not None:
            prev_lengths = prev_output_tokens.ne(pad).sum(1)
            gather_index = prev_lengths[:, None, None].expand(x.size(0), 1, x.size(2)) - 1
            sentence_representation = x.gather(1, gather_index).squeeze()
            if self.classification_heads[classification_head_name].use_two_images:
                hidden_size = sentence_representation.size(1)
                sentence_representation = sentence_representation.view(-1, hidden_size * 2)
            for k, head in self.classification_heads.items():
                # for torch script only supports iteration
                if k == classification_head_name:
                    x = head(sentence_representation)
                    break

        return x, extra
