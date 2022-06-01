import torch
from mmcv.runner import BaseModule

from ..builder import BACKBONES

from .clip import _MODELS, _download, build_model


@BACKBONES.register_module()
class CLIPModel(BaseModule):
    def __init__(self,
                 backbone_name='RN50',
                 load_ckpt_from=None,
                 precision='fp16',
                 pretrained=None,
                 init_cfg=None):
        super(CLIPModel, self).__init__(init_cfg)
        if backbone_name not in _MODELS.keys():
            raise KeyError(f'invalid backbone_name {backbone_name} for CLIPModel')

        assert precision in ["fp16", "fp32", "amp"]

        url = _MODELS[backbone_name]
        if load_ckpt_from is None:
            load_ckpt_from = _download(url)
        try:
            # loading JIT archive
            model = torch.jit.load(load_ckpt_from, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(load_ckpt_from, map_location="cpu")

        self.model = build_model(state_dict)

        if precision == "fp32" or precision == "amp":
            # CLIP's default precision is fp16
            self.model = self.model.float()


@BACKBONES.register_module()
class TextEncoder(BaseModule):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding  # 77 512
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)  # 620 77 512

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        # x: 620x1024
        return x

