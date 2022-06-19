import torch.nn
from mmcv.utils import _BatchNorm, _InstanceNorm, build_from_cfg, is_list_of
from mmcv.runner.optimizer import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmcv.runner.optimizer import DefaultOptimizerConstructor


@OPTIMIZER_BUILDERS.register_module()
class SubModelConstructor(DefaultOptimizerConstructor):
    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        sub_models = optimizer_cfg.pop('sub_model', None)
        if isinstance(sub_models, str):
            sub_models = [sub_models]
        needed_train_sub_models = []
        for sub_model in sub_models:
            if hasattr(model, sub_model):
                sub_model = getattr(model, sub_model)
                needed_train_sub_models.append(sub_model)
            else:
                raise ModuleNotFoundError(f'{optimizer_cfg["sub_model"]} not in model')
        print('All sub models:')
        for name, module in model.named_children():
            print('children module:', name, end=', ')
        print('')
        print('Needed train models:')

        for needed_train_sub_model in needed_train_sub_models:
            print('children module:', repr(needed_train_sub_model))
        print('')
        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            optimizer_cfg['params'] = [needed_train_sub_model.parameters() for needed_train_sub_model in needed_train_sub_models]
            return build_from_cfg(optimizer_cfg, OPTIMIZERS)

        # set param-wise lr and weight decay recursively
        params = []
        self.add_params(params, model)
        optimizer_cfg['params'] = params

        return build_from_cfg(optimizer_cfg, OPTIMIZERS)
