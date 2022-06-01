from mmcv.utils import _BatchNorm, _InstanceNorm, build_from_cfg, is_list_of
from mmcv.runner.optimizer import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmcv.runner.optimizer import DefaultOptimizerConstructor


@OPTIMIZER_BUILDERS.register_module()
class SubModelConstructor(DefaultOptimizerConstructor):
    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        sub_model = optimizer_cfg.pop('sub_model', None)
        if hasattr(model, sub_model):
            model = getattr(model, sub_model)

        else:
            raise ModuleNotFoundError(f'{optimizer_cfg["sub_model"]} not in model')

        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            optimizer_cfg['params'] = model.parameters()
            return build_from_cfg(optimizer_cfg, OPTIMIZERS)

        # set param-wise lr and weight decay recursively
        params = []
        self.add_params(params, model)
        optimizer_cfg['params'] = params

        return build_from_cfg(optimizer_cfg, OPTIMIZERS)
