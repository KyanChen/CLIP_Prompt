import warnings
from abc import abstractmethod

from mmcv.runner import force_fp32

from ..builder import HEADS, build_loss
from mmcv.runner import BaseModule
from mmdet.datasets_my.evaluate_tools import cal_metrics


@HEADS.register_module()
class OFAPromptHead(BaseModule):
    def __init__(self,
                 task,
                 data_root='',
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None
                 ):
        super(OFAPromptHead, self).__init__(init_cfg)
        self.data_root = data_root

        loss_cls.update(task=task)
        self.loss_cls = build_loss(loss_cls)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward_train(self,
                      model,
                      sample,
                      update_num=0,
                      reduce=True,
                      **kwargs):
        loss, sample_size, logging_output = self.loss_cls(model, sample, update_num=update_num)
        losses = {
            "loss": loss
        }
        return losses

    def forward(self, feats):
        return feats

