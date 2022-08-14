from .vaw_datasset import VAWDataset
from .nwpu_detection_dataset import NWPUDataset
from .piplines_my import *
from .coco_clip import CocoCLIPDataset
from .coco_clip_annotated import CocoCLIPAnnDataset
from .vaw_rpn_datasset import VAWRPNDataset
from .vaw_proposal_datasset import VAWProposalDataset
from .vaw_region_att_pred_datasset import VAWRegionDataset
from .vaw_crop_img_datasset import VAWCropDataset

from .vg_rpn_datasset import VGRPNDataset
from .coco_rpn_dataset import CocoRPNDataset
from .rpn_attribute_datasset import RPNAttributeDataset

__all__ = [
    'VAWDataset', 'NWPUDataset', 'CocoCLIPDataset', 'CocoCLIPAnnDataset', 'VAWRPNDataset', 'VAWProposalDataset',
    'VAWRegionDataset', 'VAWCropDataset', 'CocoRPNDataset', 'VGRPNDataset', 'RPNAttributeDataset'
]