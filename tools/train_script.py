import os
import time

os.system("CUDA_VISIBLE_DEVICES=4,5,6,7 sh dist_train.sh ../configs_my/maskrcnn_clip_coco_ann.py 4")

