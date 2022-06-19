import os
import time

os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 sh dist_train.sh ../configs_my/maskrcnn_clip_coco.py 4")

