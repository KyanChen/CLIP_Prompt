import os
import time

os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
          "sh dist_test.sh "
          "../configs_my/maskrcnn_clip_coco_ann.py "
          "results/EXP20220619_1/latest.pth "
          "8"
)

