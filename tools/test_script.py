import os
import time


os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
          "sh dist_test.sh "
          "../configs_my/rpn_r50_fpn_mstrain_vaw.py "
          # '../configs_my/rpn_r50_fpn_mstrain_vg.py '
          # "../configs_my/rpn_r50_fpn_mstrain_coco.py "
          f"results/EXP20220809_3/latest.pth "
          "8"
)

