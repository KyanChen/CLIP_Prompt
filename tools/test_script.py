import os
import time

for i in range(20, 150, 20):
    os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
              "sh dist_test.sh "
              # "../configs_my/RPN_CLIPPrompt_Region_KD_COCO_VAW.py "
              # '../configs_my/rpn_r50_fpn_mstrain_vg.py '
              # '../configs_my/rpn_r50_fpn_mstrain_vaw.py '
              # "../configs_my/rpn_r50_fpn_mstrain_coco.py "
              "../configs_my/CLIPPrompt_Crop_Img_VAW.py "
              f"results/EXP20220901_5/epoch_{i}.pth "
              "8"
    )

