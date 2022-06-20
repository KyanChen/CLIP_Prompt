import os
import time

for i in range(20, 120, 20):
    print('epoch: ', i)
    os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
              "sh dist_test.sh "
              "../configs_my/maskrcnn_clip_coco_ann.py "
              f"results/EXP20220619_0/epoch_{i}.pth "
              "8"
    )

