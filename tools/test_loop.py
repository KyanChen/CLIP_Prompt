import os
import time

for i_epoch in range(10, 55, 5):
    os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
              "sh dist_test.sh "
              f"../configs_my/test_clip_config.py "
              f"results/EXP20220913_/epoch_{i_epoch}.pth "
              # f"results/EXP20220916_/epoch_{i_epoch}.pth "
              "8"
              )

