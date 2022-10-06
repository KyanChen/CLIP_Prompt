import os
import time

for i_epoch in range(10, 65, 10):
    os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
              "sh dist_test.sh "
              f"../configs_my/CLIPPrompt_Crop_Img_VAW.py "
              f"results/EXP20221005_1/epoch_{i_epoch}.pth "
              # f"results/EXP20220916_/epoch_{i_epoch}.pth "
              "8"
              )

