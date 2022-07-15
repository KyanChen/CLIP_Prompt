import os
import time

while True:
    os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
              "sh dist_train.sh "
              # "../configs_my/CLIPPrompt_Crop_Img_VAW.py "
              "../configs_my/CLIPPrompt_Region_KD_VAW.py "
              # "../configs_my/CLIPPrompt_Region_VAW.py "
              "results/EXP20220714_3 "
              "8")
    time.sleep(60*2)

