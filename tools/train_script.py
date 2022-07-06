import os
import time

while True:
    os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
              "sh dist_train.sh "
              "../configs_my/CLIPPrompt_Region_VAW.py "
              "results/EXP20220706_1 "
              "8")
    time.sleep(60*2)

