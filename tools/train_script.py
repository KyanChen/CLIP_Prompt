import os
import time

os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
          "sh dist_train.sh "
          "../configs_my/CLIPPrompt_Crop_Img_VAW.py "
          "results/EXP20220702_3 "
          "8")

