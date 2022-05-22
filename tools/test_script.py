import os
import time

os.system("CUDA_VISIBLE_DEVICES=0,2,3 "
          "sh dist_test.sh "
          "../configs_my/CLIPPrompt_VAW.py "
          "results/EXP20220519_1/latest.pth "
          "3"
)

