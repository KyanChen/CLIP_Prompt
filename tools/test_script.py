import os
import time

os.system("CUDA_VISIBLE_DEVICES=4,5,6,7 "
          "sh dist_test.sh "
          "../configs_my/OFAPrompt_VAW.py "
          "results/EXP20220523_3/latest.pth "
          "3"
)

