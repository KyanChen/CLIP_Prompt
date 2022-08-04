import os
import time

for i in range(90, 100, 10):
    print('epoch: ', i)
    os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
              "sh dist_test.sh "
              "../configs_my/CLIPPrompt_Region_FasterRcnn_KD_VAW.py "
              f"results/EXP20220701/epoch_{i}.pth "
              "8"
    )

