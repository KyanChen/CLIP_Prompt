import os
import time

for i in range(120, 165, 15):
    print('epoch: ', i)
    os.system("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
              "sh dist_test.sh "
              "../configs_my/CLIPPrompt_Proposal_VAW.py "
              f"results/EXP20220627_1/epoch_{i}.pth "
              "8"
    )

