import os
import time

for i_epoch in range(10, 60, 10):
    os.system(f"python test.py "
              f"--config ../configs_my/RPN_CLIPPrompt_Region_KD_COCO_VAW.py "
              f"--checkpoint results/EXP20220905_0/epoch_{i_epoch}.pth"
              )

