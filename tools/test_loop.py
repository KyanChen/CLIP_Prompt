import os
import time

for i_epoch in range(40, 65, 5):
    os.system(f"python test.py "
              f"--config ../configs_my/CLIPPrompt_Region_VAW.py "
              f"--checkpoint results/EXP20220807_0/epoch_{i_epoch}.pth"
              )

