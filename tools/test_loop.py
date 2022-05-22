import os
import time

for i_epoch in range(130, 160, 10):
    os.system(f"python test.py "
              f"--config ../configs_my/CLIPPrompt_VAW.py "
              f"--checkpoint results/EXP20220518_3/epoch_{i_epoch}.pth"
              )

