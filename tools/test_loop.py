import os
import time

for i_epoch in range(15, 200, 15):
    os.system(f"python test.py "
              f"--config ../configs_my/CLIPPrompt_Region_VAW.py "
              f"--checkpoint results/EXP20220701_0/epoch_{i_epoch}.pth"
              )

