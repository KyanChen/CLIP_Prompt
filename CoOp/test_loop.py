import os
import time
for i_epoch in range(60, 110, 15):
    os.system(f"python test.py --load-epoch {i_epoch}")
