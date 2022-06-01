import os
import time
for i_epoch in range(20, 110, 20):
    os.system(f"python test.py --load-epoch {i_epoch}")
