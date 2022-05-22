import os
import time
SEED = 1
DIR = 'results/tmp'
os.system(f"python train.py --seed ${SEED} --output-dir ${DIR}")
