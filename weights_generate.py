import argparse
import os
from tqdm import trange
              
for Re in trange(21,101,1):
    os.system('python model.py --Re='+str(Re))