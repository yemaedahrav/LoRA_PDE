import os
from tqdm import trange
              
for Re in trange(21,101,1):
    os.system('python pinn_model.py --Re='+str(Re))