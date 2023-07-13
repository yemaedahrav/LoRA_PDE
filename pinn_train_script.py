import os
from tqdm import trange
              
for Re in trange(10,101,1):
    os.system('python pinn_model.py --Re '+str(Re))

Re_in = 100
for Re_out in trange(10,101,1):
    os.system('python pinn_finetuning_model.py --Re_out '+str(Re_out)+' --Re_in '+str(Re_in))