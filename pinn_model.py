import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_seed = 1
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)


parser = argparse.ArgumentParser(description ='Reynolds Number')
parser.add_argument('--Re', type=int, default=10, help='Reynolds Number')
args = parser.parse_args()
Re = args.Re
nu = 1/Re

f1 = open('pinns/pretrained/training_details/Re_{}.txt'.format(Re), 'a+')
f1.write('Reynolds Number: {}\n'.format(Re))


f = h5py.File('data/boundary_points.h5', 'r')
X_bc = f['X_bc']
Y_bc = f['Y_bc']
X_bc = np.asarray(X_bc)
Y_bc = np.asarray(Y_bc)
f.close()

f = h5py.File('data/Analytical_Solutions/boundary/Re_{}.h5'.format(Re), 'r')
S_bc = f['S_bc']
S_bc = np.asarray(S_bc)
f.close()

X_bc = Variable(torch.from_numpy(X_bc).float(), requires_grad=False).to(device)
Y_bc = Variable(torch.from_numpy(Y_bc).float(), requires_grad=False).to(device)
S_bc = Variable(torch.from_numpy(S_bc).float(), requires_grad=False).to(device)



f = h5py.File('data/colocation_points.h5', 'r')
X_col = f['X_col']
Y_col = f['Y_col']
X_col = np.asarray(X_col)
Y_col = np.asarray(Y_col)
f.close()

f = h5py.File('data/Analytical_Solutions/colocation/Re_{}.h5'.format(Re), 'r')
S_col = f['S_col']
S_col = np.asarray(S_col)
f.close()

X_col = Variable(torch.from_numpy(X_col).float(), requires_grad=True).to(device)
Y_col = Variable(torch.from_numpy(Y_col).float(), requires_grad=True).to(device)
S_col = Variable(torch.from_numpy(S_col).float(), requires_grad=True).to(device)
S = torch.zeros_like(X_col).to(device)


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.layer1 = nn.Linear(2,20)
        self.layer2 = nn.Linear(20,20)
        self.layer3 = nn.Linear(20,20)
        self.layer4 = nn.Linear(20,20)
        self.layer5 = nn.Linear(20,20)
        self.layer6 = nn.Linear(20,20)
        self.output_layer = nn.Linear(20,3)

    def forward(self, x, y): 
        inputs = torch.cat([x, y], axis=1)
        layer1_out = torch.tanh(self.layer1(inputs))
        layer2_out = torch.tanh(self.layer2(layer1_out))
        layer3_out = torch.tanh(self.layer3(layer2_out))
        layer4_out = torch.tanh(self.layer4(layer3_out))
        layer5_out = torch.tanh(self.layer5(layer4_out))
        layer6_out = torch.tanh(self.layer6(layer5_out))
        output = self.output_layer(layer6_out) 
        return output
    
pinn = PINN()
pinn = pinn.to(device)
mse_cost_function = nn.MSELoss() 
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-2)
scheduler = lr_scheduler.StepLR(optimizer, step_size=8000, gamma=0.1, last_epoch=-1, verbose=False)

def residual(x, y, pinn):
    s = pinn.forward(x, y)
    u = s[:,0:1]
    v = s[:,1:2]
    p = s[:,2:]

    du_dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    dv_dx = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    dp_dx = torch.autograd.grad(p.sum(), x, create_graph=True)[0]

    du_dy = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    dv_dy = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    dp_dy = torch.autograd.grad(p.sum(), y, create_graph=True)[0]

    du_dxx = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0]
    dv_dxx = torch.autograd.grad(dv_dx.sum(), x, create_graph=True)[0]
    du_dyy = torch.autograd.grad(du_dy.sum(), y, create_graph=True)[0]
    dv_dyy = torch.autograd.grad(dv_dy.sum(), y, create_graph=True)[0]

    f1 = u*du_dx + v*du_dy + dp_dx - nu*(du_dxx + du_dyy)
    f2 = u*dv_dx + v*dv_dy + dp_dy - nu*(dv_dxx + dv_dyy)
    f3 = du_dx + dv_dy
    f = f1 + f2 + f3
    return f


iterations = 50000
for epoch in range(iterations):
    optimizer.zero_grad()
    
    PINN_S_bc = pinn.forward(X_bc, Y_bc)
    MSE_U = mse_cost_function(PINN_S_bc, S_bc)

    PINN_Residual = residual(X_col, Y_col, pinn)
    MSE_F = mse_cost_function(PINN_Residual, S)

    Loss = MSE_U + MSE_F

    Loss.backward()
    optimizer.step()
    scheduler.step()

    with torch.autograd.no_grad():
        if epoch%1000 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            f1.write('Epoch %d, LR: %.4e, Loss: %.4e, Data Loss: %.4e, Physics Loss: %.4e \n' % (epoch, current_lr, Loss, MSE_U, MSE_F))

S_pinn = pinn(X_col, Y_col)
# Mean L2 Error
error = torch.linalg.norm(S_col-S_pinn,2)
f1.write('Mean L2 Error: %.4e \n' % (error))
# Relative L2 Error
L2_error = torch.linalg.norm(S_col-S_pinn,2)/torch.linalg.norm(S_col,2)
f1.write('Relative L2 Error: %.4e \n' % (L2_error))
f1.close()

# Save weights to file
params = pinn.state_dict()
flattened_weights = []
for key in params.keys():
    flattened_tensor = torch.reshape(params[key], (-1,)).tolist()
    for val in flattened_tensor:
        flattened_weights.append(val)
weights_array = np.array(flattened_weights)
np.savetxt('pinns/pretrained/weights/weights_'+str(Re)+'.txt', weights_array)

torch.save(pinn.state_dict(), 'pinns/pretrained/weights/model_'+str(Re)+'.pt')