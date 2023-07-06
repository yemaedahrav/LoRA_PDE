import math
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import argparse
import h5py
from tqdm import trange
from pyDOE import lhs

parser = argparse.ArgumentParser(description ='Colocation Points Generation')
parser.add_argument('--col', type=int, default=2500, help='number of collocation points')
parser.add_argument('--bc', type=int, default=50, help='number of boundary data points along each edge')
args = parser.parse_args()

# Hyperparameters and model architecture
N_c = args.col
N_bc = args.bc

xmin = -0.5
xmax = 1
ymin = -0.5
ymax = 1.5

# Generate Boundary Condition Points
x_data = np.arange(xmin, xmax, (xmax-xmin)/N_bc)
y_data = np.arange(ymin, ymax, (ymax-ymin)/N_bc)
#print("x_data: ", x_data.shape)
#print("y_data: ", y_data.shape)

x = []
y = []
for i in range(N_bc):
    x.append(xmin)
    y.append(y_data[i])
    x.append(xmax)
    y.append(y_data[i])
    x.append(x_data[i])
    y.append(ymin)
    x.append(x_data[i])
    y.append(ymax)

x = np.array(x, dtype=float)
y = np.array(y, dtype=float)
X_bc = np.ndarray.reshape(x, (x.shape[0],1))
Y_bc = np.ndarray.reshape(y, (y.shape[0],1))
print("X_bc: ", X_bc.shape)
print("Y_bc: ", Y_bc.shape)

# Store the boundary point coordinates in an HDF5 file
filename = 'data\\boundary_points.h5'
h5_file = h5py.File(filename, 'w')
h5_file.create_dataset('X_bc', data=X_bc)
h5_file.create_dataset('Y_bc', data=Y_bc)
h5_file.close()



# Generate colocation points (Latin Hypercube Sampling)
X_col = xmin + (xmax-xmin)*lhs(1, N_c)
Y_col = ymin + (ymax-ymin)*lhs(1, N_c)
print("X_col: ", X_col.shape)
print("Y_col: ", Y_col.shape)

# # Generate colocation points (Uniform Mesh Grid)
# x = np.arange(xmin, xmax, (xmax-xmin)/N_c)
# y = np.arange(ymin, ymax, (ymax-ymin)/N_c)
# print("x: ", x.shape)
# print("y: ", y.shape)
# X, Y = np.meshgrid(x,y)
# X = X.flatten()[:,None]
# Y = Y.flatten()[:,None]
# print("X: ", X.shape)
# print("Y: ", Y.shape)

# Store the colocation point coordinates in an HDF5 file
filename = 'data\colocation_points.h5'   
h5_file = h5py.File(filename, 'w')  
h5_file.create_dataset('X_col', data=X_col)
h5_file.create_dataset('Y_col', data=Y_col)
h5_file.close()

# Generate and store the analytical solutions in an HDF5 file (for boundary conditions and collocation points)
for Re in trange(10,101,1):
    nu = 1 / Re
    lamb = 1 / (2 * nu) - np.sqrt(1 / (4 * nu ** 2) + 4 * np.pi ** 2)

    U_col = 1-(np.exp(lamb*X_col))*(np.cos(2*math.pi*Y_col))
    V_col = (lamb/(2*math.pi))*(np.exp(lamb*X_col))*(np.sin(2*math.pi*Y_col))
    P_col = (1/2)*(1-(np.exp(2*lamb*X_col)))
    U_col = np.reshape(U_col, (-1, 1))
    V_col = np.reshape(V_col, (-1, 1))
    P_col = np.reshape(P_col, (-1, 1))
    S_col = np.hstack((U_col, V_col, P_col))
    filename = 'data\Analytical_Solutions\colocation\Re_' + str(Re) + '.h5'
    h5_file = h5py.File(filename, 'w')
    h5_file.create_dataset('U_col', data=U_col)
    h5_file.create_dataset('V_col', data=V_col)
    h5_file.create_dataset('P_col', data=P_col)
    h5_file.create_dataset('S_col', data=S_col)
    h5_file.close()

    U_bc = 1-(np.exp(lamb*X_bc))*(np.cos(2*math.pi*Y_bc))
    V_bc = (lamb/(2*math.pi))*(np.exp(lamb*X_bc))*(np.sin(2*math.pi*Y_bc))
    P_bc = (1/2)*(1-(np.exp(2*lamb*X_bc)))
    U_bc = np.reshape(U_bc, (-1, 1))
    V_bc = np.reshape(V_bc, (-1, 1))
    P_bc = np.reshape(P_bc, (-1, 1))
    S_bc = np.hstack((U_bc, V_bc, P_bc))
    filename = 'data\Analytical_Solutions\\boundary\Re_' + str(Re) + '.h5'
    h5_file = h5py.File(filename, 'w')
    h5_file.create_dataset('U_bc', data=U_bc)
    h5_file.create_dataset('V_bc', data=V_bc)
    h5_file.create_dataset('P_bc', data=P_bc)
    h5_file.create_dataset('S_bc', data=S_bc)
    h5_file.close()


