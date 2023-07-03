import math
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import argparse
import h5py
from tqdm import trange

parser = argparse.ArgumentParser(description ='Colocation Points Generation')
parser.add_argument('--col', type=int, default=50, help='number of colocation data points in each direction')
parser.add_argument('--dbc', type=int, default=50, help='number of boundary data points along each edge')
args = parser.parse_args()

# Hyperparameters and model architecture
N_c = args.col
N_bc = args.dbc
layers = [2, 20, 20, 20, 20, 20, 3]
xmin = -0.5
xmax = 1
ymin = -0.5
ymax = 1.5

# Generate Boundary Condition Points (25 for each edge)
x_data = np.arange(xmin, xmax, (xmax-xmin)/N_bc)
y_data = np.arange(ymin, ymax, (ymax-ymin)/N_bc)
print("x_data: ", x_data.shape)
print("y_data: ", y_data.shape)
# u = np.zeros(shape=(100,1), dtype=float)
# v = np.zeros(shape=(100,1), dtype=float)
# p = np.zeros(shape=(100,1), dtype=float)
# x = np.zeros(shape=(100,1), dtype=float)
# y = np.zeros(shape=(100,1), dtype=float)

x = []
y = []
for i in range(args.dbc):
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
x = np.ndarray.reshape(x, (x.shape[0],1))
y = np.ndarray.reshape(y, (y.shape[0],1))
print("X_bc: ", x.shape)
print("Y_bc: ", y.shape)

# Store the boundary point coordinates in an HDF5 file
filename = 'data\dirichlet_boundary_points.h5'
h5_file = h5py.File(filename, 'w')
h5_file.create_dataset('X_bc', data=x)
h5_file.create_dataset('Y_bc', data=y)
h5_file.close()


# # Generate colocation points (Latin Hypercube Sampling)
# XY_col_points  = lb + (ub-lb)*lhs(2, N_c)
# X_col = np.reshape(XY_col_points[:,0], (-1, 1))
# Y_col = np.reshape(XY_col_points[:,1], (-1, 1))

x = np.arange(xmin, xmax, (xmax-xmin)/N_c)
y = np.arange(ymin, ymax, (ymax-ymin)/N_c)
print("x: ", x.shape)
print("y: ", y.shape)

X, Y = np.meshgrid(x,y)

X = X.flatten()[:,None]
Y = Y.flatten()[:,None]

print("X: ", X.shape)
print("Y: ", Y.shape)

# Store the colocation point coordinates in an HDF5 file
filename = 'data\colocation_points.h5'   
h5_file = h5py.File(filename, 'w')  
h5_file.create_dataset('X', data=X)
h5_file.create_dataset('Y', data=Y)
h5_file.close()

# Store the analytical solutions in an HDF5 file
for Re in trange(20,101,1):
    nu = 1 / Re
    lamb = 1 / (2 * nu) - np.sqrt(1 / (4 * nu ** 2) + 4 * np.pi ** 2)
    Analytical_U = 1-(np.exp(lamb*X))*(np.cos(2*math.pi*Y))
    Analytical_V = (lamb/(2*math.pi))*(np.exp(lamb*X))*(np.sin(2*math.pi*Y))
    Analytical_P = (1/2)*(1-(np.exp(2*lamb*X)))
    Analytical_U = np.reshape(Analytical_U, (-1, 1))
    Analytical_V = np.reshape(Analytical_V, (-1, 1))
    Analytical_P = np.reshape(Analytical_P, (-1, 1))
    filename = 'data\Analytical_Solutions\Re_' + str(Re) + '.h5'
    h5_file = h5py.File(filename, 'w')
    h5_file.create_dataset('U', data=Analytical_U)
    h5_file.create_dataset('V', data=Analytical_V)
    h5_file.create_dataset('P', data=Analytical_P)
    h5_file.close()
