import math
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from pyDOE import lhs
import argparse
import h5py

def printname(name):
    print(name)

f = h5py.File('data\dirichlet_boundary_points.h5', 'r')
#f.visit(printname)
X = f['X']
Y = f['Y']
X = np.asarray(X)
Y = np.asarray(Y)
print("X_bc: ", X.shape)
print("Y_bc: ", Y.shape)
f.close()


f = h5py.File('data\colocation_points.h5', 'r')
#f.visit(printname)
X = f['X']
Y = f['Y']
X = np.asarray(X)
Y = np.asarray(Y)
print("X: ", X.shape)
print("Y: ", Y.shape)
f.close()


f = h5py.File('data\Analytical_Solutions\Re_20.h5', 'r')
#f.visit(printname)
P = f['P']
U = f['U']
V = f['V']
P = np.asarray(P)
U = np.asarray(U)
V = np.asarray(V)
print("P: ", P.shape)
print("U: ", U.shape)
print("V: ", V.shape)
f.close()