import numpy as np
from pyDOE import lhs
import h5py

def printname(name):
    print(name)

f = h5py.File('data\\boundary_points.h5', 'r')
#f.visit(printname)
X_bc = f['X_bc']
Y_bc = f['Y_bc']
X_bc = np.asarray(X_bc)
Y_bc = np.asarray(Y_bc)
print("X_bc: ", X_bc.shape)
print("Y_bc: ", Y_bc.shape)
f.close()


f = h5py.File('data\colocation_points.h5', 'r')
#f.visit(printname)
X_col = f['X_col']
Y_col = f['Y_col']
X_col = np.asarray(X_col)
Y_col = np.asarray(Y_col)
print("X_col: ", X_col.shape)
print("Y_col: ", Y_col.shape)
f.close()


f = h5py.File('data\Analytical_Solutions\\boundary\Re_100.h5', 'r')
#f.visit(printname)
P_bc = f['P_bc']
U_bc = f['U_bc']
V_bc = f['V_bc']
S_bc = f['S_bc']
P_bc = np.asarray(P_bc)
U_bc = np.asarray(U_bc)
V_bc = np.asarray(V_bc)
S_bc = np.asarray(S_bc)
print("P_bc: ", P_bc.shape)
print("U_bc: ", U_bc.shape)
print("V_bc: ", V_bc.shape)
print("S_bc: ", S_bc.shape)
f.close()

f = h5py.File('data\Analytical_Solutions\colocation\Re_100.h5', 'r')
#f.visit(printname)
P_col = f['P_col']
U_col = f['U_col']
V_col = f['V_col']
S_col = f['S_col']
P_col = np.asarray(P_col)
U_col = np.asarray(U_col)
V_col = np.asarray(V_col)
S_col = np.asarray(S_col)
print("P_col: ", P_col.shape)
print("U_col: ", U_col.shape)
print("V_col: ", V_col.shape)
print("S_col: ", S_col.shape)
f.close()