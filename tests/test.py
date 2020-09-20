# testing SDR class
import context


import cvxpy as cp
import numpy as np
import scipy.optimize as op

from datetime import datetime

from bath_fitting.SDR_fitting import SDRFit
from bath_fitting.SDR_fitting import delta_eval

K = 2
nz = 100

z = 1j*np.linspace(1, 10, nz)
L = 9
M = [ np.random.rand(L,L) for _ in range(K)]
M_sym = [ m @ m.T + np.eye(L) for m in M ]

poles = np.array([0.1, 1.9])

diff =  1/(np.reshape(z, (-1, 1, 1, 1)) - np.reshape(poles, (1,K, 1, 1)))
M_sym_np = np.array(M_sym).reshape((1, K, L, L))

Delta = np.sum(diff*M_sym_np, axis = 1)

sdr_fitting = SDRFit(z, Delta, L, K)

fun = lambda x: sdr_fitting(x, flag='value')

print(fun(poles))

############


# poles_0 = 2*np.random.rand(K)
poles_0 =  np.array([0.2, 1.8])
Result = op.minimize(fun = fun, x0 = poles_0, options= {'tol': 1e-6, "disp" : True}) 

print("Results with SCS")
print(np.sort(Result.x))
print("exact poles")
print(poles)


dfundx = lambda x: sdr_fitting(x, flag='grad')[1]

print(op.check_grad(fun, dfundx, poles_0))

f_dfundx = lambda x: sdr_fitting(x, flag='grad')

f, grad2 = f_dfundx(poles_0)

Result = op.minimize(f_dfundx, poles_0, jac = True, options= {"disp" : True}) 

