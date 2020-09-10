# testing SDR class

import cvxpy as cp
import numpy as np
import scipy.optimize as op

from datetime import datetime

from SDR_fitting import SDRFit
from SDR_fitting import delta_eval

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
%time Result = op.minimize(fun = fun, x0 = poles_0, options= {'xtol': 1e-6, "disp" : True}) 

print("Results with SCS")
print(np.sort(Result.x))
print(poles)


dfundx = lambda x: sdr_fitting(x, flag='grad')[1]

print(op.check_grad(fun, dfundx, poles_0))

f_dfundx = lambda x: sdr_fitting(x, flag='grad')

f, grad2 = f_dfundx(poles_0)

%time Result = op.minimize(f_dfundx, poles_0, jac = True, options= {"disp" : True}) 



# # Result = op.minimize(fun = fun_fro, x0 = poles_0, options= {'xtol': 1e-6, "disp" : True}) 

# # print("Results with Mosek")
# # print(np.sort(Result.x))
# # print(poles)

# X_best = sdr_fitting(Result.x, flag='bath')  

# Delta_approx = delta_eval(Result.x, z, X_best)  

# print(np.sum(np.abs(Delta_approx - Delta))/(nz*L*L))

# # from SDR_fitting import SDRFitComplex

# # sdr_fitting_c = SDRFitComplex(z, Delta, L, K)

# # fun_c = lambda x: sdr_fitting_c(x, flag='value')
# # Result_c = op.minimize(fun = fun_c, x0 = poles_0, options= {"disp" : True}) 

# # print(np.sort(Result_c.x) - np.sort(poles))

# X_list = sdr_fitting(poles_0, flag='bath')  

# Delta_approx = delta_eval(poles_0, z, X_list)        

# val_np = np.sum(np.square(np.abs(Delta_approx - Delta)))         

# # checking that the evaluation is the same in numpy and 
# # cvxpy
# assert np.abs(val_np - fun(poles_0))/val_np < 1.e-6

# X_numpy = np.array(X_list).reshape(1, K, L, L)

# diff_array_np = np.array([1/(np.reshape(z, (-1,)) - pole) for pole in poles_0])

# diff_np_2 =  1/(np.reshape(z, (-1, 1, 1, 1)) - np.reshape(np.array(poles_0), (1, K, 1, 1))) 
# mat = np.square(diff_array_np).reshape(nz, K, 1, 1)

# mat_normal = diff_array_np.reshape(nz, K, 1, 1)

# Delta_approx_np = np.sum(X_numpy*diff_np_2,axis = 1)

# assert np.sum(np.abs(Delta_approx - Delta_approx_np)) < 1.e-12

# error_fit = Delta - Delta_approx
# error_fit = error_fit.reshape(nz, 1, L, L )

# C = X_numpy*mat
# grad = 2*np.sum(np.real(np.conj(C)*error_fit), axis = (0,2,3))

# dfundx = lambda x: sdr_fitting(x, flag='grad')[1]

# print(op.check_grad(fun, dfundx, poles_0))


# f_dfundx = lambda x: sdr_fitting(x, flag='grad')

# f, grad2 = f_dfundx(poles_0)

# Result = op.minimize(f_dfundx, poles_0, jac = True, options= {"disp" : True}) 


