# testing SDR class
import context

import cvxpy as cp
import numpy as np
import scipy.optimize as op

from datetime import datetime

from bath_fitting.bath_fitting import bath_fitting
from bath_fitting.SDR_fitting import SDRFit


K = 2
nz = 100

rank = 3

z = 1j*np.linspace(1, 10, nz)
L = 9
M = [ np.random.rand(L,L) for _ in range(K)]
Eigs = [ np.linalg.eig(m@m.T) for m in M ]
M_sym = [ X[:,:rank] @ (np.diag(e[:rank]) @ X[:,:rank].T)  for e, X in Eigs ]

poles = np.array([0.1, 1.9])

diff =  1/(np.reshape(z, (-1, 1, 1, 1)) - np.reshape(poles, (1,K, 1, 1)))
M_sym_np = np.array(M_sym).reshape((1, K, L, L))

Delta = np.sum(diff*M_sym_np, axis = 1)

# poles_0 = 2*np.random.rand(K)
poles_0 =  np.array([0.2, 1.8])

Result = bath_fitting(K, z, Delta,
         	          tol = 1.e-4,
              		  opt_tol = 1.e-6,  poles0 = poles_0)

print("Relative error on the reconstruction is %.2g"%(\
			np.linalg.norm(Result[0][0] @ Result[0][0].T - M_sym[0], 'fro')/
			np.linalg.norm( M_sym[0], 'fro')))     

print("Relative error on the reconstruction is %.2g"%(\
			np.linalg.norm(Result[0][1] @ Result[0][1].T - M_sym[1], 'fro')/
			np.linalg.norm( M_sym[1], 'fro')))     