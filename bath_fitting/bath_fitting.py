import numpy as np
import scipy.optimize as op
from bath_fitting.SDR_fitting import SDRFit


def bath_fitting(K, z, Delta,
                 tol = 1.e-3,
                 opt_tol = 1.e-6,  poles0 = [], 
                 verbose = False):

    # if there is not initial guess randomly create one
    if type(poles0) == list:
        # random initialization of the poles
        poles0 = np.random.rand(K)

    L = Delta.shape[-1]

    # last two dimensions are the same
    assert Delta.shape[1] == Delta.shape[2]
    # number of 
    assert z.shape[0] == Delta.shape[0]

    sdr_fitting = SDRFit(z, Delta, L, K)

    print("defining the optimization function")
    fun = lambda x: sdr_fitting(x, flag='grad')

    print("performing the optimization")
    res = op.minimize(fun,
                   poles0, #method='L-BFGS-B',
                   jac = True,
                   options={'eps': opt_tol, 'disp': True})

    print("optimization done")
    print("final value is %.6f"%(res.value))
    # compute the bath from the optimized poles
    Xbath = sdr_fitting(res.x, flag = 'bath')

    # extracting the V vectors with their corresponding energies
    V = []
    for ii in range(len(Xbath)):
        (Lambda, E) = np.linalg.eig(Xbath[ii])
        ind = Lambda > tol
        V.append(E[:,ind].dot(np.diag(np.sqrt(Lambda[ind]))))
    return (V, res.x)
