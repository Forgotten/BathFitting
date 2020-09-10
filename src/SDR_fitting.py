import cvxpy as cp 
import numpy as np
import mosek

def delta_eval(poles, z, X):
    # compute \sum \frac{X_j}{z_i - \lambda_j }
    # output = 
    K = poles.shape[0]
    L = X[0].shape[0]

    assert X[0].shape[1] == L

    diff =  1/(np.reshape(z, (-1, 1, 1, 1)) - np.reshape(poles, (1, K, 1, 1)))
    X_array = np.array(X).reshape((1,K,L,L))

    return np.sum(diff*X_array, axis = 1)


class SDRFit:

    def __init__(self, z, Delta, L, K, solver="SCS"):

        self.K = K
        self.nz = z.shape[0]
        self.L = L
        self.Delta = Delta
        self.z = z

        self.solver = solver

        # creating the parameters for the poles
        self.poles = [ cp.Parameter() for _ in range(self.K)]

        # definin the SDP variables
        self.X = []
        for ii in range(K):
            self.X.append(cp.Variable((L,L), PSD=True))

        # convenient stored matrix for SDP 
        self.diff_param_real = [ cp.Parameter((self.nz)) for _ in range(K)]
        self.diff_param_imag = [ cp.Parameter((self.nz)) for _ in range(K)]

        # compute \sum \frac{X_j}{z_i - \lambda_j }
        sum_poles = []
        for j in range(self.nz):
            partial_sum = self.X[0]*(     self.diff_param_real[0][j]\
                                     + 1j*self.diff_param_imag[0][j])
            for i in range(1, self.K):
                partial_sum += self.X[i]*(     self.diff_param_real[i][j] \
                                          + 1j*self.diff_param_imag[i][j])
            sum_poles.append(partial_sum)

        # compute sum_u [ \sum \frac{X_j}{z_i - \lambda_j } - Delta(z_i) ]^2
        self.loss = 0
        for j in range(self.nz): 
            self.loss += cp.sum_squares( sum_poles[j] - Delta[j,:,:])
            # self.loss += cp.norm(sum_poles[j] - Delta[j,:,:], p = 'fro')

        # setting up the problem 
        self.prob = cp.Problem(cp.Minimize(self.loss))

    def __call__(self, poles, flag = "value", verbose = False):
        assert poles.shape[0] == self.K

        # creating the diff array 
        diff_array = [1/(np.reshape(self.z, (-1,)) - pole) for pole in poles]

        for diff, val in zip(self.diff_param_real, diff_array):
            diff.value = val.real

        for diff, val in zip(self.diff_param_imag, diff_array):
            diff.value = val.imag

        # TODO: add the rest of the 
        if self.solver == "MOSEK":
            mosek_params_dict = {"MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1.e-10,\
                                 "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1.e-10,
                                 "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1.e-10, 
                                 "MSK_DPAR_INTPNT_CO_TOL_NEAR_REL": 1000}
            self.prob.solve(solver = "MOSEK", verbose=verbose,\
                            mosek_params = mosek_params_dict)
        else :
            self.prob.solve(solver = self.solver, 
                            verbose = verbose, 
                            eps = 1.e-11)
    
        if flag == 'value':
            # flag (by default) to return the value of the optimization
            return self.prob.value
        if flag == 'bath':
            # flag to return the bath matrices
            return [ self.X[ii].value for ii in range(self.K)]
        if flag == 'grad':
            # flag to compute the derivative of the function

            # we extract the values of X 
            X_list = [ self.X[ii].value for ii in range(self.K)]
            # We compute the approximation
            Delta_approx = delta_eval(poles, self.z, X_list)

            error_fit = self.Delta - Delta_approx

            X_numpy = np.array(X_list).reshape(1, self.K, self.L, self.L)
            mat = np.square(diff_array).T.reshape(self.nz, self.K, 1, 1)

            C = np.conj(X_numpy*mat)
            error_fit = error_fit.reshape(self.nz, 1, self.L, self.L )
            
            grad = -2*np.sum(np.real(C*error_fit), axis = (0,2,3))

            # we return the value of the function together with the derivative
            return self.prob.value, grad

