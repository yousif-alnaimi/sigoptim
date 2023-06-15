import numpy as np
from utils import *
from scipy.optimize import minimize, basinhopping

class fBM_MC_pricer(object):
    """
    Class for conditional MC pricer under fBM.
    """
    def __init__(self, n_increments, n_samples_MC, T, a):
        self.n_increments = n_increments
        self.n_samples_MC = n_samples_MC
        self.T            = T 
        self.a            = a
        self.dt           = T/n_increments
        self.t_grid       = np.linspace(0, T, 1+n_increments)[np.newaxis,:]
        self.dW1          = generate_dW1(a, n_increments, n_samples_MC)
    
    def fit_predict(self, t_inds_eval, paths_eval, payoff):
        """MC prices"""
        mc_prices = []
        for t_ind, path in zip(t_inds_eval, paths_eval):
            path = np.repeat(path[:,1][np.newaxis,:], self.n_samples_MC, axis=0)
            log_prices = path[-1,1] + generate_I(t_ind, self.a, self.dW1)[-1]
            mc_prices.append(np.mean([payoff(p) for p in log_prices]))
        return np.array(mc_prices)

class fBM_sigkernel_pricer(object):
    """
    Class for conditional sigkernel pricer under fBM.
    """
    def __init__(self, n_increments, m, n, T, a, sigma_t, sigma_sig, dyadic_order, max_batch, device):
        self.n_increments  = n_increments        
        self.m             = m 
        self.n             = n 
        self.T             = T 
        self.a             = a
        self.dt            = T/n_increments
        self.t_grid        = np.linspace(0, T, 1+n_increments)
        self.sigma_t       = sigma_t
        self.sigma_sig     = sigma_sig
        self.max_batch     = max_batch
        self.device        = device
        self.static_kernel = sigkernel.RBFKernel(sigma=sigma_sig)
        self.sig_kernel    = sigkernel.SigKernel(self.static_kernel, dyadic_order=dyadic_order)
        
    def _generate_ts(self):
        """Generate m interior times uniformly at random on [0,T) and n boundary times = T"""
        self.t_inds_interior = np.random.choice(self.n_increments-1, self.m)
        self.ts_interior     = np.array([self.t_grid[t_ind] for t_ind in self.t_inds_interior])
        self.t_inds_boundary = np.repeat(self.n_increments, self.n)
        self.ts_boundary     = np.repeat(self.T, self.n)
        self.t_inds          = np.concatenate([self.t_inds_interior, self.t_inds_boundary])
        self.ts              = np.concatenate([self.ts_interior, self.ts_boundary])
        
    def _generate_paths(self):
        """Generate m interior paths X \otimes_t \Theta (time-augmented) and n boundary "0" paths"""        
        self.paths_interior = generate_theta_paths(self.t_inds_interior, self.n_increments, self.T, self.a)
        self.paths_boundary = generate_theta_paths(self.t_inds_boundary, self.n_increments, self.T, self.a)
        self.paths = np.concatenate([self.paths_interior, self.paths_boundary], axis=0)
        
    def _generate_directions(self):
        """Generate m paths for directional derivatives"""
        self.directions = np.zeros((self.m, self.n_increments+1, 2))
        for i, (t_ind, t) in enumerate(zip(self.t_inds_interior, self.ts_interior)):
            self.directions[i, t_ind+1:, 1] = [np.sqrt(2*self.a+1)*(s-t)**self.a for s in self.t_grid[t_ind+1:]]

    def mixed_kernel_matrix(self, s, t, p, q):
        """Compute mixed kernel matrix"""
        K_t         = exp_kernel_matrix(s, t, self.sigma_t)
        K_sig, _, _ = sig_kernel_matrices(p, q, q, self.sig_kernel, self.max_batch, self.device)
        return K_t * K_sig

    def _generate_kernel_matrix(self):
        """Generate kernel Gram matrix K"""
        self.K = self.mixed_kernel_matrix(self.ts, self.ts, self.paths, self.paths) 
            
    def _generate_kernel_matrix_constraints(self):
        """Generate kernel matrix K_hat for constraints"""
        K_t_up = exp_kernel_matrix(self.ts_interior, self.ts, self.sigma_t)
        K_sig_up, _, K_sig_diff_diff_up = sig_kernel_matrices(self.paths_interior, self.paths, self.directions, self.sig_kernel, self.max_batch, self.device)
        factor = factor1_matrix(self.ts_interior, self.ts, self.sigma_t)
        
        K_hat_up = factor*K_t_up*K_sig_up + 0.5*K_t_up*K_sig_diff_diff_up
        
        K_hat_down = self.mixed_kernel_matrix(self.ts_boundary, self.ts, self.paths_boundary, self.paths) 
        self.K_hat = np.concatenate([K_hat_up, K_hat_down], axis=0)
        
    def _generate_rhs(self, payoff):
        """Generate right-hand-side of linear system with terminal condition"""
        self.rhs = np.zeros((self.m+self.n,))
        for i in range(self.m,self.m+self.n):
            self.rhs[i] = payoff(self.paths[i,-1,1])
        
    def fit(self, payoff):
        self._generate_ts()
        self._generate_paths()
        self._generate_directions()
        self._generate_kernel_matrix()
        self._generate_kernel_matrix_constraints()
        self._generate_rhs(payoff)

        M_up        = np.concatenate([self.K, self.K_hat.transpose()], axis=1)
        M_down      = np.concatenate([self.K_hat, np.zeros_like(self.K_hat)], axis=1)
        M           = np.concatenate([M_up, M_down], axis=0)
        rhs_        = np.concatenate([np.zeros([self.m+self.n]), self.rhs])
        self.alphas = (np.linalg.pinv(M) @ rhs_)[:self.m+self.n]

    def predict(self, t_inds_eval, paths_eval):
        ts_eval = np.array([self.t_grid[t_ind] for t_ind in t_inds_eval])
        K_eval  = self.mixed_kernel_matrix(ts_eval, self.ts, paths_eval, self.paths)
        return np.matmul(K_eval, self.alphas)
    
