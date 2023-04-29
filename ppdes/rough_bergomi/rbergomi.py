import numpy as np
from utils import *
from scipy.optimize import minimize

class rBergomi(object):
    """
    Class for generating paths of the rBergomi model.
    """
    def __init__(self, n = 100, N = 1000, T = 1.00, a = -0.4):
        """
        Constructor for class.
        """
        # Basic assignments
        self.T = T # Maturity
        self.n = n # Granularity (steps per year)
        self.dt = T/self.n # Step size
        self.s = int(self.n * self.T) # Steps
        self.t = np.linspace(0, self.T, 1 + self.s)[np.newaxis,:] # Time grid
        self.a = a # Alpha
        self.N = N # Paths

        # Construct hybrid scheme correlation structure for kappa = 1
        self.e = np.array([0,0])
        self.c = cov(self.a, self.n)

    def dW1(self):
        """
        Produces random numbers for variance process with required
        covariance structure.
        """
        rng = np.random.multivariate_normal
        return rng(self.e, self.c, (self.N, self.s))

    def Y(self, dW):
        """
        Constructs Volterra process from appropriately
        correlated 2d Brownian increments.
        """
        Y1 = np.zeros((self.N, 1 + self.s)) # Exact integrals
        Y2 = np.zeros((self.N, 1 + self.s)) # Riemann sums

        # Construct Y1 through exact integral
        for i in np.arange(1, 1 + self.s, 1):
            Y1[:,i] = dW[:,i-1,1] # Assumes kappa = 1

        # Construct arrays for convolution
        G = np.zeros(1 + self.s) # Gamma
        for k in np.arange(2, 1 + self.s, 1):
            G[k] = g(b(k, self.a)/self.n, self.a)

        X = dW[:,:,0] # Xi

        # Initialise convolution result, GX
        GX = np.zeros((self.N, len(X[0,:]) + len(G) - 1))

        # Compute convolution, FFT not used for small n
        # Possible to compute for all paths in C-layer?
        for i in range(self.N):
            GX[i,:] = np.convolve(G, X[i,:])

        # Extract appropriate part of convolution
        Y2 = GX[:,:1 + self.s]

        # Finally contruct and return full process
        Y = np.sqrt(2 * self.a + 1) * (Y1 + Y2)
        return Y

    def dW2(self):
        """
        Obtain orthogonal increments.
        """
        return np.random.randn(self.N, self.s) * np.sqrt(self.dt)

    def dB(self, dW1, dW2, rho = 0.0):
        """
        Constructs correlated price Brownian increments, dB.
        """
        self.rho = rho
        dB = rho * dW1[:,:,0] + np.sqrt(1 - rho**2) * dW2
        return dB

    def V(self, Y, xi = 1.0, eta = 1.0):
        """
        rBergomi variance process.
        """
        self.xi = xi
        self.eta = eta
        a = self.a
        t = self.t
        V = xi * np.exp(eta * Y - 0.5 * eta**2 * t**(2 * a + 1))
        return V

    def S(self, V, dB, S0 = 1):
        """
        rBergomi price process.
        """
        self.S0 = S0
        dt = self.dt
        rho = self.rho

        # Construct non-anticipative Riemann increments
        increments = np.sqrt(V[:,:-1]) * dB - 0.5 * V[:,:-1] * dt

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis = 1)

        S = np.zeros_like(V)
        S[:,0] = S0
        S[:,1:] = S0 * np.exp(integral)
        return S

    def S1(self, V, dW1, rho, S0 = 1):
        """
        rBergomi parallel price process.
        """
        dt = self.dt

        # Construct non-anticipative Riemann increments
        increments = rho * np.sqrt(V[:,:-1]) * dW1[:,:,0] - 0.5 * rho**2 * V[:,:-1] * dt

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis = 1)

        S = np.zeros_like(V)
        S[:,0] = S0
        S[:,1:] = S0 * np.exp(integral)
        return S


class rBergomi_MC_pricer(object):
    """
    Class for conditional MC pricer under rough Bergomi.
    """
    def __init__(self, n_increments, n_samples_MC, T, a, xi, eta, rho):
        
        self.n_increments = n_increments
        self.n_samples_MC = n_samples_MC
        
        self.T   = T 
        self.a   = a
        self.xi  = xi
        self.eta = eta
        self.rho = rho 
        self.dt  = T/n_increments
        
        self.t_grid = np.linspace(0, T, 1+n_increments)[np.newaxis,:]
        self.rB     = rBergomi(n=n_increments, N=n_samples_MC, T=T, a=a)
        self.dW1    = self.rB.dW1()
        self.dW2    = self.rB.dW2()
        self.dB     = self.rB.dB(self.dW1, self.dW2, rho=rho)
        
    
    def Y(self, t_ind):
        """Shifted Volterra process"""
        dW1_shifted = np.zeros_like(self.dW1)
        for u_ind in range(self.n_increments-t_ind):
            dW1_shifted[:,u_ind,:] = self.dW1[:,u_ind+t_ind,:]

        Y_shifted = self.rB.Y(dW1_shifted)
        Y         = np.zeros_like(Y_shifted)
        for s_ind in range(self.n_increments-t_ind+1):
            Y[:,s_ind+t_ind] = Y_shifted[:,s_ind]
        
        return Y
    
    def V(self, t_ind, path):
        """Path-dependent Variance process"""
        return self.xi*np.exp(self.eta*(path + self.Y(t_ind)) - .5*self.eta**2*self.t_grid**(2*self.a+1))
        
    def X(self, t_ind, x, path):
        """rBergomi log-prices"""
        V = self.V(t_ind, path)[:,:-1]
        return x + np.cumsum(np.sqrt(V)*self.dB - .5*V*self.dt, axis=1)
    
    def fit_predict(self, t_inds_eval, xs_eval, paths_eval, payoff):
        """MC prices"""
        mc_prices = []
        for t_ind, x, path in zip(t_inds_eval, xs_eval, paths_eval):
            path = np.repeat(path[:,1][np.newaxis,:], self.n_samples_MC, axis=0)
            X = self.X(t_ind, x, path)
            mc_prices.append(np.mean([payoff(x[-1]) for x in X]))
        return np.array(mc_prices)


class rBergomi_sigkernel_pricer(object):
    """
    Class for conditional sigkernel pricer under rough Bergomi.
    """
    def __init__(self, n_increments, x_mean, x_var, m, n, T, a, xi, eta, rho, sigma_t, sigma_x, sigma_sig, dyadic_order, max_batch, device):
        
        self.n_increments = n_increments
        
        self.x_mean = x_mean
        self.x_var  = x_var
        
        self.m = m # collocation points interior
        self.n = n # collocation points boundary
        
        self.T   = T 
        self.a   = a
        self.xi  = xi
        self.eta = eta
        self.rho = rho 
        self.dt  = T/n_increments
        
        self.t_grid       = np.linspace(0, T, 1+n_increments)
        self.sigma_t      = sigma_t
        self.sigma_x      = sigma_x
        self.sigma_sig    = sigma_sig
        self.dyadic_order = dyadic_order 
        self.max_batch    = max_batch
        self.device       = device
        
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
        
    def _generate_xs(self):
        """Generate m+n interior+boundary prices randomly sampled from N(mid_price, 0.1)"""
        self.xs          = np.random.normal(loc=self.x_mean, scale=self.x_var, size=self.m+self.n)
        self.xs_interior = self.xs[:self.m]
        self.xs_boundary = self.xs[self.m:]
        
    def _generate_paths(self):
        """Generate m interior paths as (time-augmented) forward variance curves and n boundary "0" paths"""
        self.paths_interior = generate_evaluation_paths(self.t_inds_interior, self.n_increments, self.T, self.a) 
        self.paths_boundary = np.zeros([self.n, self.n_increments+1, 2])
        for i in range(self.n):
            self.paths_boundary[i,:,0] = self.t_grid
        self.paths = np.concatenate([self.paths_interior, self.paths_boundary], axis=0)
        
    def _generate_directions(self):
        """Generate m paths for directional derivatives"""
        self.directions = np.zeros((self.m, self.n_increments+1, 2))
        for i, (t_ind, t) in enumerate(zip(self.t_inds_interior, self.ts_interior)):
            self.directions[i, t_ind+1:, 1] = [np.sqrt(2*self.a+1)*(s-t)**self.a for s in self.t_grid[t_ind+1:]]
            
    def _generate_kernel_matrix(self):
        """Generate kernel Gram matrix K"""
        self.K = mixed_kernel_matrix(self.ts, self.ts, self.xs, self.xs, self.paths, self.paths, self.sigma_t, self.sigma_x, self.sig_kernel, self.max_batch, self.device)
            
    def _generate_kernel_matrix_constraints(self):
        """Generate kernel matrix K_hat for constraints"""
        self.K_hat_up   = L_kernel_matrix(self.t_grid, self.a, self.xi, self.eta, self.rho, self.ts_interior, self.ts, self.xs_interior, self.xs, self.paths_interior, self.paths, self.directions, self.sigma_t, self.sigma_x, self.sig_kernel, self.max_batch, self.device)
        self.K_hat_down = mixed_kernel_matrix(self.ts_boundary, self.ts, self.xs_boundary, self.xs, self.paths_boundary, self.paths, self.sigma_t, self.sigma_x, self.sig_kernel, self.max_batch, self.device)
        self.K_hat = np.concatenate([self.K_hat_up, self.K_hat_down], axis=0)
        
    def _generate_rhs(self, payoff):
        """Generate right-hand-side of linear system with terminal condition"""
        self.rhs = np.zeros((self.m+self.n,))
        for i in range(self.m,self.m+self.n):
            self.rhs[i] = payoff(self.xs[i])
        
    def fit(self, payoff):
        self._generate_ts()
        self._generate_xs()
        self._generate_paths()
        self._generate_directions()
        self._generate_kernel_matrix()
        self._generate_kernel_matrix_constraints()
        self._generate_rhs(payoff)
        
        # initialise weights
        alpha0 = np.ones(self.m+self.n)

        # objective
        objective = lambda alpha: np.matmul(alpha.T, np.matmul(self.K, alpha))

        # constraints
        cons = [{"type": "eq", "fun": lambda alpha: np.matmul(self.K_hat, alpha) - self.rhs}]

        # run optimisation
        optim = minimize(fun=objective, x0=alpha0, constraints=cons)

        # return optimal weights 
        self.alphas = optim.x
        
    def fit_predict(self, t_inds_eval, xs_eval, paths_eval, payoff):
        self.fit(payoff)
        ts_eval = np.array([self.t_grid[t_ind] for t_ind in t_inds_eval])
        K_eval  = mixed_kernel_matrix(ts_eval, self.ts, xs_eval, self.xs, paths_eval, self.paths, self.sigma_t, self.sigma_x, self.sig_kernel, self.max_batch, self.device)
        return np.matmul(K_eval, self.alphas)
    
    def predict(self, t_inds_eval, xs_eval, paths_eval):
        ts_eval = np.array([self.t_grid[t_ind] for t_ind in t_inds_eval])
        K_eval  = mixed_kernel_matrix(ts_eval, self.ts, xs_eval, self.xs, paths_eval, self.paths, self.sigma_t, self.sigma_x, self.sig_kernel, self.max_batch, self.device)
        return np.matmul(K_eval, self.alphas)
    
