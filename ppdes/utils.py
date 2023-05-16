import numpy as np
import torch
from scipy.stats import norm, pearsonr
from scipy.optimize import brentq
import sigkernel
import matplotlib.pyplot as plt

def r2(x, y):
    return pearsonr(x, y)[0] ** 2

def l2_error(x, y):
    return np.mean((x-y)**2)

def linf_error(x, y):
    return np.max(np.abs(x-y))

def exp_kernel(x, y, sigma):
    return np.exp(-(x-y)**2/(2.*sigma**2))

def psi(t, x, a, xi, eta):
    return xi*np.exp(eta*x-(eta**2/2)*(t**(2*a+1)))

# From https://github.com/ryanmccrickerd/rough_bergomi.
def g(x, a):
    """TBSS kernel applicable to the rBergomi variance process"""
    return x**a

# From https://github.com/ryanmccrickerd/rough_bergomi.
def b(k, a):
    """Optimal discretisation of TBSS process for minimising hybrid scheme error"""
    return ((k**(a+1)-(k-1)**(a+1))/(a+1))**(1/a)

# From https://github.com/ryanmccrickerd/rough_bergomi.
def cov(a, n):
    """Covariance matrix for given alpha and n, assuming kappa = 1 for tractability"""
    cov = np.array([[0.,0.],[0.,0.]])
    cov[0,0] = 1./n
    cov[0,1] = 1./((a+1) * n**(a+1))
    cov[1,1] = 1./((2.*a+1) * n**(2.*a+1))
    cov[1,0] = cov[0,1]
    return cov

# From https://github.com/ryanmccrickerd/rough_bergomi.
def bs(F, K, V, o = 'call'):
    """Returns the Black call price for given forward, strike and integrated variance"""
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    sv = np.sqrt(V)
    d1 = np.log(F/K) / sv + 0.5 * sv
    d2 = d1 - sv
    P = w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)
    return P

# From https://github.com/ryanmccrickerd/rough_bergomi.
def bsinv(P, F, K, t, o = 'call'):
    """Returns implied Black vol from given call price, forward, strike and time to maturity"""
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    P = np.maximum(P, np.maximum(w * (F - K), 0))

    def error(s):
        return bs(F, K, s**2 * t, o) - P
    s = brentq(error, 1e-9, 1e+9)
    return s

def exp_kernel_matrix(x_samples, y_samples, sigma):
    I = x_samples.size  
    J = y_samples.size
    M = np.zeros((I, J))
    for i in range(I):
        for j in range(J):
            M[i,j] = exp_kernel(x_samples[i], y_samples[j], sigma)
    return M

def factor1_matrix(x_samples, y_samples, sigma):
    I = x_samples.size  
    J = y_samples.size
    M = np.zeros((I, J))
    for i in range(I):
        for j in range(J):
            M[i,j] = -(x_samples[i] - y_samples[j])/(sigma**2)
    return M

def factor2_matrix(x_samples, y_samples, sigma):
    I = x_samples.size  
    J = y_samples.size
    M = np.zeros((I, J))
    for i in range(I):
        for j in range(J):
            M[i,j] = -(sigma**2 - (x_samples[i] - y_samples[j])**2)/(sigma**4)
    return M

def psi_matrix(t_grid, s_samples, t_samples, X_samples, a, xi, eta):
    I = s_samples.size  
    J = t_samples.size
    M = np.zeros((I,J))
    for i,s in enumerate(s_samples):
        ind_s = list(t_grid).index(s)
        x = X_samples[i,ind_s,1]
        M[i,:] = psi(s, x, a, xi, eta)
    return M

def sig_kernel_matrices(X_samples, Y_samples, Z_samples, sig_kernel, max_batch, device):
    X_samples = torch.tensor(X_samples, dtype=torch.float64, device=device)
    Y_samples = torch.tensor(Y_samples, dtype=torch.float64, device=device)
    Z_samples = torch.tensor(Z_samples, dtype=torch.float64, device=device)
    M, M_diff, M_diff_diff = sig_kernel.compute_kernel_and_derivatives_Gram(X_samples, Y_samples, Z_samples, max_batch=max_batch)
    return M.cpu().numpy(), M_diff.cpu().numpy(), M_diff_diff.cpu().numpy()

# From https://github.com/ryanmccrickerd/rough_bergomi.
def generate_dW1(a, n_increments, n_samples):
    """1st BM increments with hybrid scheme correlation structure for kappa = 1"""
    return np.random.multivariate_normal(np.array([0,0]), cov(a, n_increments), (n_samples, n_increments))

# From https://github.com/ryanmccrickerd/rough_bergomi.
def generate_dW2(dt, n_increments, n_samples):
    """2nd BM increments"""
    return np.random.randn(n_samples, n_increments)*np.sqrt(dt)

# From https://github.com/ryanmccrickerd/rough_bergomi.
def generate_dB(rho, dW1, dW2):
    """Correllate BM increments"""
    return rho * dW1[:,:,0] + np.sqrt(1.-rho**2) * dW2

# From https://github.com/ryanmccrickerd/rough_bergomi.
def generate_X(a, dW1):
    """Volterra process I"""
    
    n_increments = dW1.shape[1]
    n_samples    = dW1.shape[0]
    
    X1 = np.zeros((n_samples, 1 + n_increments)) 
    X2 = np.zeros((n_samples, 1 + n_increments)) 

    for i in np.arange(1, 1 + n_increments, 1):
        X1[:,i] = dW1[:,i-1,1]

    G = np.zeros(1 + n_increments) 
    for k in np.arange(2, 1 + n_increments, 1):
        G[k] = g(b(k, a)/n_increments, a)

    GX = np.zeros((n_samples, len(dW1[0,:,0]) + len(G) - 1))

    for i in range(n_samples):
        GX[i,:] = np.convolve(G, dW1[i,:,0])

    X2 = GX[:,:1 + n_increments]

    return np.sqrt(2*a + 1.) * (X1 + X2)

def generate_I(t_ind, a, dW1):
    """Shifted Volterra process I """

    n_increments = dW1.shape[1]
    
    dW1_shifted = np.zeros_like(dW1)
    for u_ind in range(n_increments-t_ind):
        dW1_shifted[:,u_ind,:] = dW1[:,u_ind+t_ind,:]

    I_shifted_ = generate_X(a, dW1_shifted)

    I_shifted = np.zeros_like(I_shifted_)
    for s_ind in range(n_increments-t_ind+1):
        I_shifted[:,s_ind+t_ind] = I_shifted_[:,s_ind]
    
    return I_shifted

def generate_xs(xi, x_var, ts):
    return np.array([np.random.uniform(low=-xi*t/2-x_var, high=-xi*t/2+x_var, size=1) for t in ts])

def generate_theta_paths(t_inds, n_increments, T, a):
    t_grid = np.linspace(0, T, n_increments+1)
    dt     = T/n_increments
    paths  = []
    for t_ind in t_inds:
        # Brownian increments
        dW = np.random.normal(loc=0., scale=np.sqrt(dt), size=t_ind)
        # initialise path theta
        path = np.zeros((n_increments+1, 2))
        # set first coordinate = time
        path[:,0] = t_grid
        # set second coordinate = integral kernel against bm 
        t = t_grid[t_ind]
        for (i,s) in zip(range(t_ind, len(t_grid)), t_grid[t_ind:]):
            path[i,1] = np.sqrt(2*a+1.)*np.sum([((s-t_grid[j])**a)*dW[j-1] for j in range(1,t_ind)]) 
        paths.append(path)
    return np.array(paths)

def generate_X_theta_paths(t_inds, n_increments, T, a):
    t_grid = np.linspace(0, T, n_increments+1)
    paths  = []
    for t_ind in t_inds:
        # Brownian increments
        dW_ = generate_dW1(a, n_increments, 1)
        X   = generate_X(a, dW_)[0]
        dW  = dW_[0,:,0]
        # initialise path theta
        path = np.zeros((n_increments+1, 2))
        # set first coordinate = time
        path[:,0] = t_grid
        # set second coordinate I \otimes_t_ind Theta^t_ind
        path[:t_ind,1] = X[:t_ind] 
        for (i,s) in zip(range(t_ind, len(t_grid)), t_grid[t_ind:]):
            path[i,1] = np.sqrt(2*a+1.)*np.sum([((s-t_grid[j])**a)*dW[j-1] for j in range(1,t_ind)]) 
        paths.append(path)   
    return np.array(paths)

def plot_results(mc_prices, sig_prices, m, n, error_fn, error_name):
    r2_score = r2(mc_prices, sig_prices)
    error = error_fn(mc_prices, sig_prices)
    fig, ax = plt.subplots(1, 2, figsize=(16,5))
    ax[0].plot(mc_prices, label='mc_prices')
    ax[0].plot(sig_prices, label='sig_prices')
    ax[0].set_title(f"Interior cpts: {m} --- Boundary cpts: {n} --- {error_name} = {np.round(error, 5)}")
    ax[0].legend()
    ax[1].scatter(mc_prices, sig_prices, label='mc_prices')
    ax[1].set_xlabel('mc_prices') 
    ax[1].set_ylabel('sig_prices')
    ax[1].set_title(f"$R^2$ = {np.round(r2_score, 3)}")
    ax[1].plot([0,1],[0,1])
    plt.tight_layout()
    plt.show()