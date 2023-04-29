import numpy as np
import torch
from scipy.stats import norm, pearsonr
from scipy.optimize import brentq
import sigkernel

def g(x, a):
    """
    TBSS kernel applicable to the rBergomi variance process.
    """
    return x**a

def b(k, a):
    """
    Optimal discretisation of TBSS process for minimising hybrid scheme error.
    """
    return ((k**(a+1)-(k-1)**(a+1))/(a+1))**(1/a)

def cov(a, n):
    """
    Covariance matrix for given alpha and n, assuming kappa = 1 for
    tractability.
    """
    cov = np.array([[0.,0.],[0.,0.]])
    cov[0,0] = 1./n
    cov[0,1] = 1./((1.*a+1) * n**(1.*a+1))
    cov[1,1] = 1./((2.*a+1) * n**(2.*a+1))
    cov[1,0] = cov[0,1]
    return cov

def bs(F, K, V, o = 'call'):
    """
    Returns the Black call price for given forward, strike and integrated
    variance.
    """
    # Set appropriate weight for option token o
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

def bsinv(P, F, K, t, o = 'call'):
    """
    Returns implied Black vol from given call price, forward, strike and time
    to maturity.
    """
    # Set appropriate weight for option token o
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    # Ensure at least instrinsic value
    P = np.maximum(P, np.maximum(w * (F - K), 0))

    def error(s):
        return bs(F, K, s**2 * t, o) - P
    s = brentq(error, 1e-9, 1e+9)
    return s

def r2(x, y):
    return pearsonr(x, y)[0] ** 2

def exp_kernel(x, y, sigma):
    return np.exp(-(x-y)**2/(2.*sigma**2))

def psi(t, x, a, xi, eta):
    return xi*np.exp(eta*np.sqrt(2*a+1)*x-(eta**2/2)*t**(2*a-1))

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

def mixed_kernel_matrix(s_samples, t_samples, x_samples, y_samples, X_samples, Y_samples, sigma_t, sigma_x, sig_kernel, max_batch, device):
    K_t         = exp_kernel_matrix(s_samples, t_samples, sigma_t)
    K_x         = exp_kernel_matrix(x_samples, y_samples, sigma_x)
    K_sig, _, _ = sig_kernel_matrices(X_samples, Y_samples, Y_samples, sig_kernel, max_batch, device)
    return K_t * K_x * K_sig

def L_kernel_matrix(t_grid, a, xi, eta, rho, s_samples, t_samples, x_samples, y_samples, X_samples, Y_samples, Z_samples, sigma_t, sigma_x, sig_kernel, max_batch, device):
    
    K_t   = exp_kernel_matrix(s_samples, t_samples, sigma_t)
    K_x   = exp_kernel_matrix(x_samples, y_samples, sigma_x)

    K_sig, K_sig_diff, K_sig_diff_diff = sig_kernel_matrices(X_samples, Y_samples, Z_samples, sig_kernel, max_batch, device)
    
    M_t   = factor1_matrix(s_samples, t_samples, sigma_t)
    M_x   = factor1_matrix(x_samples, y_samples, sigma_x)
    M_xx  = factor2_matrix(x_samples, y_samples, sigma_x)
    M_psi = psi_matrix(t_grid, s_samples, t_samples, X_samples, a, xi, eta)

    K_mixed = K_t*K_x*K_sig

    A1 = M_t*K_mixed
    A2 = -0.5*M_psi*M_x*K_mixed
    A3 = 0.5*M_psi*M_xx*K_mixed
    A4 = 0.5*K_t*K_x*K_sig_diff_diff
    A5 = rho*np.sqrt(M_psi)*M_x*K_t*K_x*K_sig_diff

    return A1 + A2 + A3 + A4 + A5

def generate_evaluation_paths(t_inds, n_increments, T, a):
    
    # time grid
    t_grid = np.linspace(0, T, n_increments+1)

    # time step
    dt = T/n_increments
    
    paths = []
    for t_ind in t_inds:
        # Brownian increments
        dW = np.random.normal(loc=0., scale=np.sqrt(dt), size=t_ind)

        # initialise path theta
        path = np.zeros((n_increments+1, 2))

        # set first coordinate = time
        path[:,0] = t_grid

        # set second coordinate = integral kernel against bm 
        t = t_grid[t_ind]
        for (i,s) in zip(range(t_ind+1, len(t_grid)), t_grid[t_ind+1:]):
            path[i,1] = np.sqrt(2*a+1)*np.sum([((s-t_grid[j])**a)*dW[j-1] for j in range(1,t_ind)]) 
        paths.append(path)
        
    return np.array(paths)