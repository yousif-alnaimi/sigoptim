import numpy as np
import scipy


def mean_variance_optim(expected_returns, covariance_matrix_returns, expected_pnl):

    # initialise weights
    N = expected_returns.size
    w0 = np.ones(N)/N

    # objective
    objective = lambda w: np.matmul(w.T, np.matmul(covariance_matrix_returns, w))

    # constraints
    bnds = tuple(N*[(0., 1.)])

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
            {'type': 'eq', 'fun': lambda w: np.matmul(w.T, expected_returns) - expected_pnl})

    # run optimisation
    optim = scipy.optimize.minimize(fun=objective, x0=w0, bounds=bnds, constraints=cons, tol=1e-4)

    # return optimal weights
    return optim.x