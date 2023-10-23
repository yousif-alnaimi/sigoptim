import numpy as np
import iisignature


def extract_signature_terms(x):
    """
    Method to extract values from sympy object (representing expected signature)
    :param x:   Expected signature
    :return:    Values of expected signature
    """
    if type(x) == dict:
        return x.values()
    else:
        return x


def get_signature_values(X, n_):
    """
    Extracts expected signature values without passing to linear functional
    :param X:       Path bank, N x l x d
    :param n_:      Order of signature to take
    :return:        Expected signature
    """
    N, l, d = X.shape
    n_sig_terms = sum([d ** i for i in range(1, n_ + 1)])
    empty_word = [1.]
    if (n_ == 0) or (l in [0, 1]):
        esigX = np.array(empty_word + [np.nextafter(0, 1) for _ in range(n_sig_terms)])
        n_ += (n_ == 0)

    else:
        sigX = iisignature.sig(X, n_)
        esigX = np.array(empty_word + list(np.mean(sigX, axis=0)))

    return esigX


def get_signature_weights(ell, sig, N_stocks, order):
    """
    Calculates w = <l, S(x)> where w \in R^d, i.e., returns weights associated to the linear functional l and
    signature S(x).

    :param ell:         Vector of coefficients representing linear functional l \in T((R^d)^*)
    :param sig:         Signature S(x)
    :param N_stocks:    Number of assets
    :param order:       Order of signature taken
    :return:            Weights w = (w_1, \dots, w_d)
    """
    return np.matmul(ell.reshape(N_stocks, order), sig)


def get_sig_weights_time_series(ell, process, order):
    """
    Calculates the stream of weights (w^1, w^2, \dots, w^N) associated to a process S = (S_1, S_2, \dots, S_N).

    Each w^i is given by the function @get_signature_weights.

    :param ell:         Vector of coefficients representing linear functional l \in T((R^d)^*)
    :param process:     Process S = (S_1, S_2, \dots, S_N) where S_i \in R^d.
    :param order:       Order of signature taken
    :return:            Vector of signature weights for each timestep
    """

    l, d = process.shape
    n_terms = int(1 + np.sum([d ** o for o in range(1, order + 1)]))

    empty_word = [1.]
    empty_sig = empty_word + [np.nextafter(0, 1) for _ in range(n_terms - 1)]

    if order == 0:
        sigs = np.array([empty_sig for _ in range(l)])
    else:
        sigs = np.array([
            empty_word + list(iisignature.sig(process[:li], order)) for li in range(1, l)
        ])

        sigs = np.concatenate([[empty_sig], sigs])

    # Calculate weights through time
    weights = np.array([
        get_signature_weights(ell, sig, d, n_terms) for sig in sigs
    ])

    return weights


def shift(i, d, lead_lag=False, add_time=False):
    """
    Shift operator for signature of path, in order to get correct return.
    See the notebook "mean-variance.ipynb" for a detailed explanation.

    :param i:           Index of asset to integrate against
    :param d:           Used if lead-lag is taken
    :param lead_lag:    Optional parameter. If TRUE, shift by d units
    :param add_time:    Optional parameter. If TRUE, shift by 2 units. Assumes time added BEFORE lead-lag
    :return:            Shift in order to get Ito integral when calculating PnL
    """
    res = i+1
    if lead_lag:
        res += d
    if add_time:
        res += 2
    return res


def get_max_letter(ell):
    """
    Gets the max letter of the alphabet associated to the linear functional(s) l

    :param ell:     Linear functional l = (l^1, \dots, l^d)
    :return:        Maximum letter d
    """
    return int(ell.maxLetter())

