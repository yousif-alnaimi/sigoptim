import signatory
import numpy as np
from itertools import product
import sympy as sym
import torch

from src.free_lie_algebra import wordIter, word2Elt, rightHalfShuffleProduct, dotprod, shuffleProduct


def ES(X, n_,use_gpu=0):
    """
    Calculates the expected signature associated to a bank of paths X, of shape N x l x d.

    :param X:  Bank of paths of shape N x l x d
    :param n_: Order of expected signature to take

    :return:   Expected signature estimate of X

    """
    
    if use_gpu!=None and torch.cuda.is_available():
        device = f"cuda:{use_gpu}"
    elif not torch.cuda.is_available():
        print("CUDA is not available.")
    else:
        device = "cpu"

    X = torch.tensor(X,device=device)

    N, l, d     = X.shape
    n_sig_terms = sum([d**i for i in range(1, n_ + 1)])

    empty_word = [1.]

    if (n_ == 0) or (l in [0, 1]):
        esigX = np.array(empty_word + [np.nextafter(0, 1) for _ in range(n_sig_terms)])
        n_ += (n_ == 0)

    else:
        sigX  = signatory.signature(X, n_)
        esigX = np.array(empty_word + list(torch.mean(sigX, axis=0).cpu()))

    return make_linear_functional(esigX, d, n_)


def make_linear_functional(coeffs, d, n_):
    """
    Turns coefficients into a linear function in the tensor algebra T((R^d)^*).

    :param coeffs:  Coefficients. Output of @make_ell_coeffs function
    :param d:       Dimension (number of assets)
    :param n_:      Order of tensor algebra to truncate up to
    :return:        Linear functional
    """
    linear_functional = 0.*word2Elt('')

    for i, w in enumerate(wordIter(d, n_, asNumbers=True)):
        linear_functional += coeffs[i]*word2Elt(w)

    return linear_functional


def make_ell_coeffs(N, n_, id_=""):
    """
    Makes coefficients of linear functional as symbolic object

    :param N:       Number of dimensions
    :param n_:      Order to approximate up to
    :param id_:     Optional extra identification on the parameter names
    :return:        Coefficients as symbolic expressions
    """
    prod_ = [i+1 for i in range(N)]

    a0    = ["a0" + id_]
    ais   = ['a' + ''.join(map(str, i)) + id_ for j in range(1, n_+1) for i in product(prod_, repeat=j)]

    return sym.symbols(a0 + ais)


def get_weights(ells, sig):
    """
    Performs the calculation w = ((l_1, S(X)), (l_2, S(x)),..., (l_d, S(x)))
    :param ells:    Linear functional l = (l_1, \dots, l_d)
    :param sig:     S(x)
    :return:        As above
    """
    weights = np.array([dotprod(ell, sig) for ell in ells])
    return weights


def get_weights_softmax(ells, sig):
    weights = np.array([1 + dotprod(ell, sig) for ell in ells])
    return weights / weights.sum()


def sum_weights_softmax(ells, sig):
    return sum(get_weights_softmax(ells, sig))

# \sum_{i=1}^N (l_i, S(X))
def sum_weights(ells, sig):
    """
    Sums weights for being passed to portfolio optimisation

    :param ells:    Linear functional l = (l_1, \dots, l_d)
    :param sig:     S(x)
    :return:        \sum_i @get_weights(l_i, S(x))
    """
    return sum(get_weights(ells, sig))


def variance_linear_functional(ell1, ell2, i, j, shift):
    """
    Calculates the quadratic term in the calculation for the variance

    :param ell1:    l_i
    :param ell2:    l_j
    :param i:       Index i
    :param j:       Index j
    :param shift:   Function f which appropriately shifts to the correct index
    :return:        (l_i \preq f(i)) * (l_j \preq f(j))
    """
    # d = get_max_letter(ell1)
    fi = shift(i) #, d)
    fj = shift(j) #, d)
    lpreqi = rightHalfShuffleProduct(ell1, word2Elt(f'{fi}'))
    lpreqj = rightHalfShuffleProduct(ell2, word2Elt(f'{fj}'))

    return shuffleProduct(lpreqi, lpreqj)


def integrate_linear_functional(ell, i, sig, shift):
    """
    Performs the operation (l_i \preq f(i), S(X)) for each l_i.

    :param ell:     Linear functional l_i \in (l_1, \dots, l_d)
    :param i:       Index to integrate
    :param sig:     Signature to integrate against
    :param shift:   Shift function f(i)
    :return:        Integral as described above
    """
    # d      = get_max_letter(ell)
    fi     = shift(i) #, d)
    lpreqi = rightHalfShuffleProduct(ell, word2Elt(f'{fi}'))
    return dotprod(lpreqi, sig)


def portfolio_return(ells, sig, shift):
    """
    Calculates \sum_{i=1}^N (l_i \preq f(i), S(\hat X^LL)), i.e. the portfolio return

    :param ells:    l = (l_1,..., l_d)
    :param sig:     S(x)
    :param shift:   f(i) shift operator
    :return:        Portfolio return
    """
    mu = 0
    for dim, ell in enumerate(ells):
        mu += integrate_linear_functional(ell, dim, sig, shift)

    return mu


def integrate_linear_functional_softmax(ells, ell, i, sig, shift):
    """
    Performs the operation (l_i \preq f(i), S(X)) for each l_i.

    :param ell:     Linear functional l_i \in (l_1, \dots, l_d)
    :param i:       Index to integrate
    :param sig:     Signature to integrate against
    :param shift:   Shift function f(i)
    :return:        Integral as described above
    """
    # d      = get_max_letter(ell)
    fi     = shift(i) #, d)
    lpreqi = rightHalfShuffleProduct(ell, word2Elt(f'{fi}'))
    lpreqi_sig = 1 + dotprod(lpreqi, sig)
    all_lpreqi_sig = sum([1 + dotprod(rightHalfShuffleProduct(ell_, word2Elt(f'{shift(i)}')), sig) for ell_ in ells])
    return lpreqi_sig / all_lpreqi_sig


def portfolio_return_softmax(ells, sig, shift):
    """
    Calculates the portfolio return with softmax truncation

    :param ells:    l = (l_1,..., l_d)
    :param sig:     S(x)
    :param shift:   f(i) shift operator
    :return:        Portfolio return
    """
    mu = 0

    for dim, ell in enumerate(ells):
        mu += integrate_linear_functional_softmax(ells, ell, dim, sig, shift)

    return mu


def variance_linear_functional_softmax(ells, ell1, ell2, i, j, shift, sig):
    """
    Calculates the quadratic term in the calculation for the variance with softmax truncation

    :param ells:    l = (l_1,..., l_d)
    :param ell1:    l_i
    :param ell2:    l_j
    :param i:       Index i
    :param j:       Index j
    :param shift:   Function f which appropriately shifts to the correct index
    :return:        (l_i \preq f(i)) * (l_j \preq f(j))
    """
    # d = get_max_letter(ell1)
    fi = shift(i) #, d)
    fj = shift(j) #, d)
    lpreqi = rightHalfShuffleProduct(ell1, word2Elt(f'{fi}'))
    lpreqj = rightHalfShuffleProduct(ell2, word2Elt(f'{fj}'))

    lpreqi_lpreqj = shuffleProduct(lpreqi, lpreqj)
    li_lj_sig = 1 + dotprod(lpreqi_lpreqj, sig)
    divisor = 0
    for ell in ells:
        elli = rightHalfShuffleProduct(ell, word2Elt(f'{fi}'))
        for ell_ in ells:
            ellj = rightHalfShuffleProduct(ell_, word2Elt(f'{fj}'))
            elli_ellj = shuffleProduct(elli, ellj)
            divisor += 1 + dotprod(elli_ellj, sig)

    return li_lj_sig / divisor

def portfolio_variance_softmax(ells, sig, shift):
    """
    Calculates the portfolio variance associated to the signature trading problem with softmax truncation

    :param ells:        l = (l_1,..., l_d)
    :param sig:         S(x)
    :param shift:       f(i) Shift operator
    :return:            Portfolio variance
    """
    var = 0
    for dim1, ell1 in enumerate(ells):
        for dim2, ell2 in enumerate(ells):
            crossvar = variance_linear_functional_softmax(ells, ell1, ell2, dim1, dim2, shift, sig)
            mean1    = integrate_linear_functional_softmax(ells, ell1, dim1, sig, shift)
            mean2    = integrate_linear_functional_softmax(ells, ell2, dim2, sig, shift)
            var     += crossvar - mean1*mean2
    return var


def portfolio_variance(ells, sig, shift):
    """
    Calculates the portfolio variance associated to the signature trading problem

    :param ells:        l = (l_1,..., l_d)
    :param sig:         S(x)
    :param shift:       f(i) Shift operator
    :return:            Portfolio variance
    """
    var = 0
    for dim1, ell1 in enumerate(ells):
        for dim2, ell2 in enumerate(ells):
            crossvar = dotprod(variance_linear_functional(ell1, ell2, dim1, dim2, shift), sig)
            mean1    = integrate_linear_functional(ell1, dim1, sig, shift)
            mean2    = integrate_linear_functional(ell2, dim2, sig, shift)
            var     += crossvar - mean1*mean2
    return var


def wrapper_factory(N_assets, n_terms):
    """
    # General wrapper required interface with scipy
    :param N_assets:    Number of assets considered
    :param n_terms:     Number of terms in the signature
    :return:            Wrapper for functions above
    """
    def decorator(func):
        def _wrapper(val):
            if n_terms == 1:
                val = np.expand_dims(val, 1)
            else:
                val = val.reshape(N_assets, n_terms)
            return func(val)
        return _wrapper
    return decorator
