from collections import OrderedDict

import numpy as np
import sympy as sym
import mogptk
import torch
from .transformations import *

from .free_lie_algebra import *
from .optimisation.signature import *
from .helper_functions.signature_helper_functions import get_signature_weights, shift, get_signature_values

# The signature trading part is from Zacharia Issa. Please see the `mean-variacne.ipynb` file for detailed explanation.
# This module merges the code into several classes and also integrate the Gaussian Process with Zach's code.

# Shift operator
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
    res = i + 1
    if lead_lag:
        res += d
    if add_time:
        res += 2
    return res

def _compose(*functions, **kwargs):
    if functions:
        def composed_function(x):
            result = x
            for i, (func, func_kwargs) in enumerate(zip(functions, kwargs.values())):
                result = func(result, **func_kwargs)
            return result
        return composed_function
    else:
        return lambda x: x

class _transformation:
    def __init__(self,dim,dt=None,_transformations=OrderedDict(
                {"AddTime": False, "TranslatePaths": False, "ScalePaths": False, "LeadLag": False, "HoffLeadLag":False})):
        if dt and _transformations['ScalePaths'] == True:
            lambda_ = [dt**-0.5 for _ in range(dim)]
        elif dt == None:
            raise KeyboardInterrupt('Please Specify dt')
        else:
            lambda_ = 1
        
        _transformation_args = OrderedDict(
            {"AddTime": {}, "TranslatePaths": {}, "ScalePaths": {"lambda_":lambda_}, "LeadLag": {}, "HoffLeadLag": {}}
        )

        filtered_functions = [globals()[func_name] for func_name, value in _transformations.items() if value]
        filtered_transformations = {k : v for k, v in _transformation_args.items() if _transformations[k]}

        self.transformations = _compose(*filtered_functions, **filtered_transformations)

        self.f = lambda x: shift(x, dim, int(_transformations["LeadLag"] + _transformations["HoffLeadLag"]), int(_transformations["AddTime"]))

class _pre_get_funcs:
    def __init__(self,dim,n_terms,ell_coeffs):
        self.dim = dim
        self.n_terms = n_terms
        self.ell_coeffs = ell_coeffs

    # Set the various functions used in the optimisation
    def set_signature_variance_func(self,ells, esig, shift):
        # Lambdify optimising functional
        variance_polynomial_ = portfolio_variance(ells, esig, shift)
        variance_polynomial  = sym.lambdify([self.ell_coeffs], variance_polynomial_)

        @wrapper_factory(N_assets=self.dim, n_terms=self.n_terms)
        def variance_function(a):
            return variance_polynomial(a)

        return variance_function

    def set_signature_return_func(self,ells, esig, shift):

        return_polynomial_ = portfolio_return(ells, esig, shift)
        return_polynomial  = sym.lambdify([self.ell_coeffs], return_polynomial_)

        @wrapper_factory(N_assets=self.dim, n_terms=self.n_terms)
        def return_function(a):
            return return_polynomial(a)

        return return_function

    def set_signature_weight_sum_func(self,ells, esig):

        weight_sum_polynomial_= sum_weights(ells, esig)
        weight_sum_polynomial = sym.lambdify([self.ell_coeffs], weight_sum_polynomial_)

        @wrapper_factory(N_assets=self.dim, n_terms=self.n_terms)
        def weight_sum_function(a):
            return weight_sum_polynomial(a)

        return weight_sum_function

    def set_signature_individual_weight_func(self,ells, esig):
        # Individual weight constraints are a little tougher, we need to extract each \ell_i
        individual_weights_polynomial_ = get_weights(ells, esig)
        individual_weights_polynomial  = sym.lambdify([self.ell_coeffs], individual_weights_polynomial_)

        @wrapper_factory(N_assets=self.dim, n_terms=self.n_terms)
        def individual_weights_function(a):
            return individual_weights_polynomial(a)

        return individual_weights_function

class _get_funcs:
    def __init__(self,dim,level,ES_computer,transformation=None,*o,**kwarg):
        ell_coeffs  = [make_ell_coeffs(dim, level, "^" + str(i+1)) for i in range(dim)]
        ells        = [make_linear_functional(ell_coeff, dim, level) for ell_coeff in ell_coeffs]
        self.n_terms     = len(ell_coeffs[0])
        self.transformation = transformation
        if transformation:
            self.shift = self.transformation.f
        else:
            self.shift = lambda i: i+1
# Build the expected signature (from data, for now). normalise prices to start at 1
        esig    = ES_computer(2*(level+1),*o,**kwarg)
        esig_ll = ES_computer(2*(level+1),*o,**kwarg,transformation=transformation)
        

        _get_funcs = _pre_get_funcs(dim,self.n_terms,ell_coeffs)
        variance_function           = _get_funcs.set_signature_variance_func(ells, esig_ll, self.shift)
        return_function             = _get_funcs.set_signature_return_func(ells, esig_ll, self.shift)
        weight_sum_function         = _get_funcs.set_signature_weight_sum_func(ells, esig)
        individual_weights_function = _get_funcs.set_signature_individual_weight_func(ells, esig)
        self.funcs    = [variance_function, return_function, weight_sum_function, individual_weights_function]

class ExpectedSignature():
    """Class contains methods for compuing expected signatures"""
    model_life = np.infty

    def ExpectedSignature(self,T1=0,T2=1,transformation=None):
        pass

class GaussianProcessExpectedSiganture(ExpectedSignature):
    """ Estimate its expected signature from a trained Gaussian Process Model.

    Attributes:
    -----------------------------
    names: list
        names of the assets
    dim: int
        number of assets
    gpm: mogptk.Model
        a Gaussian Process model of type `mogptk.Models`[1]
    level: int
        the truncation level of expected signatures
    T1,T2: float
        the start/end time of the data
    train_mean: float
        the mean of starting point of sampled paths used to estimate the expected signature.
        It would be used to normalize price in test data
    paths: nd_array
        sampled paths from gpm model for estimating expected signatures

    Methods:
    --------------------------
    ExpectedSignature(self,transformation=None):
        estimate the expected signature from the sampled paths

    References:
    -------------------------
    [1]  T. de Wolff, A. Cuevas, and F. Tobar. MOGPTK: The Multi-Output Gaussian Process Toolkit. Neurocomputing, 2020
    """
    def __init__(self,gpm:mogptk.Model,model_life=5):
        """
        Arguments:
        ----------------
        gpm: mogptk.Models
            a Gaussian Process Model mogptk.Model
        model_life: int
            the pre-set limit for the range where the fit works well
        """
        self.model_life = model_life
        self.names = list(gpm.dataset.get_names())
        self.dim = len(self.names)
        # self.indexes = data.index.to_numpy()
        
        self.gpm = gpm
        #self.data.loc[:,'time'] = np.linspace(T1,T2,len(data)+model_life)[:len(data)]
    
        self.train_mean = None
        self.paths = None
    
    def _get_paths(self,T1=0,T2=1,time_step=100,n=1000):
        X = self.gpm.sample(X = np.linspace(T1,T2,time_step+1),n=n)

        # append the new observation to the (normed) price path
        self.paths = np.concatenate([x.T.reshape(-1,time_step+1,1) for x in X],axis=2)
        self.train_mean = np.mean(self.paths[:,0],axis=0)
        self.paths/=self.train_mean

    def ExpectedSignature(self,level,transformation=None):
        """
        Arguments:
        -----------------
        transformation=None: signature_trading._transformation
            if specified, the paths would firstly be transformed according to the transformation
        
        Returns:
        -----------------
        list
            carries the estimated expected signature of the Gaussian Process
        """
        if transformation:
            return ES(transformation.transformations(self.paths), max(level, 1))
        else:
            return ES(self.paths, max(level, 1))

class SignatureTrading:
    """ A class for signature trading.

    Attributes:
    ------------------
    data: pd.dataframe
    es: signature_trading.GaussianProcessExpectedSiganture
        the class carries information of lmc model, sampled paths and estimated expected signatures.
        the training and sampling is done in `__init__ `.
    normed_price: nd_array
        the price path normalized by the mean of starting point of the sampled paths
    funcs: list of callable objects
        used for optimizating the coefficients of expected signatures
    ellstars: nd_array
        the optimized coefficients the expected siganture. It is computed by `_get_coeffs` methods
    count: int
        count the life time of the model(a warning would be given if it exceeds the model life)

    Methods:
    ----------------
    _get_coeffs(self,interval=(0.05,0.15)):
        Optimize the portofolio variance with a given expected return. The coefficients corresponding to 
        minimzed portofolio variance among all expected return would be set to `ellstars`. The index of 
        the chosen coefficents, set of all expected return and minimized variance would be returned

    get_weights(self,price,interval=(0.05,0.15)):
        compute the optimized sig-weights of each assets
    """
    def __init__(self,data,es:ExpectedSignature,level=2):
        self.data = data

        self.es = es
        self.normed_price = self.data.loc[:,self.es.names].to_numpy()/self.es.train_mean
        self.funcs = None
        self.ellstars = None
        self.count = 0
        self.level = level

    def _get_funcs(self,transformation:_transformation=None):
        self.funcs = _get_funcs(self.es.dim,self.level,self.es.ExpectedSignature,transformation=transformation)

    def _get_coeffs(self,interval=(0.05,0.15)):
        """
        Return:
        -----------------
        (nd_array,nd_array)
            (expected return, corresponding optimized variance)
        """
        
        funcs,n_terms = self.funcs.funcs,self.funcs.n_terms
    
        expected_pnls = np.linspace(*interval,50)
        
        def signature_mean_variance_optim(expected_pnls, funcs, N, n_terms):
        
            res = []
            variance_function, return_function, weight_sum_function, individual_weights_function = funcs

            total_weight_con = [{'type': 'eq'  , 'fun': lambda w       : weight_sum_function(w) - 1.}]

            weight_lb_cons   = [{"type": "ineq", "fun": lambda w, ind=i: individual_weights_function(w)[ind]} for i in range(N)]
            weight_ub_cons   = [{"type": "ineq", "fun": lambda w, ind=i: 1. - individual_weights_function(w)[ind]} for i in range(N)]

            for pnl in expected_pnls:
                w0 = np.ones((N, n_terms), dtype=np.float64)/(N*n_terms)

                pnl_con  = [{'type': 'eq', 'fun': lambda w: return_function(w) - pnl}]

                all_cons = tuple(pnl_con + total_weight_con + weight_lb_cons + weight_ub_cons)

                optim = scipy.optimize.minimize(fun=variance_function, x0=w0.flatten(), constraints=all_cons, tol=1e-4)

                res.append(optim.x)

            return res
        ellstars = signature_mean_variance_optim(expected_pnls, funcs, self.es.dim, n_terms)

        sig_pnl     = np.array([funcs[1](ellstar) for ellstar in ellstars])
        sig_var     = np.array([funcs[0](ellstar) for ellstar in ellstars])
        
        mask = np.where(np.abs(sig_pnl-expected_pnls)<0.05)
        sig_pnl_m = sig_pnl[mask]
        sig_var_m = sig_var[mask]
        ellstars_m = [ellstars[i] for i in mask[0]]
        
        if ellstars_m:
            index = np.argmin(sig_var_m)
            self.ellstars = ellstars_m[index]
            return sig_pnl_m,sig_var_m
        else:
            index = np.argmin(sig_var)
            self.ellstars = ellstars[index]
            return sig_pnl,sig_var
        
    # consider using signatory for concatenating signatures
    def get_weights(self,price,interval=(0.05,0.15)):
        """
        Arguments:
        -----------------
        interval: tuple
            the upper and lower bound of the chosen expected return

        Return:
        -----------------
        nd_array
            optimized weights assigned to each assets
        """
        if self.count > self.es.model_life:
            print("Warning: Exceeds model life, please consider retrain the model.")
        elif self.count==0:
            self._get_coeffs(interval=interval)
        else:
            self.normed_price = np.concatenate((self.normed_price,price.reshape(1,-1)/self.es.train_mean))
        self.count += 1
        signatures_ = get_signature_values(self.normed_price[np.newaxis, :, :], self.es.level)
        es_weights = get_signature_weights(self.ellstars,signatures_, self.es.dim, self.funcs.n_terms)
        return es_weights
    
def trading_strategies(capital,weights,price,regularisation="ReLU"):
    """ Compute the number of shares to hold for each assets.

    Arguments:
    -------------------
    capital: float
        The total value of the portofolio
    weights: float
        Weights assigned to each assets
    price: float
        Price of each assets
    regularisation="ReLU": str
        mode of regularisation. No regularisation if is set to `None`

    Return:
    ------------------
    nd_array
        the number of shares to hold for each assets
    """
    if regularisation == "ReLU":
        pos_weight = weights*np.array(weights>0,dtype=np.float64)
        weights = pos_weight/np.sum(pos_weight)
    shares = capital*weights/price
    return shares