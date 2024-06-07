from collections import OrderedDict

import numpy as np
import sympy as sym
import mogptk
import torch
from .transformations import *
from copy import copy, deepcopy
import time
from .free_lie_algebra import *
from .optimisation.signature import *
from .helper_functions.signature_helper_functions import get_signature_weights, shift, get_signature_values

from tqdm import tqdm
get_weights_ell = copy(get_weights)

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


def shuffle_power(ell, power):
    """
    Shuffle product of a linear functional with itself power times

    :param ell:     Linear functional
    :param power:   Power to shuffle to
    :return:        Shuffled linear functional
    """
    res = deepcopy(ell)
    for _ in range(power - 1):
        # print(type(res))
        res = shuffleProduct(res, ell)
    return res

def shuffle_softmax(ells, truncate=5):
    """
    Shuffle product of a linear functional with itself truncated times

    :param ell:     Linear functional
    :param truncate: Number of times to shuffle
    :return:        Shuffled linear functional
    """
    if isinstance(ells, list):
        ells_ = []
        for ell in ells:
            res = ell
            for i in range(2, truncate + 1):
                # print(type(res))
                res += shuffle_power(res, i) *(1/np.math.factorial(i))
            ells_.append(res)
        return ells_
    else:
        res = ells
        for i in range(2, truncate + 1):
            # print(type(res))
            res += shuffle_power(res, i) *(1/np.math.factorial(i))
        return res


class _pre_get_funcs:
    def __init__(self,dim,n_terms,ell_coeffs, default_truncate):
        self.dim = dim
        self.n_terms = n_terms
        self.ell_coeffs = ell_coeffs
        self.truncate = default_truncate

    # Set the various functions used in the optimisation
    def set_signature_variance_func(self,ells, esig, shift):
        # Lambdify optimising functional
        if self.truncate == 1:
            variance_polynomial_ = portfolio_variance(ells, esig, shift)
        else:
            variance_polynomial_ = portfolio_variance_softmax(shuffle_softmax(ells, self.truncate), esig, shift)
        variance_polynomial  = sym.lambdify([self.ell_coeffs], variance_polynomial_)

        @wrapper_factory(N_assets=self.dim, n_terms=self.n_terms)
        def variance_function(a):
            return variance_polynomial(a)

        return variance_function

    def set_signature_return_func(self,ells, esig, shift):
        if self.truncate == 1:
            return_polynomial_ = portfolio_return(ells, esig, shift)
        else:
            return_polynomial_ = portfolio_return_softmax(shuffle_softmax(ells, self.truncate), esig, shift)
        return_polynomial  = sym.lambdify([self.ell_coeffs], return_polynomial_)

        @wrapper_factory(N_assets=self.dim, n_terms=self.n_terms)
        def return_function(a):
            return return_polynomial(a)

        return return_function

    def set_signature_weight_sum_func(self,ells, esig):
        if self.truncate == 1:
            weight_sum_polynomial_= sum_weights(ells, esig)
        else:
            weight_sum_polynomial_= sum_weights_softmax(shuffle_softmax(ells, self.truncate), esig)
        weight_sum_polynomial = sym.lambdify([self.ell_coeffs], weight_sum_polynomial_)

        @wrapper_factory(N_assets=self.dim, n_terms=self.n_terms)
        def weight_sum_function(a):
            return weight_sum_polynomial(a)

        return weight_sum_function

    def set_signature_individual_weight_func(self,ells, esig):
        # Individual weight constraints are a little tougher, we need to extract each \ell_i
        if self.truncate == 1:
            individual_weights_polynomial_ = get_weights(ells, esig)
        else:
            individual_weights_polynomial_ = get_weights_softmax(shuffle_softmax(ells, self.truncate), esig)
        individual_weights_polynomial  = sym.lambdify([self.ell_coeffs], individual_weights_polynomial_)

        @wrapper_factory(N_assets=self.dim, n_terms=self.n_terms)
        def individual_weights_function(a):
            return individual_weights_polynomial(a)

        return individual_weights_function
    


class _get_funcs:
    def __init__(self,dim,level,ES_computer,truncate,transformation=None,*o,**kwarg):
        ell_coeffs  = [make_ell_coeffs(dim, level, "^" + str(i+1)) for i in range(dim)]
        # print([i * c for i,c in enumerate(ell_coeffs)])
        # print(ell_coeffs[0])
        ells        = [make_linear_functional(ell_coeff, dim, level) for ell_coeff in ell_coeffs]
        # print(ells[0])
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # print(ells[0], type(ells[0]))
        # shuf = shuffleProduct(ells[0], ells[0])
        # print(shuf, type(shuf))
        # print(shuf*(.5))
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # ells = [shuffle_softmax(ell, 2) for ell in tqdm(ells)]
        self.n_terms     = len(ell_coeffs[0])
        self.transformation = transformation
        if transformation:
            self.shift = self.transformation.f
        else:
            self.shift = lambda i: i+1
# Build the expected signature (from data, for now). normalise prices to start at 1
        esig    = ES_computer(2*(level+1),*o,**kwarg)
        print("sigs")
        esig_ll = ES_computer(2*(level+1),*o,**kwarg,transformation=transformation)
        print("SIGS")
        

        _get_funcs = _pre_get_funcs(dim,self.n_terms,ell_coeffs, truncate)
        print("GET FUNCS")
        variance_function           = _get_funcs.set_signature_variance_func(ells, esig_ll, self.shift)
        return_function             = _get_funcs.set_signature_return_func(ells, esig_ll, self.shift)
        weight_sum_function         = _get_funcs.set_signature_weight_sum_func(ells, esig)
        individual_weights_function = _get_funcs.set_signature_individual_weight_func(ells, esig)
        print("GOT FUNCS")
        self.funcs    = [variance_function, return_function, weight_sum_function, individual_weights_function]

class ExpectedSignature():
    """Class contains methods for compuing expected signatures"""
    model_life = np.infty

    def ExpectedSignature(self,T1=0,T2=1,transformation=None):
        pass


class OriginalSignature(ExpectedSignature):
    def __init__(self, data):
        self.data = data

    def _get_paths(self):
        self.paths = np.expand_dims(self.data.to_numpy(), 0)
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
    def __init__(self,data,es:ExpectedSignature,sig:ExpectedSignature,level=2, truncate=1):
        self.data = data

        self.es = es
        self.normed_price = self.data.loc[:,self.es.names].to_numpy()/self.es.train_mean
        self.funcs = None
        self.ellstars = None
        self.count = 0
        self.level = level
        self.truncate = truncate
        self.sig = sig

    def _get_funcs(self,transformation:_transformation=None):
        self.funcs = _get_funcs(self.es.dim,self.level,self.es.ExpectedSignature,self.truncate,transformation=transformation)
        self.funcs2 = _get_funcs(self.es.dim,self.level,self.sig.ExpectedSignature,self.truncate, transformation=transformation)

    def _get_coeffs(self,interval=(0.05,0.15),export_ellstars=False):
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

            for pnl in tqdm(expected_pnls):
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
            if export_ellstars:
                return sig_pnl_m,sig_var_m,ellstars_m
            else:
                return sig_pnl_m,sig_var_m
        else:
            index = np.argmin(sig_var)
            self.ellstars = ellstars[index]
            print("Warning: No expected return in the range, the lowest variance is chosen.")
            return sig_pnl,sig_var
        
    # consider using signatory for concatenating signatures
    def get_weights(self,price,interval=(0.05,0.15), ellstars_list=None):
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

        signatures_ = get_signature_values(self.normed_price[np.newaxis, :, :], self.level)
        f = self.funcs.funcs[3]
        f2 = self.funcs2.funcs[3]

        if ellstars_list:
            es_weights_list = []
            f_weights_list = []
            f2_weights_list = []
            for i in range(len(ellstars_list)):
                es_weights_list.append(get_signature_weights(ellstars_list[i],signatures_, self.es.dim, self.funcs.n_terms))
                f_weights_list.append(f(ellstars_list[i]))
                f2_weights_list.append(f2(ellstars_list[i]))

            return es_weights_list, f_weights_list, f2_weights_list
        else:
            es_weights = get_signature_weights(self.ellstars,signatures_, self.es.dim, self.funcs.n_terms)
            # ell_coeffs = make_ell_coeffs(self.es.dim, self.level)
            # ell_coeffs  = [make_ell_coeffs(self.es.dim, self.level, "^" + str(i+1)) for i in range(self.es.dim)]
            # # print(len(ell_coeffs), len(self.ellstars))
            # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            # # ell_coeffs2 = [self.ellstars[i] * ell_coeffs[i] for i in range(len(ell_coeffs))]
            # ells2 = [make_linear_functional(ell_coeff, self.es.dim, self.level) for ell_coeff in ell_coeffs]
            # ells3 = [shuffle_softmax(ell, self.truncate) for ell in ells2]
            # # print(len(ells3))
            # print(len(self.ellstars))
            # individual_weights_polynomial_ = get_weights_ell(shuffle_softmax(ells3, self.truncate), signatures_)
            # individual_weights_polynomial  = sym.lambdify([ell_coeffs], individual_weights_polynomial_)
            # print("bweioafuiowafiouwafniowaf")
            # @wrapper_factory(N_assets=self.es.dim, n_terms=self.es.n_terms)
            # def individual_weights_function(a):
            #     return individual_weights_polynomial(a)

            # print(type(individual_weights_function(self.ellstars)))
            # print("ionfqenweioaioaneaiofnwafnio")
            # time.sleep(.1)
            w = f(self.ellstars)
            print("Expected:")
            print(w)
            print(w.sum())
            print("Old Real:")
            print(es_weights)
            print("Real:")
            return f2(self.ellstars)
    
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