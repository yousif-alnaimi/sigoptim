# --------------------------------------------------------------------
# This script is a test script for the faster implementation
# of the signature trading strategy. This script is not used in the 
# project as it cannot implement Hoff Lead Lag Paths, but is kept
# for future reference.
# --------------------------------------------------------------------



import numba
from numba import jit,njit
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import copy
import pandas as pd
import mogptk  # pytorch >= 11.0 is required
import torch # pytorch 11.0 is used
import signatory # if cannot find a version compatible with pytorch, 
                 # install locally follow instructions on https://github.com/patrick-kidger/signatory
import time
from scipy.optimize import minimize, LinearConstraint
from src.optimisation.signature import ES
from src.signature_trading import _get_funcs
import unittest

from mv_utils import *

device = torch.device("cuda")
N = 3
level = 2

# stocks
df      = pd.read_csv('./data/stocks.csv', index_col=0)
names   = df.columns[:N].to_list()
# format needs to be specified as default parsing is American format
df.index = pd.to_datetime(df.index, format="%d/%m/%y")

# df2 = df[names].loc[(df.index > pd.Timestamp('2017-04-01')) & (df.index < pd.Timestamp('2017-06-01'))]
df2 = df[names].loc[(df.index > pd.Timestamp('2017-01-01')) & (df.index < pd.Timestamp('2018-01-01'))]

def monthly_ts_split(df):
    """
    Takes in df and splits it into a list of progressively longer time series,
    end at the end of the next month each time.
    """
    df_list = []
    start_month, start_year = df.index[0].month, df.index[0].year
    curr_month, curr_year = start_month, start_year

    unfinished = True
    while unfinished:
        curr_month += 1
        if curr_month > 12:
            curr_month = 1
            curr_year += 1

        new_df = df.loc[:pd.Timestamp(f"{curr_year}-{curr_month}-01") - pd.Timedelta(1, "s"), :]
        df_list.append(new_df)
        if new_df.index[-1] == df.index[-1]:
            unfinished = False

    return df_list

df_split = monthly_ts_split(df2)

# mogptk.use_cpu()

if torch.cuda.is_available():
   mogptk.use_gpu(0)

training_size = 0.95
Q = N
init_method = 'BNSE'
method = 'Adam'

df2.loc[:,'time'] = np.linspace(0, 1, len(df2))
# print(df2.to_numpy().shape)

gpm_dataset = mogptk.LoadDataFrame(df2, x_col='time', y_col=names)

for channel in gpm_dataset:
    # normalize the path and fit the normalized path by y=a*t+b_t.
    # The train would be performed on b_t 
    channel.transform(mogptk.TransformNormalize())
    channel.transform(mogptk.TransformDetrend())
for name in names:
    # remove the data after `training_size` as test data 
    gpm_dataset[name].remove_randomly(pct=1-training_size)

# set the model and train it
gpm = mogptk.SM_LMC(gpm_dataset, Q=Q, inference=mogptk.Exact())
gpm.init_parameters(init_method)
loss,error = gpm.train(method=method, lr=0.01, iters=1000, verbose=False, plot=False)

# dim = N
word_concatenate_shuffle_dict = word_concatenate_shuffle(level,level)

# generate a list containing paths of each channels
n_samples = 100
time_steps = len(df2.index)
X = gpm.sample(np.linspace(0,1,time_steps),n=n_samples)
# rearrange the result into (number_of_paths,time_step+1,dim)
paths = np.concatenate([x.T.reshape(-1,time_steps,1) for x in X],axis=2)
# normalize with the mean of the price at time 0
paths /= np.mean(paths[:,0],axis=0)
# time_inds = np.expand_dims(np.stack([np.linspace(0,1,time_steps) for _ in range(n_samples)], axis=0), axis=2)
# print(time_inds.shape)

paths = subtract_first_row(paths)
dim = N + 1
# dim = N
# paths = np.concatenate((time_inds, paths), axis=2)

# print(paths.shape)
# print(paths[0])
lengths = [len(df.index) for df in df_split]
path_list = [paths[:, :length, :] for length in lengths]


# # apply transformations to paths
hoff_path_list = [HoffLeadLag(path) for path in path_list]
print(path_list[-1][0])

# tensor_path = torch.tensor(paths, device=device)
# # print(tensor_path.shape)

# # compute expected signature
# signature = signatory.signature(tensor_path,level*2).mean(axis=0)

# # add zero level to the signature
# signature = np.concatenate([[1],np.array(signature.cpu())])

# do the same as the above but for time points specified above
tensor_path_list = [torch.tensor(path, device=device) for path in path_list]
signature_list = [signatory.signature(path, level*2).mean(axis=0) for path in tensor_path_list]
signature_list = [np.concatenate([[1],np.array(sig.cpu())]) for sig in signature_list]

hoff_tensor_path_list = [torch.tensor(path, device=device) for path in hoff_path_list]
hoff_signature_list = [signatory.signature(path, level*2).mean(axis=0) for path in hoff_tensor_path_list]
hoff_signature_list = [np.concatenate([[1],np.array(sig.cpu())]) for sig in hoff_signature_list]


start0 = time.time()
# # get coefficients of a^i_w in mean and variance functions 
# mean_weights = np.array(
#     [integration_functional_coeff(signature,dim,level,i) for i in range(dim)]).reshape(1,-1)
# var_coeff = get_sig_variance(signature,word_concatenate_shuffle_dict,dim,level)

mean_weights_list = [np.array([integration_functional_coeff(sig,dim,level,i) for i in range(dim, 2*dim)]).reshape(1,-1) for sig in hoff_signature_list] # change dims here for hoff
var_coeff_list = [get_sig_variance(sig,word_concatenate_shuffle_dict,dim,level, hoff=True) for sig in hoff_signature_list]
weights_sum_list = [get_weights_sum_coeff(sig,dim,level) for sig in signature_list]

# # get coefficients of a^i_w in weights and weight sum functions 
# weights_sum = get_weights_sum_coeff(signature,dim,level)
# A = np.array([get_weights_coeff(signature,i,dim,level) for i in range(dim)])

z = df2.to_numpy()
z = subtract_first_row(np.expand_dims(z, axis=0))
orig_path_list = [z[:length, :] for length in lengths]
orig_tpath_list = [torch.tensor(path, device=device) for path in orig_path_list]
orig_sig_list = [signatory.signature(path, level*2).mean(axis=0) for path in orig_tpath_list]
print(orig_sig_list[0].shape)
orig_sig_list = [np.concatenate([[1],np.array(sig.cpu())]) for sig in orig_sig_list]

# dim = N + 1
# A_mats = []
# for sig in signature_list:
#     mat = []
#     for i in range(dim):
#         print(get_weights_coeff(sig,i,dim,level))
#         mat.append(get_weights_coeff(sig,i,dim,level))
#     A_mats.append(np.array(mat))
# print(get_weights_coeff(signature_list[-1],2,dim,level))
A_mats = [np.array([get_weights_coeff(sig,i,dim,level) for i in range(dim)]) for sig in signature_list]
# A_mats = [np.array([get_weights_coeff(sig,i,dim,level) for i in range(dim)]) for sig in orig_sig_list]
# sig_len = length_of_signature(dim,level)
# print(A[0, :sig_len], A[1, sig_len:2*sig_len])

print('time for getting coefficents(including compiling time):', time.time() - start0)

# exp_return = 0.05

def signature_mean_variance_optim(exp_return, var_coeff=var_coeff_list[-1], mean_weights=mean_weights_list[-1], weights_sum=weights_sum_list[-1], A=A_mats[-1]):
    object_function = lambda coeff: coeff.T@(var_coeff+np.eye(var_coeff.shape[0])*1e-5)@coeff
    
    cons = ({'type': 'eq', 'fun': lambda coeff: np.squeeze(mean_weights)@coeff-exp_return},
                       {'type': 'eq', 'fun': lambda coeff: weights_sum@coeff-1},
                       LinearConstraint(A,lb=np.zeros(dim),ub=np.ones(dim)),
                    #    {'type': 'eq', 'fun': lambda coeff: (A@coeff)[0]} # only enable if time aug
                       ) # time aug 
    
    res = minimize(object_function, np.ones(length_of_signature(dim,level)*dim), method='SLSQP',
                   constraints=cons)
    
    return res

def portfolio_variance(res, var_coeff):
    return res.x.T @ var_coeff @ res.x

def portfolio_exp_ret(mean_weights, res):
    return np.squeeze(mean_weights) @ res.x

# define shorthand to pick out a specific signature and optimise for that point in time
print('No. of trade points: ', len(signature_list))
fig, axs = plt.subplots(len(signature_list), 2, figsize=(10, 5*len(signature_list)))
start = time.time()
for index in range(len(signature_list)):
    vc = var_coeff_list[index]
    mw = mean_weights_list[index]
    ws = weights_sum_list[index]
    A = A_mats[index]

    expected_pnls = np.linspace(-0.02, 0.1, 50)
    ellstars = []
    # for pnl in tqdm(expected_pnls):
    for pnl in expected_pnls:
        ellstars.append(signature_mean_variance_optim(pnl, vc, mw, ws, A))

    # plot efficient frontier
    risks = [portfolio_variance(ell, vc) for ell in ellstars]
    pnls_real = [portfolio_exp_ret(mw, ell) for ell in ellstars]

    axs[index, 0].scatter(risks, expected_pnls)
    axs[index, 0].set_xlabel('Portfolio risk')
    axs[index, 0].set_ylabel('Portfolio P&L')
    axs[index, 0].set_title('Efficient frontier ideal')

    axs[index, 1].scatter(risks, pnls_real)
    axs[index, 1].set_xlabel('Portfolio risk')
    axs[index, 1].set_ylabel('Portfolio P&L')
    axs[index, 1].set_title('Efficient frontier real')

    sharpe_list = []
    for i in range(len(ellstars)):
        res = ellstars[i]
        var = portfolio_variance(res, vc)
        ret = portfolio_exp_ret(mw, res)
        rfr = 0
        sharpe = (ret - rfr) / np.sqrt(var)
        sharpe_list.append(sharpe)

    max_sharpe = np.max(sharpe_list)
    argmax_sharpe = np.argmax(sharpe_list)
    print('Max sharpe ratio:', max_sharpe, 'at index:', argmax_sharpe)

    res = ellstars[argmax_sharpe]
    # print(len(res.x), A.shape)
    asset_weights = A @ res.x
    print('weights of each asset:', asset_weights) 
    print('result portfolio variance:', portfolio_variance(res, vc))
    print('result portfolio expected return:', portfolio_exp_ret(mw, res))
    print('weights sum: ', (asset_weights).sum())
    if abs((asset_weights).sum() - 1) > 1e-5:
        print("FAILED! Weights do not sum to 1!")
    else:
        print("PASSED! Weights sum to 1!")
    print("\n")

print("Time taken for all iterations:", time.time() - start)
fig.tight_layout()
fig.savefig('efficient_frontier.png')
