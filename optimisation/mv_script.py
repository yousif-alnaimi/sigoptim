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
                 # install locally follows insturctions on https://github.com/patrick-kidger/signatory
import time
from scipy.optimize import minimize, LinearConstraint
from src.optimisation.signature import ES
from src.signature_trading import _get_funcs
import unittest

from mv_utils import *

device = torch.device("cuda")
N = 3
level = 3

# stocks
df      = pd.read_csv('./stocks.csv', index_col=0)
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

mogptk.use_cpu()

# if torch.cuda.is_available():
#    mogptk.use_gpu(0)

training_size = 0.95
Q = N
init_method = 'BNSE'
method = 'Adam'

df2.loc[:,'time'] = np.linspace(0, 1, len(df2))

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

dim = N
word_concatenate_shuffle_dict = word_concatenate_shuffle(level,level)

# generate a list containing paths of each channels
time_steps = len(df2.index)
X = gpm.sample(np.linspace(0,1,time_steps),n=1000)
# rearrange the result into (number_of_paths,time_step+1,dim)
paths = np.concatenate([x.T.reshape(-1,time_steps,1) for x in X],axis=2)
# normalize with the mean of the price at time 0
paths /= np.mean(paths[:,0],axis=0)

# print(paths.shape)
lengths = [len(df.index) for df in df_split]
path_list = [paths[:, :length, :] for length in lengths]

tensor_path = torch.tensor(paths, device=device)

# compute expected signature
signature = signatory.signature(tensor_path,level*2).mean(axis=0)

# add zero level to the signature
signature = np.concatenate([[1],np.array(signature.cpu())])

# do the same as the above but for time points specified above
tensor_path_list = [torch.tensor(path, device=device) for path in path_list]
signature_list = [signatory.signature(path, level*2).mean(axis=0) for path in tensor_path_list]
signature_list = [np.concatenate([[1],np.array(sig.cpu())]) for sig in signature_list]


start0 = time.time()
# # get coefficients of a^i_w in mean and variance functions 
# mean_weights = np.array(
#     [integration_functional_coeff(signature,dim,level,i) for i in range(dim)]).reshape(1,-1)
# var_coeff = get_sig_variance(signature,word_concatenate_shuffle_dict,dim,level)

mean_weights_list = [np.array([integration_functional_coeff(sig,dim,level,i) for i in range(dim)]).reshape(1,-1) for sig in signature_list]
var_coeff_list = [get_sig_variance(sig,word_concatenate_shuffle_dict,dim,level) for sig in signature_list]
weights_sum_list = [get_weights_sum_coeff(sig,dim,level) for sig in signature_list]

# # get coefficients of a^i_w in weights and weight sum functions 
# weights_sum = get_weights_sum_coeff(signature,dim,level)
# A = np.array([get_weights_coeff(signature,i,dim,level) for i in range(dim)])
A_mats = [np.array([get_weights_coeff(sig,i,dim,level) for i in range(dim)]) for sig in signature_list]
# sig_len = length_of_signature(dim,level)
# print(A[0, :sig_len], A[1, sig_len:2*sig_len])

print('time for getting coefficents(including compiling time):', time.time() - start0)

# exp_return = 0.05

def signature_mean_variance_optim(exp_return, var_coeff=var_coeff_list[-1], mean_weights=mean_weights_list[-1], weights_sum=weights_sum_list[-1], A=A_mats[-1]):
    object_function = lambda coeff: coeff.T@(var_coeff+np.eye(var_coeff.shape[0])*1e-5)@coeff
    
    cons = ({'type': 'eq', 'fun': lambda coeff: np.squeeze(mean_weights)@coeff-exp_return},
                       {'type': 'eq', 'fun': lambda coeff: weights_sum@coeff-1},
                       LinearConstraint(A,lb=np.zeros(dim),ub=np.ones(dim)))
    
    start = time.time()
    res = minimize(object_function, np.ones(length_of_signature(dim,level)*dim), method='SLSQP',
                   constraints=cons)
    # print('time for optimisation:',time.time()-start)
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
    for pnl in tqdm(expected_pnls):
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

    res = ellstars[index]
    # print(len(res.x), A.shape)
    asset_weights = A @ res.x
    print('weights of each asset:', asset_weights) 
    print('result portfolio variance:', portfolio_variance(res, vc))
    print('result portfolio expected return:', portfolio_exp_ret(mw, res))
    print('weights sum: ', (asset_weights).sum())
    if (asset_weights).sum() - 1 > 1e-5:
        print("FAILED! Weights do not sum to 1!")
    else:
        print("PASSED! Weights sum to 1!")

print("Time taken for all iterations:", time.time() - start)
fig.savefig('efficient_frontier.png')
