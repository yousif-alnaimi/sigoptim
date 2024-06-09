from src.signature_trading import SignatureTrading, GaussianProcessExpectedSiganture, _transformation, OriginalSignature
# from src.optimisation.signature import *
from collections import OrderedDict
import time
import signatory
import mogptk
import numpy as np
import torch
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'text.usetex' : True, 'font.size': 16, 'axes.labelsize': 16, 'legend.fontsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16})

# warnings.simplefilter("ignore")

device = torch.device("cuda")

if torch.cuda.is_available():
   mogptk.use_gpu(0)


def read_data(dim, offset, start_date, end_date, include_ffr=False):
    # stocks
    df      = pd.read_csv('./stocks.csv', index_col=0)
    names   = df.columns[offset:dim+offset].to_list()
    # format needs to be specified as default parsing is American format
    df.index = pd.to_datetime(df.index, format="%d/%m/%y")

    df2 = df[names].loc[(df.index >= pd.Timestamp(start_date)) & (df.index < pd.Timestamp(end_date))]
    df2.loc[:,'time'] = np.linspace(0, 1, len(df2))

    ffr = pd.read_csv('./fedfunds.csv', index_col=0)
    ffr.index = pd.to_datetime(ffr.index)
    ffr = ffr.loc[(ffr.index >= pd.Timestamp(start_date)) & (ffr.index < pd.Timestamp(end_date))]

    test = ffr.reindex(df2.index, method='ffill')
    test["days1"] = test.index
    test["days2"] = test.index
    test["days"] = (test["days1"] - test["days2"].shift(1)).dt.days
    test["mult"] = (1. + test["FEDFUNDS"]/100.)**(test["days"]/365.)
    test["ffr"] = test["mult"].cumprod()
    test["ffr"].fillna(1., inplace=True)
    df2['ffr'] = test["ffr"]

    if include_ffr:
        df2 = df2[['ffr'] + names + ['time']]
        names = ['ffr'] + names

    return df2, names


def relu_scale(x):
    new_x = np.maximum(x, 0)
    return new_x / new_x.sum()

def make_gpm(df2, names, training_size, Q, init_method, method):
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
    return gpm


def sig_trading(gpm, df2, names, frontier_interval, level=2, sample_dict={2: 1000, 3:250, 4:10}, softmax_truncate=1):
    dim = len(names)
    start = time.time()
    transformation = _transformation(dim, 1, OrderedDict(
                    {"AddTime": True, "TranslatePaths": True, "ScalePaths": False, "LeadLag": False, "HoffLeadLag":True}))
        
    es = GaussianProcessExpectedSiganture(gpm)
    es._get_paths(
        time_step=len(df2.index),
        n=sample_dict[dim],
        )

    print("Paths successfully generated. Signature trading now in progress...")
    sig = None

    sig_trading = SignatureTrading(df2, es, sig, level, softmax_truncate)
    sig_trading._get_funcs(transformation)
    pnl_list, var_list, ellstars_list = sig_trading._get_coeffs(interval=frontier_interval, export_ellstars=True)
    real_weights = sig_trading.get_weights(sig_trading.data.loc[:,sig_trading.es.names].to_numpy(), interval=frontier_interval, ellstars_list=ellstars_list)

    relu_real_weights = np.array([relu_scale(w) for w in real_weights])
    print("Efficient frontier calculated. Time Taken:")
    print(time.time()-start)
    return pnl_list, var_list, relu_real_weights


def make_plots(pnl_list, var_list, relu_real_weights, names):
    dim = len(names)
    fig, axs = plt.subplots(2, figsize=(10, 10))
    axs[0].plot(np.sqrt(np.array(var_list)), pnl_list)
    for i in range(dim):
        axs[1].plot(pnl_list, relu_real_weights[:,i],label=names[i])
    axs[1].set_ylim([0., 1.])
    axs[1].legend()
    plt.show()


df2, names = read_data(dim=2, offset=1, start_date='2017-01-01', end_date='2018-01-01', include_ffr=True)
gpm = make_gpm(df2, names, training_size=0.95, Q=2, init_method='BNSE', method='Adam')
pnl_list, var_list, relu_real_weights = sig_trading(gpm, df2, names, level=2, frontier_interval=(0.05, 0.25))
make_plots(pnl_list, var_list, relu_real_weights, names)
