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

warnings.simplefilter("ignore")

device = torch.device("cuda")

if torch.cuda.is_available():
   mogptk.use_gpu(0)

dim = 2
level = 2
offset = 1
softmax_truncate = 1
frontier_interval = (0.05, 0.25)

training_size = 0.95
Q = dim
init_method = 'BNSE'
method = 'Adam'



# stocks
df      = pd.read_csv('./stocks.csv', index_col=0)
names   = df.columns[offset:dim+offset].to_list()
# format needs to be specified as default parsing is American format
df.index = pd.to_datetime(df.index, format="%d/%m/%y")

df2 = df[names].loc[(df.index > pd.Timestamp('2017-01-01')) & (df.index < pd.Timestamp('2018-01-01'))]
df2.loc[:,'time'] = np.linspace(0, 1, len(df2))
# df2 = df2 - df2.iloc[0].squeeze()

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

# print(type(gpm))

# class RandomPathExpectedSignautre():
#     model_life = np.infty
#     def _get_paths(self,n,time_step,channel):
#         self.paths = torch.rand(n,time_step+1,channel)
    
#     def ExpectedSignature(self,level,transformation=None):
#         return ES(self.paths,level)

start = time.time()
print("Started")
transformation = _transformation(dim, 1, OrderedDict(
                {"AddTime": True, "TranslatePaths": True, "ScalePaths": False, "LeadLag": False, "HoffLeadLag":True}))
    
es = GaussianProcessExpectedSiganture(gpm)
es._get_paths(time_step=len(df2.index))

# print(df2.loc[:,es.names])
# print(df2.loc[:,es.names].to_numpy())
# print(df2.loc[:,es.names].to_numpy().shape)
# print(np.expand_dims(df2.loc[:,es.names].to_numpy(), 0).shape)
sig = OriginalSignature(df2.loc[:,es.names])
sig._get_paths()

print(1)
sig_trading = SignatureTrading(df2, es, sig, level, softmax_truncate)
print(2)
sig_trading._get_funcs(transformation)
print(3)
pnl_list, var_list, ellstars_list = sig_trading._get_coeffs(interval=frontier_interval, export_ellstars=True)
print(4)
old_real_weights, exp_weights, real_weights = sig_trading.get_weights(sig_trading.data.loc[:,sig_trading.es.names].to_numpy(), interval=frontier_interval, ellstars_list=ellstars_list)
print(real_weights)
print("Sum of weights:")
print([w.sum() for w in real_weights])
print("TIME TAKEN")
print(time.time()-start)#

def relu_scale(x):
    new_x = np.maximum(x, 0)
    return new_x / new_x.sum()

relu_real_weights = np.array([relu_scale(w) for w in real_weights])
# print(relu_real_weights)
# print([w.sum() for w in relu_real_weights])
print(relu_real_weights.shape)

fig, axs = plt.subplots(2, figsize=(10, 10))
axs[0].plot(var_list, pnl_list)
axs[1].plot(relu_real_weights[:,0], pnl_list)
axs[1].plot(relu_real_weights[:,1], pnl_list)
plt.show()
