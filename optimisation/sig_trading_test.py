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

warnings.simplefilter("ignore")

device = torch.device("cuda")

if torch.cuda.is_available():
   mogptk.use_gpu(0)

dim = 2
level = 2
softmax_truncate = 1

training_size = 0.95
Q = dim
init_method = 'BNSE'
method = 'Adam'



# stocks
df      = pd.read_csv('./stocks.csv', index_col=0)
names   = df.columns[:dim].to_list()
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
sig_trading._get_coeffs()
print(4)
weights = sig_trading.get_weights(sig_trading.data.loc[:,sig_trading.es.names].to_numpy())
print(weights)
print("Sum of weights:")
print(weights.sum())
print("TIME TAKEN")
print(time.time()-start)