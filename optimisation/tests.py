import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mogptk  # pytorch >= 11.0 is required
from mv_utils import *

N=2

df      = pd.read_csv('./data/stocks.csv', index_col=0)
names   = df.columns[:N].to_list()
# format needs to be specified as default parsing is American format
df.index = pd.to_datetime(df.index, format="%d/%m/%y")

# df2 = df[names].loc[(df.index > pd.Timestamp('2017-04-01')) & (df.index < pd.Timestamp('2017-06-01'))]
df2 = df[names].loc[(df.index > pd.Timestamp('2017-01-01')) & (df.index < pd.Timestamp('2017-02-01'))]

df2.loc[:,'time'] = np.linspace(0, 1, len(df2))

training_size = 0.95
Q = N
init_method = 'BNSE'
method = 'Adam'

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

dim = N + 1
level = 2

# generate a list containing paths of each channels
n_samples = 100
time_steps = len(df2.index)
X = gpm.sample(np.linspace(0,1,time_steps),n=n_samples)
# rearrange the result into (number_of_paths,time_step+1,dim)
paths = np.concatenate([x.T.reshape(-1,time_steps,1) for x in X],axis=2)
# normalize with the mean of the price at time 0
paths /= np.mean(paths[:,0],axis=0)

paths = subtract_first_row(paths)

time_inds = np.expand_dims(np.stack([np.linspace(0,1,time_steps) for _ in range(n_samples)], axis=0), axis=2)

print(paths.shape)
# print(paths[0])

paths_t = np.concatenate((time_inds, paths), axis=2)

# print(paths_t[0])
print(paths_t.shape)

hoff_paths = HoffLeadLag(paths, False)
hoff_paths_t = HoffLeadLag(paths_t, True)

print(hoff_paths[0])
print(hoff_paths_t[0])
print(hoff_paths.shape)
print(hoff_paths_t.shape)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(hoff_paths[0, :, 1])
axs[0].plot(hoff_paths[0, :, dim+1])
axs[0].plot(hoff_paths_t[0, :, 1])
axs[0].plot(hoff_paths_t[0, :, dim+1])
axs[1].plot(paths[0, :, 0])
axs[1].plot(paths_t[0, :, 1])
fig.show()
fig.savefig('test.png')
