import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mv_utils import HoffLeadLag
import pandas as pd
import datetime as dt

matplotlib.rcParams.update({'text.usetex' : True, 'font.size': 16, 'axes.labelsize': 16, 'legend.fontsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16})

N = 2
df = pd.read_csv('./data/stocks.csv', index_col=0)
df.index = pd.to_datetime(df.index, format="%d/%m/%y")
names = df.columns[:N].to_list()
df2 = df[names].loc[(df.index > pd.Timestamp('2017-01-01')) & (df.index < pd.Timestamp('2017-03-01'))]
df2.loc[:,'time'] = np.linspace(0, 1, len(df2))
df2 = df2[['time'] + names]

paths = np.expand_dims(df2.to_numpy(), 0)
hoff_paths = HoffLeadLag(paths, True)

fig, axs = plt.subplots(1,2, figsize=(10, 5), sharey=True, dpi=200)
fig.suptitle(f"AAPL Stock Price from {dt.datetime.strftime(df2.index[0], r'%d/%m/%y')} to {dt.datetime.strftime(df2.index[-1], r'%d/%m/%y')}")
axs[0].plot(df2['time'], df2[names[0]], label=names[0])
axs[1].plot(hoff_paths[0][:,0], hoff_paths[0][:,1], label="Lead")
axs[1].plot(hoff_paths[0][:,3], hoff_paths[0][:,4], label="Lag", c="xkcd:baby blue")
axs[0].get_xaxis().set_ticks([])
axs[0].set_xlabel('Time')
axs[1].get_xaxis().set_ticks([])
axs[1].set_xlabel('Time')
axs[0].set_title('Original Path')
axs[1].set_title('Hoff Lead Lag Transformed Path')
axs[0].set_ylabel('Price')
axs[1].legend()
axs[0].legend()
fig.tight_layout()
fig.savefig('./plots/hoff_plot.png')
