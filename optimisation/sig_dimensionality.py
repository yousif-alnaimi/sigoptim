import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams.update({'text.usetex' : True, 'font.size': 16, 'axes.labelsize': 16, 'legend.fontsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16})

M_range = np.arange(1, 4)
d_range = np.arange(1, 6)
sig_range_list = []
for M in M_range:
    sig_list = []
    for d in d_range:
        sig_list.append(np.sum((2*(d+1))**np.arange(2*(M+1) + 1)))
    sig_range_list.append(np.array(sig_list))
fig, ax = plt.subplots(1, figsize=(10, 5), dpi=200)
for M in M_range:
    ax.plot(d_range, sig_range_list[M-1], label=f'$M$={M}', linestyle='', marker='o', markersize=10)

ax.set_yscale('log')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel('Number of Assets ($d$)')
ax.set_ylabel('Number of Signature Terms')
ax.set_title('Number of Signature Terms Against Number of Assets ($d$) for Different Orders ($M$)')
ax.legend(loc='upper left', title="Order of Signature")
ax.set_ylim([1e2, 1e9])
fig.tight_layout()
fig.savefig('sig_dimensionality.png')
