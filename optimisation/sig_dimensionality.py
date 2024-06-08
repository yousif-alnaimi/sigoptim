import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'text.usetex' : True, 'font.size': 16, 'axes.labelsize': 16, 'legend.fontsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
# sns.set(rc={'text.usetex' : True, 'font.size': 16, 'axes.labelsize': 16, 'legend.fontsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
# sns.scatterplot(np.linspace(0,1,10))
plt.plot(np.linspace(0,1,10), np.linspace(0,1,10))
plt.show()