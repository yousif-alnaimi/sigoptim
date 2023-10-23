import matplotlib.pyplot as plt
import numpy as np


def make_grid(axis=None):
    _plt_obj = axis if axis is not None else plt
    getattr(_plt_obj, "grid")(visible=True, color='grey', linestyle=':', linewidth=1.0, alpha=0.3)
    getattr(_plt_obj, "minorticks_on")()
    getattr(_plt_obj, "grid")(visible=True, which='minor', color='grey', linestyle=':', linewidth=1.0, alpha=0.1)


def plot_bar_weights(expected_pnls, weights, names):
    fig, ax = plt.subplots()

    bottom = np.zeros(len(expected_pnls))
    labels = [f"{exp*100:.0f}" + "%" for exp in expected_pnls]

    for i, ow in enumerate(weights.T):
        ax.bar(labels, ow, bottom=bottom, label=names[i], alpha=0.5)
        bottom += ow
    ax.legend(loc=4)

    make_grid(axis=ax)


def plot_line_weights(expected_pnls, weights, names):
    fig, ax = plt.subplots()
    for i, w in enumerate(weights.T):
        ax.plot(expected_pnls, w, label=names[i])
    make_grid(axis=ax)
    ax.legend()

