import torch
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns


def portfolio_shift(position, underlyings, max_shift):

    diffs = torch.diff(underlyings, n=1, dim=0) # S_1 - S_0, ..., S_N - S_N-1

    portfolios = torch.Tensor([])
    total_shifts = 2*max_shift+1

    for shift in range(total_shifts):
        
        if shift == 0:
            current_portfolio = (diffs[:-total_shifts]*position[max_shift:-max_shift-1]).sum(dim=1).cumsum(0).unsqueeze(2)+1
        elif shift == total_shifts:
            current_portfolio = (diffs[shift:]*position[max_shift:-max_shift-1]).sum(dim=1).cumsum(0).unsqueeze(2)+1
        else:
            current_portfolio = (diffs[shift:-total_shifts+shift]*position[max_shift:-max_shift-1]).sum(dim=1).cumsum(0).unsqueeze(2)+1
    
        portfolios = torch.cat([portfolios, current_portfolio], dim=2)

    return portfolios

def alpha_shift(position, underlyings, max_shift):

    portfolios = portfolio_shift(position, underlyings, max_shift)

    sharpes = ((torch.diff(portfolios, n=1, dim=0)).mean(dim=0)/(torch.diff(portfolios, n=1, dim=0)).std(dim=0))*torch.sqrt(torch.tensor(252.0))

    return sharpes

def plot_alpha_decay(position, underlyings, max_shift):

    if position.shape[2]>1:

        sharpes = alpha_shift(position=position, underlyings=underlyings, max_shift=max_shift).swapaxes(0,1)
        labels = ["Markowitz", "Sig Order 0", "Sig Order 1", "Sig Order 2", "Sig Order 3"]
        sharpes = pd.DataFrame(sharpes, columns=labels[:sharpes.shape[1]])
        sharpes["lag"] = np.arange(2*max_shift+1)-max_shift
        cmap = ListedColormap([sns.color_palette("rocket")[4], sns.color_palette("rocket")[3], sns.color_palette("rocket")[2], sns.color_palette("rocket")[1]])
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cmap.colors)

        alpha=1
        width = 1

        fig, ax1 = plt.subplots(1,1, figsize=(8,5))
        sharpes.plot(x="lag", y=labels[:sharpes.shape[1]-1], kind='bar', align='center', alpha=alpha, ax=ax1, width = width)
        # ax1.xticks(np.arange(25)-12, sharpes)
        ax1.tick_params(axis='y', labelsize=9)
        ax1.set_title(' ', fontsize=20)
        ax1.yaxis.label.set_size(15)
        ax1.xaxis.label.set_size(15)
        # ax1.set_xticks(np.arange(0, 1.2, 0.1))
        right_side = ax1.spines["right"]
        right_side.set_visible(False)
        top_side = ax1.spines["top"]
        top_side.set_visible(False)
        left_side = ax1.spines["left"]
        left_side.set_visible(False)
        bottom_side = ax1.spines["bottom"]
        bottom_side.set_visible(False)
        ax1.tick_params(axis='x', labelsize=15)
        ax1.tick_params(axis='y', labelsize=15)
        vals = ax1.get_yticks()
        # ax1.set_yticklabels(['{:.0%}'.format(x) for x in vals])
        # vals = ax1.get_xticks()
        # ax1.set_xticklabels(['{:.0%}'.format(x) for x in vals])
        ax1.grid(alpha=0.4, linewidth=.7)
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_xlabel('Position Lag')
        fig.tight_layout()
        plt.show()
    
    else:

        sharpes = alpha_shift(position=position, underlyings=underlyings, max_shift=max_shift).swapaxes(0,1)
        labels = ["Strategy"]
        sharpes = pd.DataFrame(sharpes, columns=labels)
        sharpes["lag"] = np.arange(2*max_shift+1)-max_shift
        cmap = ListedColormap([sns.color_palette("rocket")[1]])
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cmap.colors)

        alpha=0.8
        width = 0.8

        fig, ax1 = plt.subplots(1,1, figsize=(8,5))
        sharpes.plot(x="lag", y=labels[:sharpes.shape[1]-1], kind='bar', align='center', alpha=alpha, ax=ax1, width = width)
        # ax1.xticks(np.arange(25)-12, sharpes)
        ax1.tick_params(axis='y', labelsize=9)
        ax1.set_title(' ', fontsize=20)
        ax1.yaxis.label.set_size(15)
        ax1.xaxis.label.set_size(15)
        # ax1.set_xticks(np.arange(0, 1.2, 0.1))
        right_side = ax1.spines["right"]
        right_side.set_visible(False)
        top_side = ax1.spines["top"]
        top_side.set_visible(False)
        left_side = ax1.spines["left"]
        left_side.set_visible(False)
        bottom_side = ax1.spines["bottom"]
        bottom_side.set_visible(False)
        ax1.tick_params(axis='x', labelsize=15)
        ax1.tick_params(axis='y', labelsize=15)
        vals = ax1.get_yticks()
        # ax1.set_yticklabels(['{:.0%}'.format(x) for x in vals])
        # vals = ax1.get_xticks()
        # ax1.set_xticklabels(['{:.0%}'.format(x) for x in vals])
        ax1.grid(alpha=0.4, linewidth=.7)
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_xlabel('Position Lag')
        fig.tight_layout()
        plt.show()