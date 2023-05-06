import sigkernel
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import os.path

base_path = os.getcwd()
import time
import itertools
# import pickle
# import math
# import torchcde

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import yfinance as yf
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mtick
import seaborn as sns


class SigKernelTrader:
    def __init__(self, dim: int, device: str = 'cpu', max_batch: int = 50, mult=1.0):
        """
        Sig-Kernel Trading Strategy

        Given a batch of sample paths the SigKernelTrader fits an optimal trading strategy with respect to the mean-variance criterion

        Once the strategy is fit, we can calculate the pnl for any given path

        Parameters
        ----------
        dim: int
            dimension of the incoming path / time series
        device: str
            device on which optimisation is performed. default: 'cpu'
        risk_aversion: float
            to be scaled according to the level of risk the investor is willing to take
        max_batch: int
            for when computing the signature kernel
        """

        self.d = dim
        self.device = device
        self.mult = mult

        self.max_batch = max_batch

        self.K = None
        self.dyadic_order = None
        self.sample_path = None
        self.sample_batches = None

        self.risk_aversion = None
        self.regularisation = None
        self.NN_Omega = None
        self.Omega = None
        self.N_regulariser = None
        self.regulariser = None
        self.alpha = None

    def compute_K(self, sample_path: torch.Tensor, verbose: bool = False,
                  dyadic_order=0, batch_length: int = 5):

        start = None
        if verbose:
            start = time.time()

        self.dyadic_order = dyadic_order
        self.sample_path = sample_path.to(self.device)
        self.sample_path = self.sample_path / self.sample_path[0]  # Path normalization

        # Create all X_i batches : [batch_x, batch_length, dim]
        # where batch_x = sample_path.shape[0] - batch_length + 1
        self.sample_batches = (sample_path.unfold(dimension=0, size=batch_length, step=1).swapaxes(1, 2))
        self.sample_batches = self.sample_batches.type(torch.float64)
        self.sample_batches = self.sample_batches / self.sample_batches[:, 0:1, :]  # Path normalization
        self.sample_batches = self.mult * self.sample_batches

        # Initialise sig-kernel
        static_kernel = sigkernel.LinearKernel()
        signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)

        # Compute gram matrix of sample paths K: [batch_x, batch_x]
        K = signature_kernel.compute_Gram(self.sample_batches, self.sample_batches,
                                          sym=True, max_batch=self.max_batch)
        self.K = K[:, :, -1, -1]

        if verbose:
            print('Gram Matrix Obtained: %s' % (time.time() - start))

    def fit(self,
            sample_path: torch.Tensor,
            risk_aversion=0,
            reg_type: str = 'RKHS',
            verbose: bool = False,
            dyadic_order=0,
            batch_length: int = 5,
            regularisation=0,
            NVars: bool = False,
            K_precomputed: bool = False):
        """
        Calibrate the strategy. For calibration the sample size should be as large as possible to accurately approximate the empirical measure.
        For real data a rolling window operation could be used to artificially increase the sample size.

        Parameters
        ----------
        sample_path: torch.Tensor
            incoming tensor of dimension [path length, dim]
            this is one path, to be batched up during the fitting process
        verbose: bool
            if True prints computation time for performing operations
        dyadic_order: int
            for SigKer
        batch_length: int
            the sample path will be split up into smaller paths of length batch_length
            total sample batches (batch_x) = sample_path.shape[0] - batch_length + 1
        reg_type: str = 'RKHS' or 'L2'
            user will input which type of regularisation they want, either RKHS penalisation or L2 norm
            default is 'RKHS'
        regularisation: float > 0
            the large the regularisation, the smaller the alpha's become and more stable the strategy
            often 10**(-3) to 10**(-10) is sensible range
        NVars : bool
            If True then we divide by N only at the end of computation
        K_precomputed: bool
            Has the SigKer Gram been already computed? Time saver

        Returns
        -------
        None

        """

        start = None
        self.regularisation = regularisation
        self.risk_aversion = risk_aversion

        if not K_precomputed:
            self.compute_K(sample_path, verbose, dyadic_order, batch_length)

        """
        Computing The Alpha's
        """

        if verbose:
            start = time.time()

        if NVars:

            # N_Xi = N * Xi
            # Omega = 1/N^2 * N_Xi @ I_mat @ N_Xi
            # alpha = 1/2 * (lambda * (N/N-1) * Omega + reg)^-1 @ (1/N) * N_Xi
            #       = 1/2 * (lambda * (1/N*N-1) * N_Xi @ I_mat @ N_Xi + reg)^-1 @ (1/N) * N_Xi_1
            #       = 1/2 * (lambda * (1/N-1) * N_Xi @ I_mat @ N_Xi + N * reg)^-1 @ N_Xi_1
            #       = (N-1)/2 * (lambda * N_Xi @ I_mat @ N_Xi + 1/(N-1) * N_reg)^-1 @ N_Xi_1

            N = self.K.shape[0]
            N_Xi = (self.K - torch.ones(N, N).to(self.device)).type(torch.float64)
            N_Xi_1 = N_Xi @ torch.ones(N).to(self.device).type(torch.float64)
            I_mat = (torch.eye(N) - 1 / N * torch.ones(N, N)).to(self.device).type(torch.float64)
            NN_Omega = N_Xi @ I_mat @ N_Xi
            self.NN_Omega = NN_Omega

            N_regulariser = torch.zeros(N, N)
            # Penalization in RKHS
            if reg_type == 'RKHS':
                N_regulariser = self.regularisation * N_Xi
            # Penalisation in L2 norm
            if reg_type == 'L2':
                N_regulariser = N * self.regularisation * torch.eye(N).to(self.device)

            self.N_regulariser = N_regulariser

            # Penalisation in RKHS Norm
            self.alpha = ((N - 1) / 2) * torch.inverse(
                self.risk_aversion * NN_Omega + (1 / (N - 1)) * N_regulariser) @ N_Xi_1

        else:

            N = self.K.shape[0]
            Xi = (1 / N) * (self.K.to(self.device) - torch.ones(N, N).to(self.device)).type(torch.float64)
            Xi_1 = Xi @ torch.ones(N).to(self.device).type(torch.float64)
            I_mat = (torch.eye(N) - 1 / N * torch.ones(N, N)).to(self.device).type(torch.float64)
            Omega = Xi @ I_mat @ Xi
            self.Omega = Omega

            regulariser = torch.zeros(N, N)
            # Penalization in RKHS
            if reg_type == 'RKHS':
                regulariser = self.regularisation * Xi
            # Penalisation in L2 norm
            if reg_type == 'L2':
                regulariser = self.regularisation * torch.eye(N).to(self.device)

            self.regulariser = regulariser

            # Penalisation in RKHS Norm
            self.alpha = (1 / 2) * torch.inverse(self.risk_aversion * (N / (N - 1)) * Omega + regulariser) @ Xi_1

        if verbose:
            print('Alpha Obtained: %s' % (time.time() - start))

        self.alpha = self.alpha.to(self.device)

    def compute_pnl(self,
                    path: torch.Tensor,
                    lookback_window: int = 5,
                    sample: bool = False,
                    soft_leverage='None',
                    hard_leverage='None',
                    verbose: bool = False):
        """
        For a given path, we can compute the PnL with respect to the fitted strategy

        Parameters
        ----------
        path: torch.Tensor
            incoming tensor of dimension [path length, dim]
            this is one path, to be traded using fitted strategy
        lookback_window: int
            the sample path will be split up into smaller paths of length lookback_window
            total test batches (batch_y) = path.shape[0] - lookback_window + 1
            these test batches (Y_i) will be compared to sample batches (X_j) via the sig-kernel
        sample: bool
            if you want to compute the PnL in sample as well
        soft_leverage:
            a scaling factor according to the gross leverage of the portfolio
            default = 'None', user can change to float (e.g 3.0)
        hard_leverage:
            a hard cap on the gross leverage of the portfolio
            default = 'None', user can change to float (e.g 3.0)
        verbose: bool
            if True prints computation time for performing operations

        Returns
        -------
        None

        """

        self.soft_leverage = soft_leverage
        self.hard_leverage = hard_leverage

        if verbose:
            start = time.time()

        path = path / path[0]  # normalise
        self.oos_path = path

        # Rollings paths in the out of sample phase
        self.oos_batches = path.unfold(dimension=0, size=lookback_window, step=1).swapaxes(1, 2)
        self.oos_batches = self.oos_batches / self.oos_batches[:, 0:1, :]

        # Initialize kernel
        static_kernel = sigkernel.LinearKernel()
        signature_kernel = sigkernel.SigKernel(static_kernel, self.dyadic_order)

        """
        Compute rolling gram matrix of test batches against sample batches
        K_oos: [batch_y, batch_x, lookback_window, batch_length]
        Computes the integral of sigkernel(past lookback_window path, X_i) dX_i for each batch X_i
        """

        # G_mat : [batch_y, batch_x, times_y, times_x]
        G_mat = signature_kernel.compute_Gram(self.oos_batches,
                                              self.sample_batches,
                                              sym=False,
                                              max_batch=self.max_batch)

        # We care only about K_{1,s}(y,x) for s < 1
        G_mat = G_mat[:, :, -1, ::2 ** self.dyadic_order][:, :, :-1]  # G_mat : [batch_y, batch_x, times_x - 1]
        self.K_oos = G_mat.unsqueeze(-2).unsqueeze(-2)  # K_oos : [batch_y, batch_x, 1, 1, times_x - 1]

        # dx :[1, batch_x, dim, timesteps_x-1, 1]
        self.dx = torch.diff(self.sample_batches, dim=1).swapaxes(-1, -2).unsqueeze(0).unsqueeze(-1)

        self.nu_x = (self.K_oos @ self.dx)  # nu_x : [batch_y, batch_x, d, 1, 1]
        self.nu_x = self.nu_x.flatten(start_dim=2)  # nu_x : [batch_y, batch_x, d]

        # Compute position for each time t in the test path, position : [batch_y, dim]
        position = (self.alpha.unsqueeze(0).unsqueeze(2) * self.nu_x).mean(dim=1)

        """
        Impose scaling and restrictions on the position to ensure suitable positions and leverage constraints
        """
        position = position / abs(position).sum(dim=1).mean()
        self.position = position  # position : [batch_y, dim]

        if soft_leverage == 'None':
            pass

        else:
            self.position = (self.position * ((torch.clamp(torch.abs(self.position).sum(dim=1),
                                                           max=self.soft_leverage) / torch.abs(self.position).sum(
                dim=1)).unsqueeze(1)))

        if hard_leverage == 'None':
            pass

        else:
            self.position = self.position * ((torch.clamp(torch.abs(self.position).sum(dim=1),
                                                          max=self.hard_leverage)) / (
                                                 torch.abs(self.position).sum(dim=1))).unsqueeze(1)

        """
        compute PnL over the whole path
        """
        pnl = (self.position[:-1] * torch.diff(path[-self.position.shape[0]:], dim=0)).cumsum(0).sum(dim=1) + 1
        self.pnl = pnl

        if verbose:
            print('Test PnL Obtained: %s' % (time.time() - start))

        """
        Repeat the whole process for the sample path if user set sample == True
        """
        if sample:

            static_kernel = sigkernel.LinearKernel()
            signature_kernel = sigkernel.SigKernel(static_kernel, self.dyadic_order)

            # G_mat_ : [batch_x, batch_x, times_x, times_x]
            G_mat_ = signature_kernel.compute_Gram(self.sample_batches,
                                                   self.sample_batches,
                                                   sym=False,
                                                   max_batch=self.max_batch)

            # We care only about K_{1,s}(y,x) for s < 1
            G_mat_ = G_mat_[:, :, -1, ::2 ** self.dyadic_order][:, :, :-1]  # G_mat_ : [batch_x, batch_x_, times_x - 1]
            self.K_sample = G_mat_.unsqueeze(-2).unsqueeze(-2)  # K_oos : [batch_y, batch_x, 1, 1, times_x - 1]

            # sample_dx : [1, batch_x, d, timesteps_x-1, 1]
            self.sample_dx = torch.diff(self.sample_batches, dim=1).swapaxes(-1, -2).unsqueeze(0).unsqueeze(-1)

            # sample_nu_x : [batch_y, batch_x, d]
            self.sample_nu_x = (self.K_sample @ self.sample_dx).flatten(start_dim=2)

            sample_position = (self.alpha.to(self.device).unsqueeze(0).unsqueeze(2) * self.sample_nu_x).mean(dim=1)
            sample_position = sample_position / abs(sample_position).sum(dim=1).mean()
            self.sample_position = sample_position

            if soft_leverage == 'None':
                pass

            else:
                self.sample_position = (self.sample_position * ((torch.clamp(torch.abs(self.sample_position).sum(dim=1),
                                                                             max=self.soft_leverage) / torch.abs(
                    self.sample_position).sum(dim=1)).unsqueeze(1)))

            if hard_leverage == 'None':
                pass

            else:
                self.sample_position = self.sample_position * (
                            (torch.clamp(torch.abs(self.sample_position).sum(dim=1), max=self.hard_leverage)) / (
                        torch.abs(self.sample_position).sum(dim=1))).unsqueeze(1)

            pnl = (self.sample_position[:-1] * torch.diff(self.sample_path[-self.sample_position.shape[0]:],
                                                          dim=0)).cumsum(0).sum(dim=1) + 1
            self.sample_pnl = pnl

            if verbose:
                print('Sample PnL Obtained: ' % (time.time() - start))

    def compute_portfolio_stats(self, sample: bool = False):

        """
        used to compute some basic stats of the portfolio as a performance measure

        sample: bool
            if you want to compute statistics on in sample PnL too
        """
        self.test_vol = (torch.diff(self.pnl, n=1, dim=0)).std(dim=0) * torch.sqrt(torch.tensor(252.0))
        self.test_ann_return = (torch.diff(self.pnl, n=1, dim=0)).mean(dim=0) * torch.sqrt(torch.tensor(252.0))
        self.test_sharpe = ((torch.diff(self.pnl, n=1, dim=0)).mean(dim=0) / (torch.diff(self.pnl, n=1, dim=0)).std(
            dim=0)) * torch.sqrt(torch.tensor(252.0))

        if sample == True:
            self.sample_vol = (torch.diff(self.sample_pnl, n=1, dim=0)).std(dim=0) * torch.sqrt(torch.tensor(252.0))
            self.sample_ann_return = (torch.diff(self.sample_pnl, n=1, dim=0)).mean(dim=0) * torch.sqrt(
                torch.tensor(252.0))
            self.sample_sharpe = ((torch.diff(self.sample_pnl, n=1, dim=0)).mean(dim=0) / (
                torch.diff(self.sample_pnl, n=1, dim=0)).std(dim=0)) * torch.sqrt(torch.tensor(252.0))

    def get_portfolio_stats(self, sample: bool = False):

        """
        computes and prints portfolio statistics
        """

        self.compute_portfolio_stats(sample)

        if sample == True:

            print('Sample Return (Annualised): ' + "{:.2f}".format(self.sample_ann_return))
            print('Test Return (Annualised): ' + "{:.2f}".format(self.test_ann_return))
            print('Sample Vol: ' + "{:.2f}".format(self.sample_vol))
            print('Test Vol: ' + "{:.2f}".format(self.test_vol))
            print('Sample Sharpe: ' + "{:.2f}".format(self.sample_sharpe))
            print('Test Sharpe: ' + "{:.2f}".format(self.test_sharpe))

        else:
            print('Test Return (Annualised): ' + "{:.2f}".format(self.test_ann_return))
            print('Test Vol: ' + "{:.2f}".format(self.test_vol))
            print('Test Sharpe: ' + "{:.2f}".format(self.test_sharpe))

    def plot_strategies(self, sample_test: str, title: str, lw: float = .7, alpha: float = 1, vol_scale='None'):

        """
        plot functionality for PnL curve

        sample_test: str = 'test', 'sample' or 'both'
            user can plot any of the above PnL curves
        """

        if vol_scale != 'None':
            portfolio_sample_scaled = torch.cat(
                [torch.ones(1), 1 + ((vol_scale / self.sample_vol) * torch.diff(self.sample_pnl, dim=0)).cumsum(0)],
                dim=0)
            portfolio_test_scaled = torch.cat(
                [torch.ones(1), 1 + ((vol_scale / self.test_vol) * torch.diff(self.sample_pnl, dim=0)).cumsum(0)],
                dim=0)

        if sample_test == 'test':

            cmap = ListedColormap(
                [sns.color_palette("rocket")[4], sns.color_palette("rocket")[3], sns.color_palette("rocket")[2],
                 sns.color_palette("rocket")[1]])
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cmap.colors)

            labels = ["Sig-Kernel Trader"]

            alpha = 1
            lw = 2

            fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
            if vol_scale == 'None':
                ax1.plot((self.pnl / self.pnl[0]).detach().cpu(), alpha=alpha, lw=lw);

            if vol_scale != 'None':
                ax1.plot((portfolio_test_scaled / portfolio_test_scaled[0]).detach().cpu(), alpha=alpha, lw=lw);

            ax1.tick_params(axis='y', labelsize=9)
            ax1.set_title(' ', fontsize=20)
            ax1.yaxis.label.set_size(15)
            ax1.xaxis.label.set_size(15)
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
            ax1.set_yticklabels(['{:.0%}'.format(x) for x in vals])
            ax1.grid(alpha=0.4, linewidth=.7)
            ax1.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))
            plt.legend(labels, loc='best', fontsize=12)
            fig.tight_layout()
            plt.show()

        if sample_test == 'sample':

            cmap = ListedColormap(
                [sns.color_palette("rocket")[4], sns.color_palette("rocket")[3], sns.color_palette("rocket")[2],
                 sns.color_palette("rocket")[1]])
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cmap.colors)

            labels = ["Sig-Kernel Trader"]

            alpha = 1
            lw = 2

            fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
            if vol_scale == 'None':
                ax1.plot((self.sample_pnl / self.sample_pnl[0]).detach().cpu(), alpha=alpha, lw=lw);

            if vol_scale != 'None':
                ax1.plot((portfolio_sample_scaled / portfolio_sample_scaled[0]).detach().cpu(), alpha=alpha, lw=lw);
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
            ax1.set_yticklabels(['{:.0%}'.format(x) for x in vals])
            # vals = ax1.get_xticks()
            # ax1.set_xticklabels(['{:.0%}'.format(x) for x in vals])
            ax1.grid(alpha=0.4, linewidth=.7)
            ax1.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))
            plt.legend(labels, loc='best', fontsize=12)
            fig.tight_layout()
            plt.show()

        if sample_test == 'both':

            cmap = ListedColormap(
                [sns.color_palette("rocket")[4], sns.color_palette("rocket")[3], sns.color_palette("rocket")[2],
                 sns.color_palette("rocket")[1]])
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cmap.colors)

            labels = ["Sig-Kernel Trader"]

            alpha = 1
            lw = 2

            fig, ax = plt.subplots(1, 2, figsize=(18, 6))
            if vol_scale == 'None':
                ax[0].plot((self.sample_pnl / self.sample_pnl[0]).detach().cpu(), alpha=alpha, lw=lw);

            if vol_scale != 'None':
                ax[0].plot((portfolio_sample_scaled / portfolio_sample_scaled[0]).detach().cpu(), alpha=alpha, lw=lw);

            ax[0].tick_params(axis='y', labelsize=9)
            ax[0].set_title(' ', fontsize=20)
            ax[0].yaxis.label.set_size(15)
            ax[0].xaxis.label.set_size(15)
            right_side = ax[0].spines["right"]
            right_side.set_visible(False)
            top_side = ax[0].spines["top"]
            top_side.set_visible(False)
            left_side = ax[0].spines["left"]
            left_side.set_visible(False)
            bottom_side = ax[0].spines["bottom"]
            bottom_side.set_visible(False)
            ax[0].tick_params(axis='x', labelsize=15)
            ax[0].tick_params(axis='y', labelsize=15)
            vals = ax[0].get_yticks()
            ax[0].set_yticklabels(['{:.0%}'.format(x) for x in vals])
            ax[0].grid(alpha=0.4, linewidth=.7)
            ax[0].xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(ax[0].xaxis.get_major_locator()))
            ax[0].legend(labels, loc='best', fontsize=12)

            if vol_scale == 'None':
                ax[1].plot((self.pnl / self.pnl[0]).detach().cpu(), alpha=alpha, lw=lw);

            if vol_scale != 'None':
                ax[1].plot((portfolio_test_scaled / portfolio_test_scaled[0]).detach().cpu(), alpha=alpha, lw=lw);

            ax[1].tick_params(axis='y', labelsize=9)
            ax[1].set_title(' ', fontsize=20)
            ax[1].yaxis.label.set_size(15)
            ax[1].xaxis.label.set_size(15)
            right_side = ax[1].spines["right"]
            right_side.set_visible(False)
            top_side = ax[1].spines["top"]
            top_side.set_visible(False)
            left_side = ax[1].spines["left"]
            left_side.set_visible(False)
            bottom_side = ax[1].spines["bottom"]
            bottom_side.set_visible(False)
            ax[1].tick_params(axis='x', labelsize=15)
            ax[1].tick_params(axis='y', labelsize=15)
            vals = ax[1].get_yticks()
            ax[1].set_yticklabels(['{:.0%}'.format(x) for x in vals])
            ax[1].grid(alpha=0.4, linewidth=.7)
            ax[1].xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(ax[1].xaxis.get_major_locator()))
            ax[1].legend(labels, loc='best', fontsize=12)
            fig.tight_layout()

            plt.show()