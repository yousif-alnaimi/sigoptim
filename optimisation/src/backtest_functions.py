import numpy as np


def backtest_weights(weights, prices, capital=1e6, method="static", normalize=True):
    # Weights are T - 1 x N matrix
    # Prices are T x N

    T, N         = prices.shape
    p_value      = np.zeros(T)
    cash         = np.zeros(T)
    asset_values = np.zeros((T, N))
    n_shares     = np.zeros((T, N))

    if normalize:
        weights /= weights.sum(axis=1).reshape(-1, 1)

    if method == "static":

        capital_alloc    = capital*weights[0]
        n_shares         = np.floor(capital_alloc/prices[0])

        asset_values     = n_shares*prices

        cash[-1]         = np.sum(asset_values[-1].copy())
        n_shares[-1]     = 0.
        asset_values[-1] = 0.

        p_value          = cash + np.sum(asset_values, axis=1)

    elif method == "dynamic":

        for i, (w, p) in enumerate(zip(weights, prices[:-1])):
            port_value    = cash[i-1] + np.sum(n_shares[i-1]*p) if i != 0 else capital
            capital_alloc = port_value*w
            n_shares_     = np.floor(capital_alloc /p)

            n_shares[i]     = n_shares_
            asset_values[i] = n_shares_*p

            cash[i]         = port_value - np.sum(n_shares_*p)

        # Round out portfolio
        asset_values[-1] = n_shares[-2]*prices[-1]

        cash[-1]         = cash[-2] + np.sum(asset_values[-1].copy())
        n_shares[-1]     = 0
        asset_values[-1] = 0
        p_value          = cash + np.sum(asset_values, axis=1)

    return p_value, asset_values, cash, n_shares
