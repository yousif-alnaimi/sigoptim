# Enhancing Signature Trading: Strategies for Improved Performance and Rigour

This is the GitHub Repository that accompanies the paper titled "Enhancing Signature Trading: Strategies for Improved Performance and Rigour", submitted for the M4R module at Imperial College London.

## How to set up the environment

To set up the environment, follow the following steps:
- Create a new conda environment, with some starter packages by running `conda create -n ENV_NAME python=3.11 scipy seaborn jupyter pandas numba` and activate the environment.
- Then install torch with gpu by using `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`.
- Then install pip-only libraries with `pip install mogptk==0.3.5 tqdm iisignature`.
- Then install signatory from source with:

```
git clone https://github.com/patrick-kidger/signatory.git
cd signatory
python setup.py install
```

## Obtaining results

Please note that for the below instructions, all scripts must be run from the `optimisation` directory, as otherwise the paths will not work.

### Hoff Lead Lag Plots

To obtain this plot, simply run the `hoff_plot.py` script. The plot will be written as a `png` file to the `plots` directory.

### Sig Dimensionality Plots

Similarly to the above, simply run the `sig_dimensionality.py` script. The plot will be written as a `png` file to the `plots` directory.

### Signature Trading Model

This is the main implementation concerning the paper. The `combine_all` function is the main function used to find the efficient frontier of a particular set of stocks. It accepts many flags, the most important of which are as follows:
- `stocks` is a list of the stock tickers as strings you would like to simulate.
- `level` is an integer specifying the level of the Sig Trading Model. This defaults to 2 and other numbers are not tested due to computational constraints.
- `start_date` and `end_date` control the time period being tested. These are strings specifying dates as "YYYY-MM-DD".
- `include_ffr` is a boolean flag to determine whether or not to include the bank account.
- `frontier_interval` is a tuple indicating the range of expected returns to find the efficient frontier with.

The plot making the comparison is written after this function. It can be commented out if all four combinations are not being tested.
