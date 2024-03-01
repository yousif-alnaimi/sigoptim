# How to set up environment

Create a new conda environment, with some starter packages by running `conda create -n m4r_signatory python=3.11 scipy seaborn jupyter pandas numba` and activate the environment.
Then install torch with gpu by using `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`.
Then install pip-only libraries with `pip install mogptk==0.3.5 tqdm`.
Then install signatory from source with:

```
git clone https://github.com/patrick-kidger/signatory.git
cd signatory
python setup.py install
```
