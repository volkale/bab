# bab

Bayesian estimation for two groups (e.g. A/B testing and Randomized Experiments).

Based on the paper

    John K. Kruschke, Bayesian estimation supersedes the t test., J Exp Psychol Gen. 2013

and the corresponding R-package by its author.
We refer to Kruschke's website on [BEST](http://www.indiana.edu/~kruschke/BEST/) for more information.

This is a Python 3 library using [PyStan](https://pystan.readthedocs.io/en/latest/).

Cf. also Andrew Straw's Python version of the BEST package: [link](https://github.com/strawlab/best),
which uses [PyMC](https://github.com/pymc-devs/pymc) for MCMC sampling.

## Setup

Create `conda` environment:

    $ conda create --name bab python=3.7.2 -y

Activate the conda environment:

    $ source activate bab

Clone `bab` repository to current directory:

    $ git clone git@github.com:volkale/bab.git

Install all requirements:

    $ conda install --file requirements.txt --yes

Now install the `bab` package:

    $ python setup.py develop
