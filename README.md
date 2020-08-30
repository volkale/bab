# bab

[![PyPI version](https://badge.fury.io/py/bab.svg)](https://badge.fury.io/py/bab)
[![Build Status](https://travis-ci.com/volkale/bab.svg?token=9JgTwriYTrtamJ3cXPvS&branch=develop)](https://travis-ci.com/volkale/bab)
[![codecov](https://codecov.io/gh/volkale/bab/branch/develop/graph/badge.svg)](https://codecov.io/gh/volkale/bab)

Bayesian estimation for two groups (e.g. A/B testing and Randomized Experiments), assuming approximately
normally (or student-t) distributed outcome variables.

Based on the paper

   [John K. Kruschke, Bayesian estimation supersedes the t test., J Exp Psychol Gen. 2013](https://www.ncbi.nlm.nih.gov/pubmed/22774788)

and the corresponding R-package by its author.
We refer to [BEST](https://cran.r-project.org/web/packages/BEST/BEST.pdf) for more information.

This is a Python 3 library using [PyStan](https://pystan.readthedocs.io/en/latest/).

Cf. also [Andrew Straw](https://strawlab.org/)'s Python version of the BEST package: [link](https://github.com/strawlab/best),
which uses [PyMC](https://github.com/pymc-devs/pymc) for MCMC sampling.

## Setup

Create `conda` environment:

    $ conda create --name bab python=3.7.3 -y

Activate the conda environment:

    $ conda activate bab

For the installation of the `bab` package via pip just run:

    $ pip install bab


Alternatively, clone the `bab` repository to current directory:

    $ git clone git@github.com:volkale/bab.git

Then install the `bab` package by running:

    $ make install

This assumes that you have `make` installed on your system.

## Example
```python
import numpy as np
from bab.model import BayesAB

# generate some sample data
np.random.seed(1)
sample_size = 10
y1 = 2 * np.random.randn(sample_size) + 3
y2 = np.random.randn(sample_size)

# create a BayesAB model object
model = BayesAB(rand_seed=1)

# fit the model with the data
model.fit(y1, y2)

# visualize inference and model diagnostics
model.plot_all()

```