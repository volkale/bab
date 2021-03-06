import numpy as np

from bab.mcmc import get_mcmc, get_stan_model
from bab.make_data import make_data
from bab.power import get_power
from bab.model import BayesAB


y1, y2 = make_data(0, 1, 1, 2, 10, percent_outliers=0, sd_outlier_factor=2.0, rand_seed=1)
stan_model = get_stan_model()
mcmc = get_mcmc(stan_model, y1, y2)

get_power(stan_model, y1, y2, (-1., 1.), (0., 1.), 1., 1., n_sim=10)

model = BayesAB()
model.fit(2 * np.random.randn(10) + 3, np.random.randn(10))
