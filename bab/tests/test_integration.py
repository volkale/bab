from bab.mcmc import get_mcmc, get_stan_model
from bab.make_data import make_data


y1, y2 = data = make_data(0, 1, 1, 2, 10, percent_outliers=0, sd_outlier_factor=2.0, rand_seed=1)
stan_model = get_stan_model()
mcmc = get_mcmc(stan_model, y1, y2)
