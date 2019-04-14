from bab.mcmc import get_mcmc, plot_posteriors
from bab.make_data import make_data


yt, yc = data = make_data(0, 1, 1, 2, 10, percent_outliers=0, sd_outlier_mfactor=2.0, rand_seed=1)
mcmc = get_mcmc(yt, yc)
plot_posteriors(mcmc, 'mu')
