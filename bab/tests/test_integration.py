import arviz as az
from bab.mcmc import get_mcmc, get_stan_model
from bab.make_data import make_data


y1, y2 = data = make_data(0, 1, 1, 2, 10, percent_outliers=0, sd_outlier_factor=2.0, rand_seed=1)
stan_model = get_stan_model()
mcmc = get_mcmc(stan_model, y1, y2)


data = az.from_pystan(
    posterior=mcmc,
    posterior_predictive=['y1_pred', 'y2_pred'],
    observed_data=['y1', 'y2'],
    coords={
      'group_mu': ['Group 1', 'Group 2'],
      'group_sigma': ['Group 1', 'Group 2']
    },
    dims={
      'mu': ['group_mu'], 'sigma': ['group_sigma']
    }
)

az.plot_pair(data, var_names=['mu', 'sigma', 'log_nu'])
