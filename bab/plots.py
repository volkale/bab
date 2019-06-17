import matplotlib.pyplot as plt
import arviz as az


def get_pairs(mcmc):

    data = az.from_pystan(
        posterior=mcmc,
        posterior_predictive=['y1_pred', 'y2_pred'],
        observed_data=['y1', 'y2'],
        log_likelihood='log_lik',
        coords={
            'group_mu': ['Group 1', 'Group 2'],
            'group_sigma': ['Group 1', 'Group 2']
        },
        dims={
            'mu': ['group_mu'], 'sigma': ['group_sigma']
        }
    )

    _ = az.plot_pair(data, var_names=['mu', 'sigma', 'log_nu'], figsize=(10, 10))
    plt.show()


def get_forest(mcmc):

    data = az.from_pystan(
        posterior=mcmc,
        posterior_predictive=['y1_pred', 'y2_pred'],
        observed_data=['y1', 'y2'],
        log_likelihood='log_lik',
        coords={
            'group_mu': ['Group 1', 'Group 2'],
            'group_sigma': ['Group 1', 'Group 2']
        },
        dims={
            'mu': ['group_mu'], 'sigma': ['group_sigma']
        }
    )

    _ = az.plot_forest(
        data,
        var_names=['mu', 'sigma', 'log_nu'],
        credible_interval=0.95,
        figsize=(10, 10)
    )
    plt.show()
