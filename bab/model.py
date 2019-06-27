import matplotlib.pyplot as plt
import numpy as np
import arviz as az
from bab.mcmc import get_stan_model, get_mcmc, plot_posteriors


class BayesAB:
    """
    Bayesian AB model class
    """

    def __init__(self, **kwargs):
        self.model = get_stan_model()
        self.mcmc_ = None
        self.data_ = None
        self.params = ('mu', 'sigma', 'nu')
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(self, y1, y2):
        self.mcmc_ = get_mcmc(self.model, y1, y2, **self.kwargs)

        self.data_ = az.from_pystan(
            posterior=self.mcmc_,
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

    def plot_posteriors(self, parameter=None, **kwargs):
        if not self.mcmc_:
            raise AttributeError('Object needs to be fit first.')
        if parameter:
            plot_posteriors(self.mcmc_, parameter)
        else:
            kwargs['rope'] = kwargs.get('rope', {})
            kwargs['ref_val'] = kwargs.get('ref_val', {})

            sample_dict = self.mcmc_.extract()
            _, ax = plt.subplots(3, 3, figsize=(21, 14))

            for i, param in enumerate(['mu', 'sigma']):

                self._add_posterior_plot(sample_dict, 'mu', ax[0, i], idx=i)
                ax[0, i].set_title('$\mu_{}$'.format(i+1))

                self._add_posterior_plot(sample_dict, 'sigma', ax[1, i], idx=i)
                ax[1, i].set_title('$\sigma_{}$'.format(i+1))

                _ = az.plot_posterior(  # NOQA
                    sample_dict[param][:, 0] - sample_dict[param][:, 1],
                    credible_interval=0.95,
                    ax=ax[i, 2],
                    ref_val=kwargs['ref_val'].get(param),
                    rope=kwargs['rope'].get(param)
                )

            _ = ax[0, 2].set_title('$\mu_1 - \mu_2$')
            _ = ax[1, 2].set_title('$\sigma_1 - \sigma_2$')

            self._add_posterior_plot(sample_dict, 'nu', ax[2, 0])
            _ = ax[2, 0].set_title("$\\nu$")

            self._add_posterior_plot(sample_dict, 'log_nu', ax[2, 1])
            _ = ax[2, 1].set_title("$\log(\\nu)$")

            effect_size = (sample_dict['mu'][:, 0] - sample_dict['mu'][:, 1]) / np.linalg.norm(
                sample_dict['sigma'],
                axis=1
            ) * np.sqrt(2)
            _ = az.plot_posterior(  # NOQA
                effect_size,
                credible_interval=0.95,
                ax=ax[2, 2]
            )
            _ = ax[2, 2].set_title("$(\mu_1-\mu_2)/\sqrt{(\sigma_1+\sigma_2)/2}$")
            plt.tight_layout(pad=4)
            plt.show()

    @staticmethod
    def _add_posterior_plot(sample_dict, param, ax_, idx=None):
        _ = az.plot_posterior(  # NOQA
            sample_dict[param] if idx is None else sample_dict[param][:, idx],
            credible_interval=0.95,
            ax=ax_
        )
