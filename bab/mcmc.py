import logging
import matplotlib.pyplot as plt
import numpy as np
import os

from bab.utils import stan_model_cache

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel('INFO')

plt.style.use('ggplot')


def get_mcmc(stan_model, y1, y2, prior_hyper_params=None, warmup=1000, rand_seed=None):
    """
    :param stan_model: StanModel instance
    :param y1: iterable, data from group 1 (test/active group)
    :param y2: iterable, data from group 2 (control/placebo group)
    :param prior_hyper_params: dict (optional), with keys 'muM', 'muP', 'sigmaLow', 'sigmaHigh'
    :param warmup: int (optional), number of burnin samples to use, defaults to 1000
    :param rand_seed: int (optional), random seed
    :returns: StanFit4Model instance containing the fitted results
    """

    if rand_seed is not None:
        np.random.seed(int(rand_seed))

    input_data = get_model_input(y1, y2, prior_hyper_params)

    mcmc = stan_model.sampling(
        data=input_data,
        iter=warmup + 1000,
        warmup=warmup,
        chains=3,
        control={'adapt_delta': 0.99},
        check_hmc_diagnostics=True,
        seed=rand_seed
    )

    return mcmc


def get_stan_model():
    """
    :returns: compiled StanModel instance
    """
    model_path = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(model_path, 'model.stan')
    stan_model = stan_model_cache(model_file)
    return stan_model


def get_model_input(y1, y2, prior_hyper_params):

    params = prior_hyper_params if prior_hyper_params is not None else {}

    try:
        assert set(params) in (set(), {'muM', 'muP', 'sigmaLow', 'sigmaHigh'})
    except AssertionError:
        logging.error('''
        prior_hyper_params must either specify all hyper parameters: 'muM', 'muP', 'sigmaLow', 'sigmaHigh'
        or none, in which case default priors are used.
        ''', exc_info=True)

    nt = len(y1)
    nc = len(y2)

    y = np.concatenate([y1, y2])
    mu_m = np.mean(y)
    s_y = np.sqrt(((nt - 1) * np.var(y1, ddof=1) + (nc - 1) * np.var(y2, ddof=1)) / (nt + nc - 2))

    input_data = {
        'N1': nt,
        'N2': nc,
        'y1': y1,
        'y2': y2,
        'muM': params.get('muM', mu_m),
        'muP': params.get('muP', 100 * s_y),
        'sigmaLow': params.get('sigmaLow', s_y / 1000),
        'sigmaHigh': params.get('sigmaHigh', s_y * 1000),
        'run_estimation': 1
    }

    return input_data


def plot_posteriors(mcmc, parameter):
    assert parameter in ('mu', 'sigma', 'nu')

    if not parameter == 'nu':
        _ = plt.hist(mcmc.extract()[parameter][:, 0], bins=100, density=True, alpha=0.5)  # NOQA
        _ = plt.hist(mcmc.extract()[parameter][:, 1], bins=100, density=True, alpha=0.5)  # NOQA
    else:
        _ = plt.hist(mcmc.extract()['nu'], bins=100, density=True, alpha=0.5)  # NOQA
