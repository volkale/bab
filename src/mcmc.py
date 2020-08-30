import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from src.utils import stan_model_cache

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel('INFO')

plt.style.use('ggplot')


def get_mcmc(stan_model, y1, y2, w1=None, w2=None, prior_hyper_params=None, warmup=1000, rand_seed=None):
    """
    :param stan_model: StanModel instance
    :param y1: iterable, data from group 1 (test/active group)
    :param y2: iterable, data from group 2 (control/placebo group)
    :param w1: iterable (optional), frequency of data from group 1 (test/active group)
    :param w2: iterable (optional), frequency of data from group 2 (control/placebo group)
    :param prior_hyper_params: dict (optional), with keys 'muM', 'muP', 'sigmaLow', 'sigmaHigh'
    :param warmup: int (optional), number of burnin samples to use, defaults to 1000
    :param rand_seed: int (optional), random seed
    :returns: StanFit4Model instance containing the fitted results
    """

    if rand_seed is not None:
        np.random.seed(int(rand_seed))

    input_data = get_model_input(y1, y2, w1, w2, prior_hyper_params)

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


def get_model_input(y1, y2, w1=None, w2=None, prior_hyper_params=None):

    params = prior_hyper_params if prior_hyper_params is not None else {}

    try:
        assert set(params) in (set(), {'muM', 'muP', 'sigmaLow', 'sigmaHigh'})
    except AssertionError:
        logging.error('''
        prior_hyper_params must either specify all hyper parameters: 'muM', 'muP', 'sigmaLow', 'sigmaHigh'
        or none, in which case default priors are used.
        ''', exc_info=True)

    if not w1:
        w1 = np.ones(len(y1))
    if not w2:
        w2 = np.ones(len(y2))

    nt = np.sum(w1)
    nc = np.sum(w2)

    df1 = pd.DataFrame(list(zip(y1, w1)), columns=['y1', 'w1']).groupby('y1').agg({'w1': 'sum'}).reset_index()
    y1 = df1.y1.values
    w1 = df1.w1.values.astype(int)
    N1 = df1.shape[0]
    del df1
    df2 = pd.DataFrame(list(zip(y2, w2)), columns=['y2', 'w2']).groupby('y2').agg({'w2': 'sum'}).reset_index()
    y2 = df2.y2.values
    w2 = df2.w2.values.astype(int)
    N2 = df2.shape[0]
    del df2

    y = np.concatenate([y1, y2])
    w = np.concatenate([w1, w2])
    mu_m = weighted_avg(y, w)
    s_y = np.sqrt(
        ((nt - 1) * weighted_var(y1, w1) + (nc - 1) * weighted_var(y2, w2)) / (nt + nc - 2)
    )

    input_data = {
        'N1': N1,
        'N2': N2,
        'y1': y1,
        'y2': y2,
        'w1': w1,
        'w2': w2,
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


def weighted_avg(values, weights):
    return np.average(values, weights=weights)


def weighted_var(values, weights):
    average = weighted_avg(values, weights)
    variance = np.average((values - average) ** 2, weights=weights) * (np.sum(weights) / (np.sum(weights) - 1))
    return variance
