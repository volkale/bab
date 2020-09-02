import logging
import numpy as np
import pandas as pd
from scipy.stats import t, beta
from scipy.optimize import fmin

from bab.mcmc import get_mcmc

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel('INFO')


def get_power(stan_model, y1, y2, sample_size, rope_m, rope_sd, max_hdi_width_m, max_hdi_width_sd,
              cred_mass=0.95, n_sim=200, precision=2, rand_seed=None):
    """
    :param stan_model: StanModel instance
    :param y1: iterable, prospective samples of group one
    :param y2: iterable, prospective samples of group two
    :param sample_size: number of samples per group
    :param rope_m: iterable, of length two, such as (-1, 1),
        specifying the limit of the ROPE on the difference of means.
    :param rope_sd: iterable, of length two, such as (-1, 1),
        specifying the limit of the ROPE on the difference of standard deviations.
    :param max_hdi_width_m: float, maximum desired width of the 95% HDI on the difference of means.
    :param max_hdi_width_sd: float, maximum desired width of the 95% HDI on the difference of standard deviations.
    :param cred_mass: (optional) float, fraction of credible mass.
    :param n_sim: (optional) int, number of simulated experiments used to estimate the power.
    :param precision: (optional) int, number of decimals to round the power statistics to.
    :param rand_seed: int (optional), random seed
    :return: power, dict
    """
    if rand_seed is not None:
        np.random.seed(int(rand_seed))

    mcmc = get_mcmc(stan_model, y1, y2, rand_seed=rand_seed)
    mcmc_chain = mcmc.extract()

    chain_length = len(mcmc_chain['mu'][:, 0])  # same as len(mcmc_chain[:, 1])
    # Select thinned steps in chain for posterior predictions:
    step_idx = list(range(1, chain_length, int(chain_length / n_sim)))

    goal_tally = {
        'HDIm > ROPE': 0,
        'HDIm < ROPE': 0,
        'HDIm in ROPE': 0,
        'HDIm width < max': 0,
        'HDIsd > ROPE': 0,
        'HDIsd < ROPE': 0,
        'HDIsd in ROPE': 0,
        'HDIsd width < max': 0
    }

    power = {
        'HDIm > ROPE': [0, 0, 0],
        'HDIm < ROPE': [0, 0, 0],
        'HDIm in ROPE': [0, 0, 0],
        'HDIm width < max': [0, 0, 0],
        'HDIsd > ROPE': [0, 0, 0],
        'HDIsd < ROPE': [0, 0, 0],
        'HDIsd in ROPE': [0, 0, 0],
        'HDIsd width < max': [0, 0, 0]
    }

    n_sim = 0
    for step in step_idx:
        n_sim += 1

        y1_sim, y2_sim = _generate_simulated_data(mcmc_chain, sample_size, step)

        # Get posterior for simulated data:
        mcmc = get_mcmc(stan_model, y1_sim, y2_sim, rand_seed=rand_seed)  # tune input parameters
        sim_chain = mcmc.extract()

        goal_tally = _update_goal_tally(sim_chain, 'mu', goal_tally, rope_m, max_hdi_width_m)
        goal_tally = _update_goal_tally(sim_chain, 'sigma', goal_tally, rope_sd, max_hdi_width_sd)

        # Assess which goals were achieved and tally them:

        for k, v in goal_tally.items():
            a = 1 + v
            b = 1 + (n_sim - v)
            power[k][0] = a / (a + b)
            power[k][1:] = get_hdi_of_lcdf(beta, cred_mass=cred_mass, a=a, b=b)

        if n_sim % 100 == 0:
            _log_progress(n_sim, power, step_idx)

    for k, v in power.items():
        power[k] = [round(e, precision) for e in v]

    return power


def _generate_simulated_data(mcmc_chain, sample_size, step):
    # Get parameter values for this simulation:
    mu1_val = mcmc_chain['mu'][step, 0]
    mu2_val = mcmc_chain['mu'][step, 1]
    sigma1_val = mcmc_chain['sigma'][step, 0]
    sigma2_val = mcmc_chain['sigma'][step, 1]
    nu_val = mcmc_chain['nu'][step]
    # Generate simulated data:
    y1_sim = t.rvs(df=nu_val, loc=mu1_val, scale=sigma1_val, size=sample_size)
    y2_sim = t.rvs(df=nu_val, loc=mu2_val, scale=sigma2_val, size=sample_size)
    return y1_sim, y2_sim


def _log_progress(n_sim, power, step_idx):
    logging.info('Power after {} of {} simulations: '.format(n_sim, len(step_idx)))
    logging.info(pd.DataFrame(power, index=['mean', 'CrIlo', 'CrIhi']).T)


def get_hdi_of_lcdf(dist_name, cred_mass=0.95, **args):
    """
    Returns the HDI of a probability density function (form scipy.stats)
    """
    # freeze distribution with given arguments
    distri = dist_name(**args)
    # initial guess for hdi_low_tail_pr
    incred_mass = 1.0 - cred_mass

    def interval_width(low_tail_pr):
        return distri.ppf(cred_mass + low_tail_pr) - distri.ppf(low_tail_pr)

    # find low_tail_pr that minimizes interval width
    hdi_low_tail_pr = fmin(interval_width, incred_mass, ftol=1e-8, disp=False)[0]
    # return interval as array([low, high])
    return distri.ppf([hdi_low_tail_pr, cred_mass + hdi_low_tail_pr])


def _update_goal_tally(sim_chain, variable, goal_tally, ROPE, maxHDIW):
    hdim_l, hdim_r = get_hdi(sim_chain[variable][:, 0] - sim_chain[variable][:, 1])

    if variable == 'mu':
        v = 'm'
    elif variable == 'sigma':
        v = 'sd'
    else:
        raise ValueError('variable argument must be either "mu" or "sigma".')
    if hdim_l > ROPE[1]:
        goal_tally['HDI{} > ROPE'.format(v)] += 1
    elif hdim_r < ROPE[0]:
        goal_tally['HDI{} < ROPE'.format(v)] += 1
    elif ROPE[0] < hdim_l and hdim_r < ROPE[1]:
        goal_tally['HDI{} in ROPE'.format(v)] += 1
    else:
        pass
    if hdim_r - hdim_l < maxHDIW:
        goal_tally['HDI{} width < max'.format(v)] += 1
    return goal_tally


def get_hdi(samples, cred_mass=0.95):
    sorted_samples = sorted(samples)
    n_samples = len(sorted_samples)

    ci_idx_inc = int(cred_mass * n_samples)
    n_cis = n_samples - ci_idx_inc

    ci_width = n_cis * [0]
    for i in range(n_cis):
        ci_width[i] = sorted_samples[i + ci_idx_inc] - sorted_samples[i]

    hdi_min = sorted_samples[np.argmin(ci_width)]
    hdi_max = sorted_samples[np.argmin(ci_width) + ci_idx_inc]

    return hdi_min, hdi_max
