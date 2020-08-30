import warnings
import math
import numpy as np
from scipy.stats import norm


def make_data(mu1, sigma1, mu2, sigma2, n_per_group, percent_outliers=0, sd_outlier_factor=2.0, rand_seed=None):
    """
    Auxiliary function for generating random values from a mixture of normal distributions.
    :param mu1: mean of first group
    :param sigma1: standard deviation of first group
    :param mu2: mean of second group
    :param sigma2: standard deviation of second group
    :param n_per_group: number of samples per group
    :param percent_outliers: percentage of outliers to generate
    :param sd_outlier_factor: factor to apply to sigma1, sigma2 to define outlier standard deviation
    :param rand_seed (optimal), int
    :return: y1, y2
    """
    if rand_seed is not None:
        np.random.seed(int(rand_seed))

    n_outliers = math.ceil(n_per_group * percent_outliers / 100)  # Number of outliers.
    n = n_per_group - n_outliers  # Number from main distribution.

    _validate_input(n, n_outliers, percent_outliers)

    y1 = _generate_data(mu1, sigma1, n, n_outliers, sd_outlier_factor)
    y2 = _generate_data(mu2, sigma2, n, n_outliers, sd_outlier_factor)

    return y1, y2


def _validate_input(n, n_outliers, percent_outliers):
    if percent_outliers > 100 or percent_outliers < 0:
        raise ValueError("percent_outliers must be between 0 and 100.")
    if 0 < percent_outliers < 1:
        raise ValueError("percent_outliers is specified as percentage 0-100, not proportion 0-1.")
    if percent_outliers > 50:
        warnings.warn("percent_outliers indicates more than 50% outliers; did you intend this?")
    if n_outliers < 2 and percent_outliers > 0:
        raise ValueError("Combination of nPerGrp and pcntOut yields too few outliers.")
    if n < 2:
        raise ValueError("Too few non-outliers.")


def _generate_data(mu, sigma, n, n_outliers, sd_outlier_factor):
    y = norm.rvs(loc=mu, scale=sigma, size=n)
    if n_outliers > 0:
        y_out = norm.rvs(size=n_outliers)  # Random values for outliers
        y_out = ((y_out - np.mean(y_out)) / np.std(y_out)) * (sigma * sd_outlier_factor) + mu  # Realize exactly.
        y = np.concatenate([y, y_out])
    return y
