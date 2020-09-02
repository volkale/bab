import numpy as np
from scipy import stats
from bab.make_data import make_data
from bab.power import get_power


np.random.seed(1)


def test_power(stan_model):
    n = 10
    samples = 5000
    delta = 1.2

    tally_rejection = 0

    for i in range(samples):
        y1 = stats.norm.rvs(delta, 1, n)
        y2 = stats.norm.rvs(0, 1, n)
        pval = stats.ttest_ind(y1, y2)[1]
        tally_rejection += int(pval < 0.05)

    frequentist_power = tally_rejection / samples

    y1, y2 = make_data(delta, 1, 0, 1, 10, percent_outliers=0, sd_outlier_factor=0, rand_seed=0)

    power_dict = get_power(stan_model, y1, y2, (-.1, .1), (0., 1.), 1., 1.)
    l, r = power_dict['HDIm in ROPE'][1:]
    bayesian_power = 1 - r  # 95 sure that the power is at least 1 - l

    assert frequentist_power <= bayesian_power
