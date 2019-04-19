import numpy as np
from bab.make_data import make_data


def test_make_data():
    mu1 = mu2 = 0
    sigma1 = sigma2 = 1
    n_per_group = int(1e6)

    y1, y2 = make_data(mu1, sigma1, mu2, sigma2, n_per_group, percent_outliers=0, sd_outlier_factor=2.0, rand_seed=0)

    assert np.isclose(np.mean(y1), mu1, atol=0.1) and np.isclose(np.mean(y2), mu2, atol=0.1)
