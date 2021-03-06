import numpy as np
from bab.mcmc import get_mcmc


def test_mcmc_random_seed(stan_model, two_group_sample_data):
    y1, y2 = two_group_sample_data

    mcmc1 = get_mcmc(stan_model, y1, y2, rand_seed=1).extract()
    mcmc2 = get_mcmc(stan_model, y1, y2, rand_seed=1).extract()

    np.testing.assert_equal(mcmc1, mcmc2)


def test_mcmc(stan_model, two_group_sample_data):
    y1, y2 = two_group_sample_data

    mcmc = get_mcmc(stan_model, y1, y2, rand_seed=1)

    row_ind_1 = list(mcmc.summary()['summary_rownames']).index('mu[1]')
    row_ind_2 = list(mcmc.summary()['summary_rownames']).index('mu[2]')
    col_ind_m = list(mcmc.summary()['summary_colnames']).index('mean')
    col_ind_rh = list(mcmc.summary()['summary_colnames']).index('Rhat')

    assert np.isclose(mcmc.summary()['summary'][row_ind_1, col_ind_m], np.mean(y1), atol=0.1)
    assert np.isclose(mcmc.summary()['summary'][row_ind_2, col_ind_m], np.mean(y2), atol=0.1)

    assert np.isclose(mcmc.summary()['summary'][:, col_ind_rh], 1.0, atol=0.1).all()


def test_mcmc_aggregation(stan_model, two_group_sample_data):
    y1, y2 = two_group_sample_data

    w1 = w2 = 6 * [2]
    mcmc_short = get_mcmc(stan_model, y1, y2, w1=w1, w2=w2, rand_seed=1)
    mcmc_long = get_mcmc(stan_model, 2 * y1, 2 * y2, rand_seed=1)

    for parameter in mcmc_short.extract().keys():
        assert np.isclose(mcmc_short.extract()[parameter], mcmc_long.extract()[parameter]).all()
