import pytest


@pytest.fixture
def two_group_sample_data():
    """
    Sample data from section 4 of
    "Bayesian Estimation Supersedes the t-Test"
    M Meredith, J Kruschke - 2018 - journal.r-project.org
    """
    y1 = [5.77, 5.33, 4.59, 4.33, 3.66, 4.48]
    y2 = [3.88, 3.55, 3.29, 2.59, 2.33, 3.59]

    return y1, y2
