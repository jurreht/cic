import os

import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy.io

import cic


def cases():
    """
    Loads all filenames of the pre-calculated test cases.
    """
    case_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'cases'
    )
    cases = []
    for dir_path, _, files in os.walk(case_dir):
        cases = cases + [os.path.join(dir_path, f) for f in files]
    return cases


@pytest.mark.parametrize('inpath', cases())
# Run both serially and in parallel
@pytest.mark.parametrize('n_jobs', [None, -1])
def test_cic(inpath, n_jobs):
    np.random.seed(323490)

    # Load the case
    objs = scipy.io.loadmat(inpath)

    y00 = objs['y00'][:, 0]
    y01 = objs['y01'][:, 0]
    y10 = objs['y10'][:, 0]
    y11 = objs['y11'][:, 0]

    est_qte, se_qte, est_ate, se_ate = cic.calculate_cic(
        y00, y01, y10, y11, n_bootstraps=499, n_draws=10000,
        moments=[np.mean],
        n_jobs=n_jobs,
        # The original code uses some small (in my view unneccessary)
        # numerical corrections when calculating cdf's and inverse cdf's.
        # Without using them here also there will always be some test
        # cases slightly off.
        use_corrections=True
    )
    est_test = objs['est'][0, 1:10]
    se_test = objs['se'][1, 1:10]

    # Test quantile treatment effects
    assert_allclose(est_qte, est_test)
    assert_allclose(se_qte, se_test, atol=5e-2, rtol=1e-3)

    # Test average treatment effect
    # It is possible to get closer than an atol of 5e-3 by increasing n_draws
    # above, at the cost of slower tests
    assert_allclose(est_ate[0], objs['est'][0, 0], atol=5e-3)
    assert_allclose(se_ate[0], objs['se'][1, 0], atol=5e-2, rtol=1e-3)
