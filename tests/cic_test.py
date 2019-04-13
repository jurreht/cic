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


@pytest.mark.parametrize(
    'inpath',
    # exp8 and exp10 don't pass without use_corrections, which is only
    # supported for the simple case.
    [c for c in cases() if not ('exp8' in c or 'exp10' in c)])
def test_multiple_cic_from_simple_case(inpath):
    np.random.seed(442342234)

    # Load the case
    objs = scipy.io.loadmat(inpath)

    y00 = objs['y00'][:, 0]
    y01 = objs['y01'][:, 0]
    y10 = objs['y10'][:, 0]
    y11 = objs['y11'][:, 0]

    y = np.concatenate([y00, y01, y10, y11])
    g = np.concatenate([np.zeros(y00.shape[0] + y01.shape[0], dtype=np.int_),
                        np.ones(y10.shape[0] + y11.shape[0], dtype=np.int_)])
    t = np.concatenate([np.zeros(y00.shape[0], dtype=np.int_),
                        np.ones(y01.shape[0], dtype=np.int_),
                        np.zeros(y10.shape[0], dtype=np.int_),
                        np.ones(y11.shape[0], dtype=np.int_)])
    treat = np.array([[0, 0], [0, 1]], dtype=np.bool_)

    model = cic.CICModel(y, g, t, treat, n_bootstraps=499, moments=[np.mean],
                         n_draws=10000)

    assert np.all(model.treatment_for == np.array([[1, 1]], dtype=np.int_))

    est_test = objs['est'][0, 1:10]
    se_test = objs['se'][1, 1:10]

    assert_allclose(model.quantile_effect[0], est_test)
    assert_allclose(model.quantile_se[0], se_test, atol=5e-2, rtol=1e-3)
    # Test average treatment effect
    # It is possible to get closer than an atol of 5e-3 by increasing n_draws
    # above, at the cost of slower tests
    assert_allclose(model.moment_effect[0], objs['est'][0, 0], atol=5e-3)
    assert_allclose(model.moment_se[0], objs['se'][1, 0], atol=5e-2, rtol=1e-3)
