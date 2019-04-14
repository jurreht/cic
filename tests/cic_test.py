import os

import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy.io
import scipy.stats

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


def test_cic_model_no_effect():
    """
    Test a 3x3 CIC model where none of the treatments have any effect.
    The test is done by simulating and estimating the model many times
    and checking the coverage of the confidence intervals.
    """
    np.random.seed(45354354)

    treat = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 1]
    ], dtype=np.bool)

    n_trials = 250
    n_obs = 1000
    quantiles = np.array([0.1, .3, .5, .7, .9])
    effect_in_ci = np.zeros((3, quantiles.shape[0]), dtype=np.int_)
    for trial_ind in range(n_trials):
        g, t, y = generate_sample(n_obs)
        model = cic.CICModel(y, g, t, treat, quantiles)
        effect_in_ci += (
            (model.quantile_effect - 1.96 * model.quantile_se <= 0) &
            (model.quantile_effect + 1.96 * model.quantile_se >= 0))

    coverage = effect_in_ci / n_trials
    assert_allclose(coverage, np.ones_like(coverage) * .95, rtol=5e-2)


def test_cic_model_shift_effect():
    """
    Test a 3x3 CIC model where the treatments are linear shifts, but
    different for different groups and times.
    The test is done by simulating and estimating the model many times
    and checking the coverage of the confidence intervals.
    """
    np.random.seed(45354354)

    treat = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 1]
    ], dtype=np.bool)

    n_trials = 250
    n_obs = 1000
    quantiles = np.array([.25, .5, .75])
    moments = [np.mean, np.std]
    quantile_in_ci = np.zeros((3, 3, 3), dtype=np.int_)
    moment_in_ci = np.zeros((3, 3, 2), dtype=np.int_)
    for trial_ind in range(n_trials):
        g, t, y = generate_sample(n_obs)
        y[(g == 1) & (t == 2)] += 1
        y[(g == 2) & (t == 1)] -= 1
        y[(g == 2) & (t == 2)] -= 2
        model = cic.CICModel(y, g, t, treat, quantiles, moments)

        mean, se = model.treatment_quantile(1, 2)
        quantile_in_ci[:, 0] += ((mean - 1.96 * se <= 1) &
                                 (mean + 1.96 * se >= 1))
        mean, se = model.treatment_quantile(2, 1)
        quantile_in_ci[:, 1] += ((mean - 1.96 * se <= -1) &
                                 (mean + 1.96 * se >= -1))
        mean, se = model.treatment_quantile(2, 2)
        quantile_in_ci[:, 2] += ((mean - 1.96 * se <= -2) &
                                 (mean + 1.96 * se >= -2))

        mean, se = model.treatment_moment(1, 2)
        moment_in_ci[:, 0, 0] += ((mean[0] - 1.96 * se[0] <= 1) &
                                  (mean[0] + 1.96 * se[0] >= 1))
        moment_in_ci[:, 0, 1] += ((mean[1] - 1.96 * se[1] <= 0) &
                                  (mean[1] + 1.96 * se[1] >= 0))
        mean, se = model.treatment_moment(2, 1)
        moment_in_ci[:, 1, 0] += ((mean[0] - 1.96 * se[0] <= -1) &
                                  (mean[0] + 1.96 * se[0] >= -1))
        moment_in_ci[:, 1, 1] += ((mean[1] - 1.96 * se[1] <= 0) &
                                  (mean[1] + 1.96 * se[1] >= 0))
        mean, se = model.treatment_moment(2, 2)
        moment_in_ci[:, 2, 0] += ((mean[0] - 1.96 * se[0] <= -2) &
                                  (mean[0] + 1.96 * se[0] >= -2))
        moment_in_ci[:, 2, 1] += ((mean[1] - 1.96 * se[1] <= 0) &
                                  (mean[1] + 1.96 * se[1] >= 0))

    quantile_coverage = quantile_in_ci / n_trials
    assert_allclose(quantile_coverage,
                    np.ones_like(quantile_coverage) * .95,
                    rtol=5e-2)
    moment_coverage = moment_in_ci / n_trials
    assert_allclose(moment_coverage,
                    np.ones_like(moment_in_ci) * .95,
                    rtol=5e-2)


def test_cic_model_dispersion_effect():
    """
    Test a 3x3 CIC model where treatments are multiplying the distribution
    by some number, which differs by group and time.
    The test is done by simulating and estimating the model many times
    and checking the coverage of the confidence intervals.
    """
    np.random.seed(45354354)

    treat = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 1]
    ], dtype=np.bool)

    n_trials = 250
    n_obs = 2000
    quantiles = np.array([.5])
    moments = [np.mean, np.std]
    quantile_in_ci = np.zeros((3, 3, 1), dtype=np.int_)
    moment_in_ci = np.zeros((3, 3, 2), dtype=np.int_)
    for trial_ind in range(n_trials):
        g, t, y = generate_sample(n_obs)
        y[(g == 1) & (t == 2)] *= 2
        y[(g == 2) & (t == 1)] *= 3
        y[(g == 2) & (t == 2)] *= .5
        model = cic.CICModel(y, g, t, treat, quantiles, moments)

        # Q_{aX}(p) = a Q_X(p) for a quantile function Q and a > 0.
        # The median here is 1000, 2 * 1000 = 2000, hence the QTE is 1000
        mean, se = model.treatment_quantile(1, 2)
        quantile_in_ci[:, 0] += ((mean - 1.96 * se <= 1000) &
                                 (mean + 1.96 * se >= 1000))
        # The median here is 0, 3 * 0 = 0, hence the QTE is 0
        mean, se = model.treatment_quantile(2, 1)
        quantile_in_ci[:, 1] += ((mean - 1.96 * se <= 0) &
                                 (mean + 1.96 * se >= 0))
        # The median here is 1000, .5 * 1000 = 500, hence the QTE is -500
        mean, se = model.treatment_quantile(2, 2)
        quantile_in_ci[:, 2] += ((mean - 1.96 * se <= -500) &
                                 (mean + 1.96 * se >= -500))

        mean, se = model.treatment_moment(1, 2)
        # The mean goes from 1000 to 2000 => ATE = 1000
        moment_in_ci[:, 0, 0] += ((mean[0] - 1.96 * se[0] <= 1000) &
                                  (mean[0] + 1.96 * se[0] >= 1000))
        # The standard deviation goes from 1 to 2 => TE = 1
        moment_in_ci[:, 0, 1] += ((mean[1] - 1.96 * se[1] <= 1) &
                                  (mean[1] + 1.96 * se[1] >= 1))
        mean, se = model.treatment_moment(2, 1)
        # The mean goes from 0 to 0 => ATE = 0
        moment_in_ci[:, 1, 0] += ((mean[0] - 1.96 * se[0] <= 0) &
                                  (mean[0] + 1.96 * se[0] >= 0))
        # The standard deviation goes from 1/3 to 1 => TE = 2/3
        moment_in_ci[:, 1, 1] += ((mean[1] - 1.96 * se[1] <= 2 / 3) &
                                  (mean[1] + 1.96 * se[1] >= 2 / 3))
        mean, se = model.treatment_moment(2, 2)
        # The mean goes from 1000 to 500 => ATE = -500
        moment_in_ci[:, 2, 0] += ((mean[0] - 1.96 * se[0] <= -500) &
                                  (mean[0] + 1.96 * se[0] >= -500))
        # The standard deviation goes from 1 to .5 => TE = -.5
        moment_in_ci[:, 2, 1] += ((mean[1] - 1.96 * se[1] <= -.5) &
                                  (mean[1] + 1.96 * se[1] >= -.5))

    quantile_coverage = quantile_in_ci / n_trials
    assert_allclose(quantile_coverage,
                    np.ones_like(quantile_coverage) * .95,
                    rtol=5e-2)
    moment_coverage = moment_in_ci / n_trials
    assert_allclose(moment_coverage,
                    np.ones_like(moment_in_ci) * .95,
                    rtol=5e-2)


def test_test_model_based_on_quantile_valid():
    np.random.seed(3423482)

    treat = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 1]
    ], dtype=np.bool)

    n_trials = 100
    n_obs = 500
    quantiles = np.array([.5])
    reject = 0
    for trial_ind in range(n_trials):
        g, t, y = generate_sample(n_obs)
        # y[(g == 1) & (t == 2)] = 2 * y[(g == 1) & (t == 2)] - 3
        # y[(g == 2) & (t == 1)] = np.exp(y[(g == 2) & (t == 1)])
        # y[(g == 1) & (t == 2)] *= 2
        # y[(g == 2) & (t == 1)] -= 3
        # y[(g == 2) & (t == 2)] += 1
        model = cic.CICModel(y, g, t, treat, quantiles)

        test_stat, rank_dist = model.test_model_based_on_quantile(0)
        crit_val = scipy.stats.chi2.ppf(.95, rank_dist)
        # import pdb; pdb.set_trace()
        if test_stat > crit_val:
            reject += 1

    reject_prob = reject / n_trials
    # Just check that the rejection probability is not too large.
    # To get reject_prob~0.05 increse n_obs above, but this slows
    # down the test too much.
    assert reject_prob <= 0.05


def generate_sample(n_obs):
    g = np.random.choice(np.arange(3), n_obs)
    t = np.random.choice(np.arange(3), n_obs)

    u = np.random.randn(n_obs)
    y = np.empty(n_obs)
    y[t == 0] = u[t == 0]**3
    y[t == 1] = u[t == 1] / 3
    y[t == 2] = u[t == 2] + 1000

    return g, t, y
