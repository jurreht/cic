import functools
import itertools

import joblib
import numba
import numpy as np


def calculate_cic(
    y00, y01, y10, y11, quantiles=np.linspace(.1, .9, 9), moments=None,
    n_bootstraps=99, n_draws=1000, n_jobs=None, use_corrections=False
):
    r"""
    Estimates a model using the Changes-In-Changes estimator.

    This function estimates non-linear treatments effects using
    the Athey & Imbens [1]_ changes-in-changes estimator. Only
    the case with continuous data is supported. `calculate_cic`
    calculates quantile treatment effects, as well as treatment
    effects on arbitrary user-specified functions of the distribution.
    The latter are calculated by simulating from the counterfactual
    distribution. Standard errors are computed using the bootstrap.

    Parameters
    ----------
    y00 : array_like
        Observations in the untreated group before the intervention.
    y01 : array_like
        Observations in the untreated group after the intervention.
    y10 : array_like
        Observations in the treated group before the intervention.
    y11 : array_like
        Observations in the treated group after the intervention.
    quantiles : array_like, optional
        Quantiles at which quantile treatment effects are calculated.
        Default is {.1, .2, ..., .9}.
    moments : sequence, optional
        A sequence of user-defined functions. These functions can be
        used to calculate the treatment effect on arbitrary "moments"
        of the distribution. Every element should take a single argument,
        which is a (simulated or observed) sample for which the
        moment should be calculated, and should return a single number,
        which is the calculated moment.
    n_bootstraps : int, optional
        Number of bootstrap simulations to calculate standard errors.
        Set to 0 to prevent bootstrapping. In this case, calculated
        standard errors will equal zero.
    n_draws : int, optional
        Number of draws from the counterfactual distribution to
        calculate the treatment effect of user-supplied moments. Only
        relevant if ``moments is not None``.
    n_jobs : int, optional
        Number of parallel jobs to use for bootstrapping standard
        errors. When None (the default), bootstrap serially. Otherwise,
        this interpreted in as in joblib.Parallel, i.e. specify a
        positive int to run that many jobs and a negative int to run
        ``num_cpu + 1 + n_jobs`` jobs.
    use_corrections : bool, optional
        Use numerical corrections when calculating CDF's and inverse
        CDF's as in the original code by Athey & Imbens. Set to
        True to obtain quantile treatment effects that are numerically
        equivalent to those calculated using the original codes. In
        general, using this parameter is unnecessary and discouraged,
        see the notes below.

    Returns
    -------
        estimated_quantile_effects : array_like
            Estimated quantile treatment effects for every point
            in `quantiles` in order.
        bootstrap_quantile_se : array_like
            Bootstrapped standard errors for every point in
            `quantiles` in order.
        estimated_moment_effects : array_like, optional
            Estimated treatment effects for every function
            specified in `moments`. Only returned when
            ``moments is not None``.
        bootstrap_moment_se : array_like, optional
            Bootstrapped standard errors for every function
            specified in `moments`. Only returned when
            ``moments is not None``.

    Notes
    -----

    `calculate_cic` calculates the exact counterfactual distribution,
    :math:`\bar{F}_{11}`, using

    .. math:: \bar{F}_{11}(x) = \hat{F}_{10}(\hat{F}_{00}^{-1}(\hat{F}_{01}(x))),

    where :math:`\hat{F}_{it}` is the empirical cdf of group :math:`i` in
    period :math:`t`.

    Quantile treatment effects are calculated using the generalized
    inverse cdf:

    .. math:: \begin{align*}
       F^{-1}(p) &= \inf\{x \in \mathbb{R} : F(x) \ge p\} \\
       \text{qte}(p) &= \hat{F}_{11}^{-1}(p) - \bar{F}_{11}^{-1}(p)
       \end{align*}

    Moments are calculated by using a probability integral transform
    to sample from :math:`\bar{F}_{11}`.

    In general, `calculate_cic` does not give numerically equivalent
    results to the original code used by Athey & Imbens [2]_. The
    reason is that their code makes small numerical adjustments when
    calculating cdf's and inverse cdf's. In particular, they calculate
    cdf's as :math:`F(x) = P(X \le x + 0.00001)` and inverse cdf's as
    :math:`F^{-1}(x) = \inf\{x : F(x) \ge p - 0.000001\}`. (Not that
    the correction factors differ by a factor 10!) According to the
    comments, this is to "prevent numerical problems". Presumably,
    this is because in general floating point comparison is a bad
    idea. However, because during the calculation of counterfactual
    cdf's floats are only copied and not modified, there is no
    reason to be afraid of floating point comparisons in this case.
    Nevertheless, the user can provide the parameter `use_corrections`
    to use the same corrections as the Athey & Imbey codes. In this case,
    the calculated quantile treatment effects will be numerically
    equivalent. This option is included mainly to enable unit testing,
    users are discouraged from using it because it will in general
    lead to slightly wrong results.

    References
    ----------
    .. [1] Athey, Susan, and Guido W. Imbens. 2006. "Identification and
       Inference in nonlinear difference‐in‐differences models."
       *Econometrica* 74 (2): 431-497.
    .. [2] Athey, Susan and Guido W. Imbens. 2006. "CIC Code".
       Accessed April 11, 2019. https://athey.people.stanford.edu/research.
    """
    if use_corrections:
        cdf_corr = 0.00001
        inv_corr = 0.000001
    else:
        cdf_corr = inv_corr = 0

    # Use the same draws for calculating moments during effect size
    # calculation as during bootstrapping
    draws = np.random.uniform(size=n_draws)

    # Quantiles and draws need to be sorted for get_quantiles()
    quantiles.sort()
    draws.sort()

    estimated_quantile_effects, estimated_moment_effects = calculate_effects(
        y00, y01, y10, y11, quantiles, moments, draws, cdf_corr, inv_corr)

    # Bootstrap standard errors
    if n_jobs is None:
        # Run sequentially
        bootstrap_quantile_eff, bootstrap_moment_eff = zip(*map(
            lambda _: bootstrap_sample(y00, y01, y10, y11, quantiles, moments,
                                       draws, cdf_corr, inv_corr),
            range(n_bootstraps)))
    else:
        # Run on multiple cores
        # Use threads a background since most of the time will be
        # spent in NumPy routines, which release the GIL
        ret = joblib.Parallel(n_jobs=n_jobs, prefer='threads')(
            joblib.delayed(bootstrap_sample)(y00, y01, y10, y11, quantiles,
                                             moments, draws, cdf_corr,
                                             inv_corr)
            for _ in range(n_bootstraps)
        )
        bootstrap_quantile_eff, bootstrap_moment_eff = zip(*ret)
    # Concatenate into a single numpy array
    bootstrap_quantile_eff = np.concatenate([
        x[np.newaxis] for x in bootstrap_quantile_eff
    ], axis=0)
    bootstrap_moment_eff = np.concatenate([
        x[np.newaxis] for x in bootstrap_moment_eff
    ], axis=0)

    if n_bootstraps > 0:
        bootstrap_quantile_se = np.std(bootstrap_quantile_eff, axis=0)
        bootstrap_moment_se = np.std(bootstrap_moment_eff, axis=0)
    else:
        bootstrap_quantile_se = np.zeros(quantiles.shape[0])
        n_moments = len(moments) if moments is not None else 0
        bootstrap_moment_se = np.zeros(n_moments)

    if moments is None:
        return estimated_quantile_effects, bootstrap_quantile_se
    else:
        return (estimated_quantile_effects, bootstrap_quantile_se,
                estimated_moment_effects, bootstrap_moment_se)


class CICModel:
    def __init__(
        self, y, g, t, treat, quantiles=np.linspace(.1, .9, 9), moments=None,
        n_draws=1000, n_bootstraps=99, n_jobs=None
    ):
        self.quantiles = quantiles
        n_obs = y.shape[0]
        n_groups = treat.shape[0]
        n_periods = treat.shape[1]
        if t.shape[0] != n_obs:
            raise ValueError('len(y) should equal len(t).')
        if g.shape[0] != n_obs:
            raise ValueError('len(y) should equal len(g).')
        if t.max() >= n_periods:
            raise ValueError('Invalid period provided for some observations.')
        if g.max() >= n_groups:
            raise ValueError('Invalid group provided for some observations.')
        if np.any((~treat[:, 1:]) & (treat[:, 1:] ^ treat[:, :-1])):
            raise ValueError('A group cannot become untreated after becoming'
                             ' treated.')
        self.g = g
        self.t = t

        # Use the same draws for calculating moments during effect size
        # calculation as during bootstrapping
        draws = np.random.uniform(size=n_draws)

        # Quantiles and draws need to be sorted for get_quantiles()
        quantiles.sort()
        draws.sort()

        # Calculate the effect using all possible combinations of treatment
        # and control
        possible_combinations = tuple(filter(
            lambda x: valid_combination(treat, *x),
            itertools.product(range(n_groups), range(n_periods), repeat=2)))
        self.effects = calculate_multiple_effects(
            y, g, t, possible_combinations, quantiles, moments, draws)

        # Bootstrap the covariance matrix of the treatments effects
        calc_bootstrap = functools.partial(
            self._bootstrap_multiple_effects, y, g, t, treat, quantiles,
            moments, possible_combinations, draws
        )
        if n_jobs is None:
            bootstrap_effects = np.empty((n_bootstraps, self.effects.shape[0],
                                          self.effects.shape[1]))
            for i in range(n_bootstraps):
                bootstrap_effects[i] = calc_bootstrap()
        else:
            bootstrap_effects = joblib.Parallel(n_jobs, prefer='threads', verbose=11)(
                joblib.delayed(calc_bootstrap)() for _ in range(n_bootstraps)
            )
            # bootstrap_effects is a list of ndarray's, make it a single
            # ndarray
            bootstrap_effects = np.concatenate([
                x[np.newaxis] for x in bootstrap_effects
            ], axis=0)

        # Calculate the combined effect
        self.n_treatment_effects = treat.sum()
        self.treatment_for = np.empty((self.n_treatment_effects, 2))
        # The matrix A maps `effects` into the (g, t)-treatment effect
        self.A = np.zeros((len(possible_combinations),
                           self.n_treatment_effects))
        i = 0
        for g1, t1 in itertools.product(range(n_groups), range(n_periods)):
            if treat[g1, t1]:
                self.A[:, i] = tuple(map(
                    lambda x: x[2] == g1 and x[3] == t1,
                    possible_combinations
                ))
                self.treatment_for[i] = g1, t1
                i += 1

        effect = np.empty((self.n_treatment_effects, self.effects.shape[1]))
        effect_se = np.empty_like(effect)
        self.cov_inv = np.empty((self.effects.shape[1],
                                 len(possible_combinations),
                                 len(possible_combinations)))
        for effect_ind in range(self.effects.shape[1]):
            # TODO: The covariance of the bootstrap sample is not necessarily a
            # good estimator of the covariance matrix! Perhaps try also using
            # the percentile method. See Machado, Jose A.F. and Paulo Parente.
            # 2005. "Bootstrap estimation of covariance matrices via the
            # percentile method." Econometrics Journal 8: 70-78.
            cov = np.cov(bootstrap_effects[:, :, effect_ind], rowvar=False,
                         bias=True)
            if self.effects.shape[0] == 1:
                # In this case np.cov() returns a scalar. Invert it and make
                # it a matrix.
                cov_inv = (1 / cov)[np.newaxis, np.newaxis]
            else:
                cov_inv = np.linalg.pinv(cov)
            self.cov_inv[effect_ind] = cov_inv

            effect[:, effect_ind] = np.linalg.solve(
                self.A.T @ cov_inv @ self.A,
                self.A.T @ cov_inv @ self.effects[:, effect_ind])
            effect_cov = np.linalg.inv(self.A.T @ cov_inv @ self.A)
            effect_se[:, effect_ind] = np.sqrt(np.diag(effect_cov))

        self.all_effect = effect
        self.all_se = effect_se
        self.quantile_effect = effect[:, :quantiles.shape[0]]
        self.quantile_se = effect_se[:, :quantiles.shape[0]]
        if moments is None:
            self.moment_effect = np.empty((self.n_treatment_effects, 0))
            self.moment_se = np.empty_like(self.moment_effect)
        else:
            self.moment_effect = effect[:, quantiles.shape[0]:]
            self.moment_se = effect_se[:, quantiles.shape[0]:]

    def treatment_quantile(self, g, t):
        ind = self._treatment_ind(g, t)
        return self.quantile_effect[ind], self.quantile_se[ind]

    def treatment_moment(self, g, t):
        ind = self._treatment_ind(g, t)
        return self.moment_effect[ind], self.moment_se[ind]

    def _treatment_ind(self, g, t):
        row_match = (self.treatment_for == np.array([g, t])).all(axis=1)
        return np.nonzero(row_match)[0][0]

    def test_model_based_on_quantile(self, quantile_ind):
        effects_ind = quantile_ind
        mean_diff = (self.effects[:, effects_ind] -
                     self.A @ self.quantile_effect[:, quantile_ind])
        cov_inv = self.cov_inv[effects_ind]
        test_stat = mean_diff.T @ cov_inv @ mean_diff
        # We need the rank of V here, and cov is the psuedo-inverse of V.
        # However, the rank of the pseudo-inverse is the same as of the
        # original matrix so there is no problem here.
        rank_dist = (np.linalg.matrix_rank(cov_inv) -
                     self.n_treatment_effects)
        return test_stat, rank_dist

    def test_model_based_on_moment(self, moment_ind):
        effects_ind = moment_ind + self.quantiles.shape[0]
        mean_diff = (self.effects[:, effects_ind] -
                     self.A @ self.moment_effect[:, moment_ind])
        cov_inv = self.cov_inv[effects_ind]
        test_stat = mean_diff.T @ cov_inv @ mean_diff
        # We need the rank of V here, and cov is the psuedo-inverse of V.
        # However, the rank of the pseudo-inverse is the same as of the
        # original matrix so there is no problem here.
        rank_dist = (np.linalg.matrix_rank(cov_inv) -
                     self.n_treatment_effects)
        return test_stat, rank_dist

    def combine_effects(self, effects_for, weigh_by='n'):
        n_effects = self.all_effect.shape[1]
        weights = np.zeros((n_effects, self.n_treatment_effects, 1))
        if weigh_by == 'n':
            for i in range(self.n_treatment_effects):
                g, t = self.treatment_for[i]
                if (g, t) in effects_for:
                    weights[:, i] = np.sum((self.g == g) & (self.t == t))
            weights /= weights[0].sum()
        elif weigh_by == 'cov':
                target = np.zeros((self.n_treatment_effects, 1),
                                  dtype=np.bool_)
                for i in range(self.n_treatment_effects):
                    g, t = self.treatment_for[i]
                    if (g, t) in effects_for:
                        target[i] = True
                for effect_ind in range(n_effects):
                    weights[effect_ind] = np.linalg.solve(
                        target.T @ self.A.T @ self.cov_inv[effect_ind] @ self.A @ target,
                        target.T @ self.A.T @ self.cov_inv[effect_ind] @ self.A
                    ).T
        else:
            raise ValueError('Invalid value for weigh_by, use n or cov.')

        weighed_effects = np.empty(n_effects)
        weighed_se = np.empty_like(weighed_effects)
        for effect_ind in range(n_effects):
            weighed_effects[effect_ind] = (weights[effect_ind].T @
                                           self.all_effect[:, effect_ind])
            weighed_cov = weights[effect_ind].T @ np.linalg.solve(
                self.A.T @ self.cov_inv[effect_ind] @ self.A,
                weights[effect_ind])
            weighed_se[effect_ind] = np.sqrt(np.diag(weighed_cov))

        quantile_combined = weighed_effects[:self.quantiles.shape[0]]
        quantile_combined_se = weighed_se[:self.quantiles.shape[0]]
        moment_combined = weighed_effects[self.quantiles.shape[0]:]
        moment_combined_se = weighed_se[self.quantiles.shape[0]:]
        return (quantile_combined, quantile_combined_se, moment_combined,
                moment_combined_se)

    def _bootstrap_multiple_effects(
        self, y, g, t, treat, quantiles, moments, possible_combinations, draws
    ):
        y_resample = np.empty_like(y)
        n_groups = treat.shape[0]
        n_periods = treat.shape[1]
        for j, k in itertools.product(range(n_groups), range(n_periods)):
            target = (g == j) & (t == k)
            y_resample[target] = np.random.choice(y[target], target.sum(),
                                                  replace=True)
        return calculate_multiple_effects(
            y_resample, g, t, possible_combinations, quantiles, moments,
            draws
        )


def calculate_effects(
    y00, y01, y10, y11, quantiles, moments, draws, cdf_corr, inv_corr
):
    # Calculate quantile treatment effects
    observed_quantiles = get_quantiles(*cdf_support(y11, cdf_corr),
                                       quantiles, inv_corr)
    cf_cdf, cf_support = calc_cf_cdf(y00, y01, y10, cdf_corr, inv_corr)
    cf_quantiles = get_quantiles(cf_cdf, cf_support, quantiles, inv_corr)
    quantile_effects = observed_quantiles - cf_quantiles

    if moments is not None:
        observed_moments = np.array([f(y11) for f in moments])
        cf_draws = sample_from_cdf(cf_cdf, cf_support, draws)
        cf_moments = np.array([f(cf_draws) for f in moments])
        moment_effects = observed_moments - cf_moments

        return quantile_effects, moment_effects
    else:
        return quantile_effects, None


def calculate_multiple_effects(
    y, g, t, possible_combinations, quantiles, moments, draws
):
    n_targets = quantiles.shape[0]
    if moments is not None:
        n_targets += len(moments)
    effects = np.empty((len(possible_combinations), n_targets))
    for i, (g0, t0, g1, t1) in enumerate(possible_combinations):
        y00 = y[(g == g0) & (t == t0)]
        y01 = y[(g == g0) & (t == t1)]
        y10 = y[(g == g1) & (t == t0)]
        y11 = y[(g == g1) & (t == t1)]
        # calculate_effects returns None as second element if moments is None.
        # When moments is not None, we want to concatenate the return elements.
        effects[i] = np.concatenate([
            x for x in
            calculate_effects(y00, y01, y10, y11, quantiles, moments,
                              draws, 0, 0)
            if x is not None])
    return effects


def bootstrap_sample(
    y00, y01, y10, y11, quantiles, moments, draws, cdf_corr, inv_corr
):
    y00_resample = np.random.choice(y00, y00.shape[0], replace=True)
    y01_resample = np.random.choice(y01, y01.shape[0], replace=True)
    y10_resample = np.random.choice(y10, y10.shape[0], replace=True)
    y11_resample = np.random.choice(y11, y11.shape[0], replace=True)

    return calculate_effects(
        y00_resample, y01_resample, y10_resample, y11_resample,
        quantiles, moments, draws, cdf_corr, inv_corr
    )


def cdf_support(y, cdf_corr):
    support = np.unique(y)
    if support.shape[0] == y.shape[0] and cdf_corr == 0:
        # In this common case we can achieve a speedup
        cdf = np.arange(1, y.shape[0] + 1) / y.shape[0]
    else:
        cdf = np.mean(y[:, np.newaxis] <= support[np.newaxis] + cdf_corr,
                      axis=0)
    return cdf, support


def calc_cf_cdf(y00, y01, y10, cdf_corr, inv_corr):
    cdf, support = cdf_support(y01, cdf_corr)
    # Do not use np.quantile() here because there is no way to make
    # it behave like the generalized inverse cdf (which is what we want).
    cdf = get_quantiles(*cdf_support(y00, cdf_corr), cdf, inv_corr)
    cdf = np.mean(y10[:, np.newaxis] <= cdf[np.newaxis] + cdf_corr, axis=0)
    cdf[-1] = 1
    return cdf, support


@numba.jit(nopython=True, cache=True, nogil=True)
def get_quantiles(cdf, support, quantiles, inv_corr):
    ret = np.empty(quantiles.shape[0])
    q_ind = 0
    p = quantiles[0] - inv_corr
    for i in range(cdf.shape[0]):
        while cdf[i] >= p:
            ret[q_ind] = support[i]
            q_ind += 1
            if q_ind >= quantiles.shape[0]:
                break
            p = quantiles[q_ind] - inv_corr
        if q_ind >= quantiles.shape[0]:
            break
    return ret


def sample_from_cdf(cdf, support, draws):
    return get_quantiles(cdf, support, draws, 0)


def valid_combination(treat, g0, t0, g1, t1):
    return ((not (treat[g0, t0] or treat[g0, t1] or treat[g1, t0])) and
            treat[g1, t1])
