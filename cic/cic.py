import numpy as np


def calculate_cic(
    y00, y01, y10, y11, quantiles=np.linspace(.1, .9, 9), moments=None,
    n_bootstraps=99, n_draws=1000, use_corrections=False
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

    estimated_quantile_effects, estimated_moment_effects = calculate_effects(
        y00, y01, y10, y11, quantiles, moments, draws, cdf_corr, inv_corr)

    # Bootstrap standard errors
    bootstrap_quantile_eff = np.empty((n_bootstraps, quantiles.shape[0]))
    n_moments = len(moments) if moments is not None else 0
    bootstrap_moment_eff = np.empty((n_bootstraps, n_moments))
    for i in range(n_bootstraps):
        y00_resample = np.random.choice(y00, y00.shape[0], replace=True)
        y01_resample = np.random.choice(y01, y01.shape[0], replace=True)
        y10_resample = np.random.choice(y10, y10.shape[0], replace=True)
        y11_resample = np.random.choice(y11, y11.shape[0], replace=True)

        bootstrap_quantile_eff[i], bootstrap_moment_eff[i] = calculate_effects(
            y00_resample, y01_resample, y10_resample, y11_resample,
            quantiles, moments, draws, cdf_corr, inv_corr
        )

    if n_bootstraps > 0:
        bootstrap_quantile_se = np.std(bootstrap_quantile_eff, axis=0)
        bootstrap_moment_se = np.std(bootstrap_moment_eff, axis=0)
    else:
        bootstrap_quantile_se = np.zeros(quantiles.shape[0])
        bootstrap_moment_se = np.zeros(n_moments)

    if moments is None:
        return estimated_quantile_effects, bootstrap_quantile_se
    else:
        return (estimated_quantile_effects, bootstrap_quantile_se,
                estimated_moment_effects, bootstrap_moment_se)


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


def cdf_support(y, cdf_corr):
    support = np.unique(y)
    cdf = np.mean(y[:, np.newaxis] <= support[np.newaxis] + cdf_corr, axis=0)
    return cdf, support


def calc_cf_cdf(y00, y01, y10, cdf_corr, inv_corr):
    cdf, support = cdf_support(y01, cdf_corr)
    # Do not use np.quantile() here because there is no way to make
    # it behave like the generalized inverse cdf (which is what we want).
    cdf = get_quantiles(*cdf_support(y00, cdf_corr), cdf, inv_corr)
    cdf = np.mean(y10[:, np.newaxis] <= cdf[np.newaxis] + cdf_corr, axis=0)
    cdf[-1] = 1
    return cdf, support


def get_quantiles(cdf, support, quantiles, inv_corr):
    indices = np.argmin(
        cdf[:, np.newaxis] < (quantiles[np.newaxis] - inv_corr),
        axis=0)
    return support[indices]


def sample_from_cdf(cdf, support, draws):
    return get_quantiles(cdf, support, draws, 0)
