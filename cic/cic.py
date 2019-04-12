import numpy as np


def calculate_cic(
    y00, y01, y10, y11, quantiles=np.linspace(.1, .9, 9), moments=None,
    n_bootstraps=99, n_draws=1000, use_corrections=False
):
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

    bootstrap_quantile_se = np.std(bootstrap_quantile_eff, axis=0)
    bootstrap_moment_se = np.std(bootstrap_moment_eff, axis=0)

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
