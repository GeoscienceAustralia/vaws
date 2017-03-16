"""
    Curve module - fun with curves
"""
from numpy import exp, log, sqrt, diag, newaxis, ones
from scipy.optimize import curve_fit
from scipy.stats import weibull_min, lognorm
from collections import OrderedDict


def fit_vulnerability_curve(cfg, df_dmg_idx, flag_obj_func='weibull'):
    """

    Args:
        cfg:
        df_dmg_idx:
        flag_obj_func:

    Returns:

    """
    # array changed to vector
    xdata = (cfg.speeds[:, newaxis] * ones((1, cfg.no_sims))).flatten()
    ydata = df_dmg_idx.values.flatten()

    popt, pcov = curve_fit(eval(cfg.dic_obj_for_fitting[flag_obj_func]),
                           xdata,
                           ydata)
    perror = sqrt(diag(pcov))

    return popt, perror


def fit_fragility_curves(cfg, df_dmg_idx):
    """

    Args:
        cfg:
        df_dmg_idx:

    Returns:

    """

    # calculate damage probability
    frag_counted = OrderedDict()
    for state, value in cfg.fragility_thresholds.iterrows():
        counted = (df_dmg_idx > value['threshold']).sum(axis=1) / \
                  float(cfg.no_sims)

        popt, pcov = curve_fit(vulnerability_lognorm,
                               cfg.speeds,
                               counted.values)

        frag_counted.setdefault(state, {})['median'] = popt[0]
        frag_counted[state]['sigma'] = popt[1]
        frag_counted[state]['error'] = sqrt(diag(pcov))

    return frag_counted


def vulnerability_weibull(x, alpha_, beta_):
    """

    vulnerability curve with Weibull function

    Args:
        _alpha, _beta: parameters for vulnerability curve
        x: 3sec gust wind speed at 10m height

    Returns: weibull_min.cdf(x, shape, loc=0, scale)

    Notes
    ----

    weibull_min.pdf = c/s * (x/s)**(c-1) * exp(-(x/s)**c)
        c: shape, s: scale, loc=0

    weibull_min.cdf = 1 - exp(-(x/s)**c)

    while Australian wind vulnerability is defined as

        DI = 1 - exp(-(x/exp(beta))**(1/alpha))

    therefore:

        s = exp(beta)
        c = 1/alpha

    """
    # convert alpha and beta to shape and scale respectively
    shape_ = 1 / alpha_
    scale_ = exp(beta_)

    return weibull_min.cdf(x, shape_, loc=0, scale=scale_)


def vulnerability_weibull_pdf(x, alpha_, beta_):
    """

    vulnerability curve with Weibull function

    Args:
        _alpha, _beta: parameters for vulnerability curve
        x: 3sec gust wind speed at 10m height

    Returns: weibull_min.pdf(x, shape, loc=0, scale)

    Notes
    ----

    weibull_min.pdf = c/s * (x/s)**(c-1) * exp(-(x/s)**c)
        c: shape, s: scale, loc=0

    weibull_min.cdf = 1 - exp(-(x/s)**c)

    while Australian wind vulnerability is defined as

        DI = 1 - exp(-(x/exp(beta))**(1/alpha))

    therefore:

        s = exp(beta)
        c = 1/alpha

    """
    # convert alpha and beta to shape and scale respectively
    shape_ = 1 / alpha_
    scale_ = exp(beta_)

    return weibull_min.pdf(x, shape_, loc=0, scale=scale_)


def vulnerability_lognorm(x, med, std):
    """

    vulnerability curve with cumulative lognormal function

    Args:
        med, std: exp(mean of log x) and standard deviation of log x
        x: 3sec gust wind speed at 10m height

    Returns: lognorm.flag(x, std, loc=0, scale=med)

    """

    return lognorm.cdf(x, std, loc=0, scale=med)
