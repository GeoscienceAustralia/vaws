import logging
import warnings

from numpy import exp, log, sqrt, diag, newaxis, ones
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import weibull_min, lognorm
from collections import OrderedDict


def fit_vulnerability_curve(cfg, df_dmg_idx):
    """

    Args:
        cfg:
        df_dmg_idx:

    Returns:

    """
    # array changed to vector
    xdata = (cfg.speeds[:, newaxis] * ones((1, cfg.no_sims))).flatten()
    ydata = df_dmg_idx.flatten()

    fitted_curve = {}
    for key, func_ in cfg.dic_obj_for_fitting.items():

        with warnings.catch_warnings():

            warnings.simplefilter("error", OptimizeWarning)

            try:
                popt, pcov = curve_fit(eval(func_), xdata, ydata)
            except RuntimeError as e:
                logging.warning(e.message + ' at {} curve fitting'.format(key))
            except OptimizeWarning as e:
                logging.warning(e.message + ' at {} curve fitting'.format(key))
            else:
                _sigma = sqrt(diag(pcov))
                fitted_curve[key] = dict(param1=popt[0],
                                         param2=popt[1],
                                         sigma1=_sigma[0],
                                         sigma2=_sigma[1])

    return fitted_curve


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

        try:
            popt, pcov = curve_fit(vulnerability_lognorm, cfg.speeds, counted)
        except RuntimeError as e:
            logging.warning(e.message + ' at {} damage state fragility fitting'.
                            format(state))
        else:
            _sigma = sqrt(diag(pcov))
            frag_counted[state] = dict(param1=popt[0],
                                       param2=popt[1],
                                       sigma1=_sigma[0],
                                       sigma2=_sigma[1])

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
