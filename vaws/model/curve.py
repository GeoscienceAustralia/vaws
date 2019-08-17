"""Curve module

    This module contains functions related to fragility and vulnerability curves.

"""
from __future__ import division, print_function
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning, minimize
from scipy.stats import weibull_min, lognorm, multinomial
from collections import OrderedDict

SMALL_VALUE = 1.0e-2


def no_within_bounds(row, bounds):
    freq = np.histogram(row, bins=bounds)[0]
    return pd.Series({f'n{i}': freq[i] for i in range(5)})


def compute_pe(row, denom):
    _dic = {}
    for i in range(1, 5):
        _dic[f'pe{i}'] = np.sum([row[f'n{i}'] for j in range(4, i-1, -1)]) / denom
    return pd.Series(_dic)


def likelihood(param, data, idx):
    """

    :param param: med, std
    :param data: pd.DataFrame with speed(index), pe1, pe2, pe3, pe4
    :param idx: index from 1 to 4
    :return:
    """
    med, std = param[0], param[1]
    pe = f'pe{idx}'
    temp = 0.0

    for speed, row in data.iterrows():
        p = lognorm.cdf(speed, std, loc=0, scale=med)
        if (p > 0) and (p < 1):
            temp += row[pe] * np.log(p) + (1.0 - row[pe]) * np.log(1 - p)
    return -1.0*temp


def fit_vulnerability_curve(cfg, df_dmg_idx):
    """

    Args:
        cfg:
        df_dmg_idx:

    Returns: dict of fitted_curve

    """
    logger = logging.getLogger(__name__)

    # array changed to vector
    xdata = (cfg.wind_speeds[:, np.newaxis] * np.ones((1, cfg.no_models))).flatten()
    ydata = df_dmg_idx.flatten()

    fitted_curve = {}
    for key, func_ in cfg.dic_obj_for_fitting.items():

        with warnings.catch_warnings():

            warnings.simplefilter("error", OptimizeWarning)

            try:
                popt, pcov = curve_fit(eval(func_), xdata, ydata)
            except RuntimeError as e:
                logger.warning(str(e) + f' at {key} curve fitting')
            except OptimizeWarning as e:
                logger.warning(str(e) + f' at {key} curve fitting')
            else:
                sigma = np.sqrt(np.diag(pcov))
                fitted_curve[key] = dict(param1=popt[0],
                                         param2=popt[1],
                                         sigma1=sigma[0],
                                         sigma2=sigma[1])

    return fitted_curve


def fit_fragility_curves(cfg, dmg_idx):
    """

    Args:
        cfg:
        dmg_idx: numpy array

    Returns: dict with keys of damage state

    """
    logger = logging.getLogger(__name__)

    bounds = cfg.fragility_i_thresholds[:]
    bounds.append(1.0)
    bounds.insert(0, 0.0)

    df = pd.DataFrame(dmg_idx).apply(no_within_bounds, args=(bounds,), axis=1)
    df.index = cfg.wind_speeds
    df = df.merge(df.apply(compute_pe, args=(cfg.no_models,), axis=1),
                  left_index=True, right_index=True)

    # calculate damage probability
    frag_counted = {'OLS': OrderedDict(), 'MLE': OrderedDict()}
    use_old = False
    for i, (state, value) in enumerate(cfg.fragility.iterrows(), 1):
        pe = f'pe{i}'

        # OLS
        try:
            popt, pcov = curve_fit(vulnerability_lognorm, cfg.wind_speeds, df[pe])
        except RuntimeError as e:
            logger.warning(str(e) + f' at {state} damage state fragility fitting')
        else:
            _sigma = np.sqrt(np.diag(pcov))
            frag_counted['OLS'][state] = dict(param1=popt[0],
                                              param2=popt[1],
                                              sigma1=_sigma[0],
                                              sigma2=_sigma[1])

        # MLE
        lower = df.loc[df[pe] == 0.0].index.min()
        upper = df[pe].idxmax()
        max_zero = df.loc[df[pe] == 0.0].index.max()
        med0 = 0.5*(lower + upper)
        std0 = (upper - max_zero) / 6.0
        if use_old:
            bounds_med = (max(lower*0.9, results.x[0]), upper*1.1)
        else:
            bounds_med = (lower*0.9, upper*1.1)
        bounds_sig = tuple(np.array([1.0e-2, 1.2]) * std0)

        results = minimize(likelihood, method='L-BFGS-B', x0=[med0, std0],
                           args=(df, i,), bounds=[bounds_med, bounds_sig])

        if results.success:
            frag_counted['MLE'][state] = dict(param1=results.x[0],
                                              param2=results.x[1])
            use_old = True
        else:
            use_old = False
    return frag_counted, df


def fit_fragility_curves_using_mle(cfg, df_dmg_idx):
    """

    Args:
        cfg:
        df_dmg_idx:

    Returns: dict with keys of damage state

    """
    logger = logging.getLogger(__name__)

    bounds = cfg.fragility.threshold.values
    bounds = np.append(bounds, 1.0)
    bounds = np.insert(bounds, 0, 0.0)

    df = df_dmg_idx.apply(no_within_bounds, args=(bounds,), axis=1)
    df['speed'] = cfg.wind_speeds
    df = df.merge(df.apply(compute_pe, args=(cfg.no_models,), axis=1),
                  left_index=True, right_index=True)
    # calculate damage probability
    plt.figure()
    for i in range(1, 5):
        pe = f'pe{i}'
        lower = df.loc[df[pe] == 0.0, 'speed'].min()
        upper = df.loc[df[pe].idxmax(), 'speed']
        max_zero = df.loc[df[pe] == 0.0, 'speed'].max()
        med0 = 0.5*(lower + upper)
        std0 = (upper - max_zero) / 6.0
        if i > 1:
            if results.success:
                bounds_med = (max(lower*0.9, results.x[0]), upper*1.1)
        else:
            bounds_med = (lower*0.9, upper*1.1)
        bounds_sig = tuple(np.array([1.0e-2, 1.2]) * std0)

        results = minimize(likelihood, method='L-BFGS-B', x0=[med0, std0],
                           args=(df, i,), bounds=[bounds_med, bounds_sig])
        if results.success:
            plt.plot(df.speed, df[pe], 'x')
            plt.plot(df.speed, lognorm.cdf(df.speed, results.x[1], loc=0, scale=results.x[0]))

        print(bounds_med)
        print(bounds_sig)
        print(results.success, results.x)

    plt.savefig('./aaa.png')

    results = minimize(likelihood_multinomial, x0=param0, args=df,
                       bounds=[bounds_med, bounds_med, bounds_med, bounds_med, bounds_sig],
                       constraints=constraints)
    return results


def vulnerability_weibull(x, alpha, beta):
    """Return vulnerability in Weibull CDF

    Args:
        x: 3sec gust wind speed at 10m height
        alpha: parameter value used in defining vulnerability curve
        beta: ditto

    Returns: weibull_min.cdf(x, shape, loc=0, scale)

    Note:

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
    shape = 1 / alpha
    scale = np.exp(beta)

    return weibull_min.cdf(x, shape, loc=0, scale=scale)


def vulnerability_weibull_pdf(x, alpha, beta):
    """Return PDF of vulnerability curve in Weibull

    Args:
        x: 3sec gust wind speed at 10m height
        alpha: parameter value used in defining vulnerability curve
        beta: ditto

    Returns: weibull_min.cdf(x, shape, loc=0, scale)
    """
    # convert alpha and beta to shape and scale respectively
    shape = 1 / alpha
    scale = np.exp(beta)

    return weibull_min.pdf(x, shape, loc=0, scale=scale)


def vulnerability_lognorm(x, med, std):
    """Return vulnerability in lognormal CDF

    Args:
        x: 3sec gust wind speed at 10m height
        med: exp(mean of log x)
        std: standard deviation of log x

    Returns: lognorm.cdf(x, std, loc=0, scale=med)

    """

    return lognorm.cdf(x, std, loc=0, scale=med)

def vulnerability_lognorm_pdf(x, med, std):
    """Return PDF of vulnerability curve in Lognormal

    Args:
        x: 3sec gust wind speed at 10m height
        med: exp(mean of log x)
        std: standard deviation of log x

    Returns: lognorm.pdf(x, std, loc=0, scale=med)

    """

    return lognorm.pdf(x, std, loc=0, scale=med)
