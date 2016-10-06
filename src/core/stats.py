from math import log, exp, sqrt
from scipy.stats import lognorm


def lognormal_mean(m, stddev):
    """ compute mean of log x with mean and std. of x
    Args:
        m: mean of x
        stddev: standard deviation of x

    Returns: mean of log x

    """
    return log(m) - (0.5 * log(1.0 + (stddev * stddev) / (m * m)))


def lognormal_stddev(m, stddev):
    """ compute std. of log x with mean and std. of x

    Args:
        m: mean of x
        stddev: standard deviation of x

    Returns: std. of log x

    """
    return sqrt(log((stddev * stddev) / (m * m) + 1))


def lognorm_rv_given_mean_stddev(m, stddev, rnd_state=None):

    mean_logx = lognormal_mean(m, stddev)
    sigma_logx = lognormal_stddev(m, stddev)

    if rnd_state:
        rnd_state.lognormal(mean=mean_logx, sigma=sigma_logx)
    else:
        lognorm.rvs(sigma_logx, scale=exp(mean_logx))

def lognormal_underlying_mean(m, stddev):
    """ compute mean of x with mean and std of log x

    Args:
        m: mean of log x
        stddev: std of log x

    Returns:

    """
    # if m == 0 or stddev == 0:
    #     print '{}'.format('why ???')
    #     return 0
    return exp(m + 0.5 * stddev * stddev)


def lognormal_underlying_stddev(m, stddev):
    """ compute std of x with mean and std of log x

    Args:
        m: mean of log x
        stddev: std of log x

    Returns: std of x

    """
    # if m == 0 or stddev == 0:
    #     print '{}'.format('strange why???')
    #     return 0
    return sqrt((exp(stddev * stddev) - 1.0) * exp(2.0 * m + stddev * stddev))
    #return lognormal_underlying_mean(m, stddev) * \
    #       math.sqrt((math.exp(stddev * stddev) - 1.0))
