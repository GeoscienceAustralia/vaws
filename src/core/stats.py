import math


def lognormal_mean(m, stddev):
    """ compute mean of log x with mean and std. of x
    Args:
        m: mean of x
        stddev: standard deviation of x

    Returns: mean of log x

    """
    return math.log(m) - (0.5 * math.log(1.0 + (stddev * stddev) / (m * m)))


def lognormal_stddev(m, stddev):
    """ compute std. of log x with mean and std. of x

    Args:
        m: mean of x
        stddev: standard deviation of x

    Returns: std. of log x

    """
    return math.sqrt(math.log((stddev * stddev) / (m * m) + 1))


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
    return math.exp(m + 0.5 * stddev * stddev)


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
    return math.sqrt((math.exp(stddev**2.0) - 1.0) *
                     math.exp(2.0*m + stddev**2.0))
    #return lognormal_underlying_mean(m, stddev) * \
    #       math.sqrt((math.exp(stddev * stddev) - 1.0))
