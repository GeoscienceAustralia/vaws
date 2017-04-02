
import logging
from math import log, exp, sqrt, gamma, copysign
from scipy.stats import genextreme


def sample_lognormal(mu_lnx, std_lnx, rnd_state):
    """
    draw a sample from a lognormal distribution with mu, std of logarithmic x
    If std is zero, just return exp(mu_lnx)

    Args:
        mu_lnx: mean of log x
        std_lnx: std of log x
        rnd_state: numpy.random.RandomState

    Returns:

    """
    try:
        return rnd_state.lognormal(mean=mu_lnx, sigma=std_lnx)
    except ValueError:  # no sampling
        return exp(mu_lnx)


def sample_gev(mean_est, cov_est, big_a, big_b, shape_k, rnd_state=None):
    """
    JHD F(u) = exp{-[1-k(U-u)/a]**(1/k)}
    where a: scale factor, u: location factor
    k < 0: Type II (Frechet), k > 0: Type III (Weibull)

    scipy.stats.genextreme.rvs(c, loc=0, scale=1, size=1, random_state=None)
    c: shape (or k)

    Args:
        mean_est:
        big_a:
        big_b:
        cov_est:
        shape_k:
        rnd_state:

    Returns: random sample from the extreme value distribution Type III

"""
    assert shape_k > 0
    a, u = calc_parameters_gev(mean_est, cov_est, big_a, big_b)
    return copysign(genextreme.rvs(shape_k, loc=u, scale=a, size=1,
                                   random_state=rnd_state)[0], mean_est)


def calc_parameters_gev(mean_est, cov_est, big_a, big_b):
    """

    Args:
        mean_est: estimated mean (can be negative)
        cov_est: cov
        big_a: A value
        big_b: B value

        CDF(x) = exp{-[1-k(x-u)/a]**(1/k)}
        where m = u+a*A, s = a*B, cov = s/m

        a = m*cov/B, u = m-a*A

    Returns:

    """
    try:
        a_est = abs(mean_est) * cov_est / big_b
        u_est = abs(mean_est) - a_est * big_a
    except TypeError:
        logging.warning('mean_est:{}, cov_est:{}, big_a:{}, big_b:{}'.format(
            mean_est, cov_est, big_a, big_b))
    else:
        return a_est, u_est


def calc_big_a_b_values(shape_k):
    """

    Args:
        shape_k: parameter k of GEV III (JHD)
        CDF(x) = exp{-[1-k(x-u)/a]**(1/k)}

        m = u + a*A, s = a * B
        where m: mean, s: std, u: location factor, a:scale factor
        A = (1/k)[1-Gamma(1+k)]
        B = (1/k)*sqrt[Gamma(1+2k)-Gamma(1+k)^2]

    Returns: A, B

    """
    assert 0.0 < shape_k < 0.5
    big_a = (1.0 - gamma(1.0 + shape_k)) / shape_k
    big_b = sqrt(gamma(1.0 + 2 * shape_k) - gamma(1.0 + shape_k) ** 2) / shape_k

    return big_a, big_b


def compute_logarithmic_mean_stddev(m, stddev):
    """ compute mean of log x with mean and std. of x
    Args:
        m: arithmetic mean of x
        stddev: arithmetic standard deviation of x

        mu = 2*log(m) - 0.5*log(v + m**2)
        std = sqrt(log(V/m**2 +1))

        if m is zero, then return -999, 0.0

    Returns: mean and std of log x

    """
    assert m >= 0.0
    assert stddev >= 0.0

    if m:
        mu = 2 * log(m) - 0.5 * log(stddev**2.0 + m**2.0)
        std = sqrt(log(stddev**2.0 / m**2.0 + 1))
    else:
        mu = -999
        std = 0.0

    return mu, std


def sample_lognorm_given_mean_stddev(m, stddev, rnd_state):
    """
    generate rv following lognorm dist
    Args:
        m: mean of x
        stddev: std of x
        rnd_state
        size: size of rv (default: 1)

    Returns:

    """
    mu_, std_ = compute_logarithmic_mean_stddev(m, stddev)
    return sample_lognormal(mu_, std_, rnd_state)


def compute_arithmetic_mean_stddev(m, stddev):
    """ compute arithmetic mean and std of x

    Args:
        m: mean of log x
        stddev: std of log x

    Returns: arithmetic mean, std of x

    """
    assert stddev >= 0, 'std can not be less than zero'
    mean_x = exp(m + 0.5 * stddev * stddev)
    std_x = mean_x * sqrt(exp(stddev**2.0) - 1.0)
    return mean_x, std_x

