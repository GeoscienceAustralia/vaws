from math import log, exp, sqrt, gamma, copysign
from scipy.stats import genextreme
from numpy import isclose


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
        return rnd_state.lognormal(mu_lnx, std_lnx)
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
        print 'mean_est:{}, cov_est:{}, big_a:{}, big_b:{}'.format(
            mean_est, cov_est, big_a, big_b)
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
    assert 0 < shape_k < 0.5
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

# unit tests
if __name__ == '__main__':
    import unittest
    import matplotlib.pyplot as plt
    import numpy as np

    class MyTestCase(unittest.TestCase):

        def test_compute_logarithmic_mean_stdev(self):
            mu, std = compute_logarithmic_mean_stddev(1.0, 0.5)
            self.assertAlmostEqual(mu, -0.1116, places=4)
            self.assertAlmostEqual(std, 0.4724, places=4)

            mu, std = compute_logarithmic_mean_stddev(0.0, 0.0)
            self.assertAlmostEqual(mu, -999, places=4)
            self.assertAlmostEqual(std, 0.0, places=4)

            m, stddev = 70.0, 14.0
            m2, stddev2 = compute_logarithmic_mean_stddev(m, stddev)
            self.assertAlmostEqual(stddev2, 0.1980422)
            self.assertAlmostEqual(m2, 4.228884885)

        def test_compute_arithmetic_mean_stdev(self):
            mu, std = compute_arithmetic_mean_stddev(0.0, 0.5)
            self.assertAlmostEqual(mu, 1.1331, places=4)
            self.assertAlmostEqual(std, 0.6039, places=4)

            mu, std = compute_arithmetic_mean_stddev(0.0, 0.0)
            self.assertAlmostEqual(mu, 1.0, places=4)
            self.assertAlmostEqual(std, 0.0, places=4)

        def test_sample_logrnormal(self):
            rnd_state = np.random.RandomState(1)

            # zero mean and std
            mu, std = compute_logarithmic_mean_stddev(0.0, 0.0)
            self.assertAlmostEqual(sample_lognormal(mu, std, rnd_state), 0.0,
                                   places=2)

            # zero std
            mu, std = compute_logarithmic_mean_stddev(1.0, 0.0)
            self.assertAlmostEqual(sample_lognormal(mu, std, rnd_state), 1.0,
                                   places=2)

            # zero std
            mu, std = compute_logarithmic_mean_stddev(4.0, 0.0)
            self.assertAlmostEqual(sample_lognormal(mu, std, rnd_state), 4.0,
                                   places=2)

        def test_calc_big_a_b_values(self):
            big_a, big_b = calc_big_a_b_values(shape_k=0.1)
            self.assertAlmostEqual(big_a, 0.48649, places=4)
            self.assertAlmostEqual(big_b, 1.14457, places=4)

        def test_calc(self):
            mean_est, cov_est, shape_k = 0.95, 0.07, 0.1
            big_a, big_b = calc_big_a_b_values(shape_k)
            a, u = calc_parameters_gev(mean_est, cov_est, big_a, big_b)

            self.assertAlmostEqual(big_a, 0.4865, 3)
            self.assertAlmostEqual(big_b, 1.1446, 3)
            self.assertAlmostEqual(a, 0.058, 2)
            self.assertAlmostEqual(u, 0.922, 2)

        def test_calc2(self):
            mean_est, cov_est, shape_k = -0.95, 0.07, 0.1
            big_a, big_b = calc_big_a_b_values(shape_k)
            a, u = calc_parameters_gev(mean_est, cov_est, big_a, big_b)

            self.assertAlmostEqual(big_a, 0.4865, 3)
            self.assertAlmostEqual(big_b, 1.1446, 3)
            self.assertAlmostEqual(a, 0.058, 2)
            self.assertAlmostEqual(u, 0.922, 2)

        def test_gev_calc(self):

            mean_est, cov_est, shape_k = 0.95, 0.07, 0.1
            big_a, big_b = calc_big_a_b_values(shape_k)

            rnd_state = np.random.RandomState(42)
            rv_ = sample_gev(mean_est, cov_est, big_a, big_b, shape_k,
                             rnd_state)

            self.assertAlmostEqual(rv_, 0.9230, 3)

            rv_list = []
            for i in range(1000):
                rv_ = sample_gev(mean_est, cov_est, big_a, big_b, shape_k,
                                 rnd_state)
                rv_list.append(rv_)

            plt.figure()
            plt.hist(rv_list)
            plt.show()

        def test_gev_calc2(self):
            mean_est, cov_est, shape_k = -0.95, 0.07, 0.1
            big_a, big_b = calc_big_a_b_values(shape_k)

            rnd_state = np.random.RandomState(42)
            rv_ = sample_gev(mean_est, cov_est, big_a, big_b, shape_k,
                             rnd_state)

            self.assertAlmostEqual(rv_, -0.9230, 3)

            rv_list = []
            for i in range(1000):
                rv_ = sample_gev(mean_est, cov_est, big_a, big_b, shape_k,
                                 rnd_state)
                rv_list.append(rv_)

            plt.figure()
            plt.hist(rv_list)
            plt.show()

    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
