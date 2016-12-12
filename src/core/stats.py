from math import log, exp, sqrt


def compute_logarithmic_mean_stddev(m, stddev):
    """ compute mean of log x with mean and std. of x
    Args:
        m: arithmetic mean of x
        stddev: arithmetic standard deviation of x

    Returns: mean and std of log x

    mu = 2*log(m) - 0.5*log(v + m**2)

    """

    try:
        mu = 2 * log(m) - 0.5 * log(stddev**2.0 + m**2.0)
        std = sqrt(log(stddev**2.0 / m**2.0 + 1))
    except ValueError as e:
        print '{}: zero returned for mu, std'.format(e)
        return 0.0, 0.0
    else:
        return mu, std


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

    class MyTestCase(unittest.TestCase):

        def test_compute_logarithmic_mean_stdev(self):
            mu, std = compute_logarithmic_mean_stddev(1.0, 0.5)
            self.assertAlmostEqual(mu, -0.1116, places=4)
            self.assertAlmostEqual(std, 0.4724, places=4)

            mu, std = compute_logarithmic_mean_stddev(0.0, 0.0)
            self.assertAlmostEqual(mu, 0.0, places=4)
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

    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
