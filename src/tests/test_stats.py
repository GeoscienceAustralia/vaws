import unittest
import matplotlib.pyplot as plt
import numpy as np

from vaws.stats import compute_logarithmic_mean_stddev, \
    compute_arithmetic_mean_stddev, sample_lognormal, calc_big_a_b_values, \
    calc_parameters_gev, sample_gev


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


if __name__ == '__main__':
    unittest.main()
# suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
# unittest.TextTestRunner(verbosity=2).run(suite)
