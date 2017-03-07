import unittest
import matplotlib.pyplot as plt
import numpy as np

from vaws.curve import single_exponential_given_V, vulnerability_weibull


class MyTestCase(unittest.TestCase):

    def test_single_exponential_given_V(self):
        # region: capital city
        alpha_, beta_ = 0.1585, 3.8909  #
        x_arr = np.arange(10, 120, 1.0)
        y1 = single_exponential_given_V(beta_, alpha_, x_arr)
        y2 = vulnerability_weibull(alpha_, beta_, x_arr, flag='cdf')
        np.testing.assert_almost_equal(y1, y2, decimal=3)

        plt.figure()
        plt.plot(x_arr, y1, 'b-', x_arr, y2, 'r--')
        # plt.show()
        plt.pause(1.0)
        plt.close()

    def test_single_exponential_given_V2(self):
        # region: Tropical town
        alpha_, beta_ = 0.10304, 4.18252
        x_arr = np.arange(10, 120, 1.0)
        y1 = single_exponential_given_V(beta_, alpha_, x_arr)
        y2 = vulnerability_weibull(alpha_, beta_, x_arr, flag='cdf')
        np.testing.assert_almost_equal(y1, y2, decimal=3)

        plt.figure()
        plt.plot(x_arr, y1, 'b-', x_arr, y2, 'r--')
        # plt.show()
        plt.pause(1.0)
        plt.close()

    def test_compare_two_approaches(self):
        # region: Tropical town
        alpha_, beta_ = 0.10304, 4.18252

        # wind increment 5.0
        x_arr = np.arange(10, 120, 5.0)
        y1 = single_exponential_given_V(beta_, alpha_, x_arr)
        y1_diff = np.diff(y1)
        y2 = vulnerability_weibull(alpha_, beta_, x_arr[1:], flag='pdf')

        # wind increment 1.0
        x_arr_1 = np.arange(10, 120, 1.0)
        y1_1 = single_exponential_given_V(beta_, alpha_, x_arr_1)
        y1_diff_1 = np.diff(y1_1)
        y2_1 = vulnerability_weibull(alpha_, beta_, x_arr_1[1:], flag='pdf')

        # wind increment 0.5
        x_arr_2 = np.arange(10, 120, 0.5)
        y1_2 = single_exponential_given_V(beta_, alpha_, x_arr_2)
        y1_diff_2 = np.diff(y1_2)
        y2_2 = vulnerability_weibull(alpha_, beta_, x_arr_2[1:], flag='pdf')

        plt.figure()
        plt.plot(x_arr[1:], y1_diff, 'b-', x_arr[1:], y2, 'r-',
                 x_arr_1[1:], y1_diff_1, 'b--', x_arr_1[1:], y2_1, 'r--',
                 x_arr_2[1:], y1_diff_2, 'b.', x_arr_2[1:], y2_2, 'r.')
        plt.legend(['wind increment: 5.0', 'wind increment: 5.0',
                    'wind increment: 1.0', 'wind increment: 1.0',
                    'wind increment: 0.5', 'wind increment: 0.5'],
                   loc=2)
        plt.pause(1.0)
        plt.close()


if __name__ == '__main__':
    unittest.main()

# suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
# unittest.TextTestRunner(verbosity=2).run(suite)



