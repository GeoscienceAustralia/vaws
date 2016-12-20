"""
    Zone Module - reference storage
        - loaded from database.
        - imported from '../data/houses/subfolder/zones.csv'
        - holds zone area and CPE means.
        - holds runtime sampled CPE per zone.
        - calculates Cpe pressure load from wind pressure.
"""
from scipy.stats import genextreme
import numpy as np
from math import gamma, sqrt, copysign
import copy


class Zone(object):
    def __init__(self, inst):
        """
        Args:
            inst: instance of database.Zone
        """
        dic_ = copy.deepcopy(inst.__dict__)
        self.id = dic_['id']
        self.name = dic_['zone_name']
        self.area = dic_['zone_area']

        self.cpi_alpha = dic_['cpi_alpha']
        self.wall_dir = dic_['wall_dir']

        self.cpe_mean = dict()
        self.cpe_str_mean = dict()
        self.cpe_eave_mean = dict()
        self.is_roof_edge = dict()

        directions = ['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE']
        for idx, item in enumerate(directions):
            self.cpe_mean[idx] = dic_['coeff_{}'.format(item)]
            self.cpe_str_mean[idx] = dic_['struct_coeff_{}'.format(item)]
            self.cpe_eave_mean[idx] = dic_['eaves_coeff_{}'.format(item)]
            self.is_roof_edge[idx] = dic_['leading_roof_{}'.format(item)]

        self.result_effective_area = self.zone_area
        self.cpe = None
        self.cpe_str = None
        self.cpe_eave = None

        self.result_pz = None
        self.result_pz_str = None

    def is_wall_zone(self):
        if len(self.name) > 3 and self.name[0] == 'W':
            return True
        else:
            return False

    def sample_zone_pressures(self, wind_dir_index, cpe_cov, cpe_k,
                              cpe_str_cov, rnd_state=None):

        """
        Sample external Zone Pressures for sheeting, structure and eaves Cpe,
        based on TypeIII General Extreme Value distribution. Prepare effective
        zone areas for load calculations.

        Args:
            wind_dir_index:
            cpe_cov: cov of dist. of CPE for sheeting and batten
            cpe_k: shape parameter of dist. of CPE
            cpe_str_cov: cov. of dist of CPE for rafter
            rnd_state: default None

        Returns:

        """
        big_a, big_b = self.calc_big_a_b_values(cpe_k)

        self.cpe = self.sample_gev(self.cpe_mean[wind_dir_index], cpe_cov,
                                   big_a, big_b, cpe_k, rnd_state)

        self.cpe_str = self.sample_gev(self.cpe_str_mean[wind_dir_index],
                                       cpe_str_cov, big_a, big_b, cpe_k,
                                       rnd_state)

        self.cpe_eave = self.sample_gev(self.cpe_eave_mean[wind_dir_index],
                                        cpe_str_cov, big_a, big_b, cpe_k,
                                        rnd_state)

    def calc_zone_pressures(self, wind_dir_index, cpi, qz, Ms, building_spacing,
                            flag_diff_shielding=False):
        """
        Determine wind pressure loads (Cpe) on each zone (to be distributed onto
        connections)

        Args:
            wind_dir_index:
            cpi: internal pressure coeff
            qz:
            Ms:
            building_spacing:
            flag_diff_shielding: flag for differential shielding (default: False)

        Returns:
            result_pz : zone pressure applied for sheeting and batten
            result_pz_struct: zone pressure applied for rafter

        """

        dsn, dsd = 1.0, 1.0

        if building_spacing > 0 and flag_diff_shielding:
            front_facing = self.is_roof_edge[wind_dir_index]
            if building_spacing == 40 and Ms >= 1.0 and front_facing == 0:
                dsd = Ms ** 2.0
            elif building_spacing == 20 and front_facing == 1:
                dsd = Ms ** 2.0
                if Ms <= 0.85:
                    dsn = 0.7 ** 2.0
                else:
                    dsn = 0.8 ** 2.0

        diff_shielding = dsn / dsd

        # calculate zone pressure for sheeting and batten
        self.result_pz = qz * (self.cpe - self.cpi_alpha * cpi) * diff_shielding

        # calculate zone structure pressure for rafter
        self.result_pz_str = qz * (self.cpe_str - self.cpi_alpha * cpi
                                   - self.cpe_eave) * diff_shielding

    @staticmethod
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
        a, u = Zone.calc_parameters_gev(mean_est, cov_est, big_a, big_b)
        return copysign(genextreme.rvs(shape_k, loc=u, scale=a, size=1,
                                       random_state=rnd_state)[0], mean_est)

    @staticmethod
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
        a_est = abs(mean_est) * cov_est / big_b
        u_est = abs(mean_est) - a_est * big_a
        return a_est, u_est

    @staticmethod
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


def getZoneLocFromGrid(gridCol, gridRow):
    """
    Create a string location (eg 'A10') from zero based grid refs (col=0,
    row=11)
    """
    locX = chr(ord('A') + gridCol)
    locY = str(gridRow + 1)
    return locX + locY


def getGridFromZoneLoc(loc):
    """
    Extract 0 based grid refs from string location (eg 'A10' to 0, 11)
    """
    locCol = loc[0]
    locRow = int(loc[1:])
    gridCol = ord(locCol) - ord('A')
    gridRow = locRow - 1
    return gridCol, gridRow


# unit tests

if __name__ == '__main__':
    import unittest
    import matplotlib.pyplot as plt

    class MyTestCase(unittest.TestCase):

        def test_zonegrid(self):
            loc = 'N12'
            gridCol, gridRow = getGridFromZoneLoc(loc)
            self.assertEquals(gridRow, 11)
            self.assertEquals(gridCol, 13)
            self.assertEquals(getZoneLocFromGrid(gridCol, gridRow), loc)

        def test_calc(self):
            mean_est, cov_est, shape_k = 0.95, 0.07, 0.1
            big_a, big_b = Zone.calc_big_a_b_values(shape_k)
            a, u = Zone.calc_parameters_gev(mean_est, cov_est, big_a, big_b)

            self.assertAlmostEqual(big_a, 0.4865, 3)
            self.assertAlmostEqual(big_b, 1.1446, 3)
            self.assertAlmostEqual(a, 0.058, 2)
            self.assertAlmostEqual(u, 0.922, 2)

        def test_calc2(self):
            mean_est, cov_est, shape_k = -0.95, 0.07, 0.1
            big_a, big_b = Zone.calc_big_a_b_values(shape_k)
            a, u = Zone.calc_parameters_gev(mean_est, cov_est, big_a, big_b)

            self.assertAlmostEqual(big_a, 0.4865, 3)
            self.assertAlmostEqual(big_b, 1.1446, 3)
            self.assertAlmostEqual(a, 0.058, 2)
            self.assertAlmostEqual(u, 0.922, 2)

        def test_gev_calc(self):

            mean_est, cov_est, shape_k = 0.95, 0.07, 0.1
            big_a, big_b = Zone.calc_big_a_b_values(shape_k)

            rnd_state = np.random.RandomState(42)
            rv_ = Zone.sample_gev(mean_est, cov_est, big_a, big_b, shape_k,
                             rnd_state)

            self.assertAlmostEqual(rv_, 0.9230, 3)

            rv_list = []
            for i in range(1000):
                rv_ = Zone.sample_gev(mean_est, cov_est, big_a, big_b, shape_k,
                             rnd_state)
                rv_list.append(rv_)

            plt.figure()
            plt.hist(rv_list)
            plt.show()

        def test_gev_calc2(self):
            mean_est, cov_est, shape_k = -0.95, 0.07, 0.1
            big_a, big_b = Zone.calc_big_a_b_values(shape_k)

            rnd_state = np.random.RandomState(42)
            rv_ = Zone.sample_gev(mean_est, cov_est, big_a, big_b, shape_k,
                             rnd_state)

            self.assertAlmostEqual(rv_, -0.9230, 3)

            rv_list = []
            for i in range(1000):
                rv_ = Zone.sample_gev(mean_est, cov_est, big_a, big_b, shape_k,
                                 rnd_state)
                rv_list.append(rv_)

            plt.figure()
            plt.hist(rv_list)
            plt.show()


    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
