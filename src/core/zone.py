"""
    Zone Module - reference storage
        - loaded from database.
        - imported from '../data/houses/subfolder/zones.csv'
        - holds zone area and CPE means.
        - holds runtime sampled CPE per zone.
        - calculates Cpe pressure load from wind pressure.
"""
import scipy.stats
import numpy
import math
from sqlalchemy import Integer, String, Float, Column, ForeignKey

import database


# hackerama to get scipy seeded
def seed_scipy(seed=42):
    myrs = numpy.random.RandomState(seed)

    def mysample(size=1):
        return myrs.uniform(size=size)

    numpy.random.sample = mysample


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


dirs = ['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE']


class Zone(database.Base):
    __tablename__ = 'zones'
    id = Column(Integer, primary_key=True)
    zone_name = Column(String)
    zone_area = Column(Float)
    coeff_N = Column(Float)
    coeff_NE = Column(Float)
    coeff_E = Column(Float)
    coeff_SE = Column(Float)
    coeff_S = Column(Float)
    coeff_SW = Column(Float)
    coeff_W = Column(Float)
    coeff_NW = Column(Float)
    struct_coeff_N = Column(Float)
    struct_coeff_NE = Column(Float)
    struct_coeff_E = Column(Float)
    struct_coeff_SE = Column(Float)
    struct_coeff_S = Column(Float)
    struct_coeff_SW = Column(Float)
    struct_coeff_W = Column(Float)
    struct_coeff_NW = Column(Float)
    eaves_coeff_N = Column(Float)
    eaves_coeff_NE = Column(Float)
    eaves_coeff_E = Column(Float)
    eaves_coeff_SE = Column(Float)
    eaves_coeff_S = Column(Float)
    eaves_coeff_SW = Column(Float)
    eaves_coeff_W = Column(Float)
    eaves_coeff_NW = Column(Float)
    leading_roof_N = Column(Integer)
    leading_roof_NE = Column(Integer)
    leading_roof_E = Column(Integer)
    leading_roof_SE = Column(Integer)
    leading_roof_S = Column(Integer)
    leading_roof_SW = Column(Integer)
    leading_roof_W = Column(Integer)
    leading_roof_NW = Column(Integer)
    cpi_alpha = Column(Float)
    wall_dir = Column(Integer)
    house_id = Column(Integer, ForeignKey('houses.id'))

    def getCpeMeanForDir(self, dir_index):
        return getattr(self, 'coeff_%s' % dirs[dir_index])

    def getCpeStructMeanForDir(self, dir_index):
        return getattr(self, 'struct_coeff_%s' % dirs[dir_index])

    def getCpeEavesMeanForDir(self, dir_index):
        return getattr(self, 'eaves_coeff_%s' % dirs[dir_index])

    def getIsLeadingRoofEdgeForDir(self, dir_index):
        return getattr(self, 'leading_roof_%s' % dirs[dir_index])

    def getIsWallZone(self):
        if len(self.zone_name) > 3 and self.zone_name[0] == 'W':
            return True
        return False

    def __repr__(self):
        return "('%s', '%f', '%f')" % (
        self.zone_name, self.zone_area, self.cpi_alpha)


def calc_A(cpe_k):
    return (1.0 / cpe_k) * (1.0 - math.gamma(1.0 + cpe_k))


def calc_B(cpe_k):
    base_ = math.pow(1.0 / cpe_k, 2) * (
        math.gamma(1.0 + 2 * cpe_k) - math.pow(math.gamma(1.0 + cpe_k), 2))
    return math.pow(base_, 0.5)


def calc_a_u(mean, cpe_V, A, B):
    if mean >= 0:
        a = (mean * cpe_V) / B
        u = mean - a * A
    else:
        mean = abs(mean)
        a = (mean * cpe_V) / B
        u = mean - a * A
    return a, u


def sample_gev(mean, A, B, cpe_V, cpe_k):
    a, u = calc_a_u(mean, cpe_V, A, B)
    if mean >= 0:
        return float(scipy.stats.genextreme.rvs(cpe_k, loc=u, scale=a, size=1))
    else:
        return float(-scipy.stats.genextreme.rvs(cpe_k, loc=u, scale=a, size=1))


def sample_zone_pressures(zones, wind_dir_index, cpe_V, cpe_k, cpe_struct_V):
    """
    Sample external Zone Pressures for sheeting, structure and eaves Cpe,
    based on TypeIII General Extreme Value distribution. Prepare effective
    zone areas for load calculations.
    """
    A = calc_A(cpe_k)
    B = calc_B(cpe_k)
    for z in zones:
        z.result_effective_area = float(z.zone_area)
        z.sampled_cpe = sample_gev(z.getCpeMeanForDir(wind_dir_index), A, B,
                                   cpe_V, cpe_k)
        z.sampled_cpe_struct = sample_gev(
            z.getCpeStructMeanForDir(wind_dir_index), A, B, cpe_struct_V, cpe_k)
        z.sampled_cpe_eaves = sample_gev(
            z.getCpeEavesMeanForDir(wind_dir_index), A, B, cpe_struct_V, cpe_k)


def calc_zone_pressures(zones, wind_dir_index, cpi, qz, Ms, building_spacing,
                        diff_shielding):
    """
    Determine wind pressure loads on each zone (to be distributed onto
    connections)
    """
    for z in zones:
        # optionally apply differential shielding
        diff_shielding = 1.0

        if building_spacing > 0 and diff_shielding:
            front_facing = z.getIsLeadingRoofEdgeForDir(wind_dir_index)
            Ms2 = math.pow(Ms, 2)
            dsn = 1.0
            dsd = 1.0
            if building_spacing == 40 and Ms >= 1.0 and front_facing == 0:
                dsd = Ms2
            elif building_spacing == 20 and front_facing == 1:
                dsd = Ms2
                if Ms <= 0.85:
                    dsn = math.pow(0.7, 2)
                else:
                    dsn = math.pow(0.8, 2)
            diff_shielding = (dsn / dsd)

            # calculate zone pressure
        z.result_pz = qz * (z.sampled_cpe - z.cpi_alpha * cpi) * diff_shielding

        # calculate zone structure pressure
        z.result_pz_struct = qz * (z.sampled_cpe_struct - z.cpi_alpha * cpi
                                   - z.sampled_cpe_eaves) * diff_shielding

    # unit tests


if __name__ == '__main__':
    import unittest


    class MyTestCase(unittest.TestCase):
        # def test_breaks(self):
        #     self.assertEquals(12, 134)

        def test_zonegrid(self):
            loc = 'N12'
            gridCol, gridRow = getGridFromZoneLoc(loc)
            self.assertEquals(gridRow, 11)
            self.assertEquals(gridCol, 13)
            self.assertEquals(getZoneLocFromGrid(gridCol, gridRow), loc)

        def test_gev_calc(self):
            A = calc_A(0.1)
            B = calc_B(0.1)
            a, u = calc_a_u(0.95, 0.07, A, B)
            self.assertAlmostEqual(a, 0.058, 2)
            self.assertAlmostEqual(u, 0.922, 2)
            self.assertAlmostEqual(A, 0.4865, 3)
            self.assertAlmostEqual(B, 1.1446, 3)


    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
