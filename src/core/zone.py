"""
    Zone Module - reference storage
        - loaded from database.
        - imported from '../data/houses/subfolder/zones.csv'
        - holds zone area and CPE means.
        - holds runtime sampled CPE per zone.
        - calculates Cpe pressure load from wind pressure.
"""
from stats import sample_gev
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

        self.effective_area = dic_['zone_area']  # default, later can be changed
        self.cpe = None
        self.cpe_str = None
        self.cpe_eave = None

        self.pz = None
        self.pz_str = None

        if len(self.name) > 3 and self.name[0] == 'W':
            self.is_wall_zone = True
        else:
            self.is_wall_zone = False

    def sample_zone_pressure(self, wind_dir_index, cpe_cov, cpe_k,
                              cpe_str_cov, big_a, big_b, rnd_state):

        """
        Sample external Zone Pressures for sheeting, structure and eaves Cpe,
        based on TypeIII General Extreme Value distribution. Prepare effective
        zone areas for load calculations.

        Args:
            wind_dir_index:
            cpe_cov: cov of dist. of CPE for sheeting and batten
            cpe_k: shape parameter of dist. of CPE
            cpe_str_cov: cov. of dist of CPE for rafter
            big_a:
            big_b:
            rnd_state: default None

        Returns: cpe, cpe_str, cpe_eave

        """

        self.cpe = sample_gev(self.cpe_mean[wind_dir_index], cpe_cov,
                              big_a, big_b, cpe_k, rnd_state)

        self.cpe_str = sample_gev(self.cpe_str_mean[wind_dir_index],
                                  cpe_str_cov, big_a, big_b, cpe_k, rnd_state)

        self.cpe_eave = sample_gev(self.cpe_eave_mean[wind_dir_index],
                                   cpe_str_cov, big_a, big_b, cpe_k, rnd_state)

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
            pz : zone pressure applied for sheeting and batten
            pz_struct: zone pressure applied for rafter

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
        self.pz = qz * (self.cpe - self.cpi_alpha * cpi) * diff_shielding

        # calculate zone structure pressure for rafter
        self.pz_str = qz * (self.cpe_str - self.cpi_alpha * cpi
                            - self.cpe_eave) * diff_shielding


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

    from database import DatabaseManager
    from database import House as TableHouse
    import numpy as np

    class MyTestCase(unittest.TestCase):

        @classmethod
        def setUpClass(cls):

            db_file = '../test_roof_sheeting2.db'
            house_name = 'Test2'

            cls.db_house = DatabaseManager(db_file).session.query(
                TableHouse).filter_by(house_name=house_name).one()
            cls.rnd_state = np.random.RandomState(13)

        def test_zonegrid(self):
            loc = 'N12'
            gridCol, gridRow = getGridFromZoneLoc(loc)
            self.assertEquals(gridRow, 11)
            self.assertEquals(gridCol, 13)
            self.assertEquals(getZoneLocFromGrid(gridCol, gridRow), loc)

        def test_is_wall(self):
            _zone = Zone(self.db_house.zones[0])
            self.assertEqual(_zone.is_wall_zone, False)

        def test_calc_zone_pressures(self):

            _zone = Zone(self.db_house.zones[0])
            _zone.sample_zone_pressure(wind_dir_index=3,
                                       cpe_cov=self.db_house.__dict__['cpe_V'],
                                       cpe_k=self.db_house.__dict__['cpe_k'],
                                       cpe_str_cov=self.db_house.__dict__['cpe_struct_V'],
                                       big_a=0.486,
                                       big_b=1.145,
                                       rnd_state=self.rnd_state)

            self.assertAlmostEqual(_zone.cpe, -0.1084, places=4)
            self.assertAlmostEqual(_zone.cpe_eave, 0.0000, places=4)
            self.assertAlmostEqual(_zone.cpe_str, -0.0474, places=4)

            wind_speed = 40.0
            mzcat = 0.9235
            qz = 0.6 * 1.0e-3 * (wind_speed * mzcat) ** 2

            _zone.calc_zone_pressures(wind_dir_index=3,
                                      cpi=0.0,
                                      qz=qz,
                                      Ms=1.0,
                                      building_spacing=0)

            self.assertAlmostEqual(_zone.pz, -0.08876, places=4)
            self.assertAlmostEqual(_zone.pz_str, -0.03881, places=4)

    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
