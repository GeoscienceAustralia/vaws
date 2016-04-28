"""
    Terrain Module - reference storage for terrain gust envelope profiles
        - loaded from database.
        - imported from mzcat_terrain_x.CSV extracted from JCU provided PDF --> constr()
"""
import os
import numpy as np
import pandas as pd

terrain_cats = ['2', '2.5', '3', '5']
heights = [3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 17.0, 20.0, 25.0, 30.0]

wind_profile = dict()


def populate_wind_profile_by_terrain():

    path = '/'.join(__file__.split('/')[:-1])
    for terrain_cat in terrain_cats:
        file_ = os.path.join(path,
                             '../../data/mzcat_terrain_' + terrain_cat + '.csv')
        wind_profile[terrain_cat] = pd.read_csv(file_, skiprows=1, header=None,
                                                index_col=0).to_dict('list')


def calculateMZCAT(terrain_cat, profile_idx, height):
    """

    Args:
        terrain_cat: terrain category
        profile_idx: profile index
        height: height

    Returns:

    """
    return np.interp(height, heights,
                     wind_profile[terrain_cat][profile_idx])


# unit tests
if __name__ == '__main__':
    import unittest

    class TerrainTestCase(unittest.TestCase):
        def setUp(self):
            populate_wind_profile_by_terrain()

        def test_basic(self):
            self.assertAlmostEqual(calculateMZCAT('2', 1, 5), 0.995)
            self.assertAlmostEqual(calculateMZCAT('2.5', 1, 5), 0.915)
            self.assertAlmostEqual(calculateMZCAT('3', 1, 5), 0.936)
            self.assertAlmostEqual(calculateMZCAT('5', 1, 5), 0.887)

        def test_calc(self):
            num = 50000
            for i in xrange(num):
                mz = calculateMZCAT('2', 3, 4.3)
            print '\n', mz

        def test_plot(self):
            import matplotlib.pyplot as plt
            for terrain_cat in ['2', '2.5', '3', '5']:
                for p in xrange(1, 11):
                    zcats = []
                    heights = np.linspace(0.5, 30.0, 100)
                    for h in heights:
                        zcats.append(calculateMZCAT(terrain_cat, p, h))
                    plt.plot(zcats, heights)
                plt.axis([0, 1.4, 0, 35.0])
                plt.show()

    suite = unittest.TestLoader().loadTestsFromTestCase(TerrainTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)

