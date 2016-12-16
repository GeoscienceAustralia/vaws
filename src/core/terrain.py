"""
    Terrain Module - reference storage for terrain gust envelope profiles
        - loaded from database.
        - imported from mzcat_terrain_x.CSV extracted from JCU provided PDF --> constr()
"""
import os
import numpy as np
import pandas as pd



def calculateMZCAT(wind_profile, terrain_cat, profile_idx, height):
    """

    Args:
        terrain_cat: terrain category
        profile_idx: profile index
        height: height

    Returns:

    """
    assert isinstance(wind_profile, dict)
    return np.interp(height, heights,
                     wind_profile[terrain_cat][profile_idx])


# unit tests
if __name__ == '__main__':
    import unittest

    class TerrainTestCase(unittest.TestCase):
        def setUp(self):
            self.wind_profile = populate_wind_profile_by_terrain()

        def test_basic(self):
            self.assertAlmostEqual(
                calculateMZCAT(self.wind_profile, '2', 1, 5), 0.995)
            self.assertAlmostEqual(
                calculateMZCAT(self.wind_profile, '2.5', 1, 5), 0.915)
            self.assertAlmostEqual(
                calculateMZCAT(self.wind_profile, '3', 1, 5), 0.936)
            self.assertAlmostEqual(
                calculateMZCAT(self.wind_profile, '5', 1, 5), 0.887)

        def test_calc(self):
            num = 50000
            for i in xrange(num):
                mz = calculateMZCAT(self.wind_profile, '2', 3, 4.3)
            print '\n', mz

        def test_plot(self):
            import matplotlib.pyplot as plt
            for terrain_cat in ['2', '2.5', '3', '5']:
                for p in xrange(1, 11):
                    zcats = []
                    heights = np.linspace(0.5, 30.0, 100)
                    for h in heights:
                        zcats.append(calculateMZCAT(self.wind_profile,
                                                    terrain_cat, p, h))
                    plt.plot(zcats, heights)
                plt.axis([0, 1.4, 0, 35.0])
                plt.show()

    suite = unittest.TestLoader().loadTestsFromTestCase(TerrainTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)

