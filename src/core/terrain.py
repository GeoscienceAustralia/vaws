'''
    Terrain Module - reference storage for terrain gust envelope profiles
        - loaded from database.
        - imported from mzcat_terrain_x.CSV extracted from JCU provided PDF --> constr()
'''
import numpy
from sqlalchemy.sql import select, and_
import database

z = (3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 17.0, 20.0, 25.0, 30.0)

cached_tcats = {}


def populate_terrain_cache():
    cached_tcats.clear()
    for tc in ['2', '2.5', '3', '5']:
        cached_profiles = []
        for p in xrange(1, 11):
            cached_mzcats = []
            terrain_table = database.db.terrain_table
            s = select([terrain_table.c.m], and_(terrain_table.c.tcat == tc, terrain_table.c.profile == p))
            for m in database.db.engine.execute(s):
                cached_mzcats.append(m[0])
            cached_profiles.append(cached_mzcats)
        cached_tcats[tc] = cached_profiles
            

def calculateMZCAT(tc, p, h):
    m = cached_tcats[tc][p-1]
    return numpy.interp(h, z, m)
    
# unit tests
if __name__ == '__main__':    
    import unittest
   
    class TerrainTestCase(unittest.TestCase):
        def setUp(self):
            populate_terrain_cache()
            
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
            for tcat in ['2', '2.5', '3', '5']:
                for p in xrange(1, 11):
                    zcats = []
                    heights = numpy.linspace(0.5, 30.0, 100)
                    for h in heights:
                        zcats.append(calculateMZCAT(tcat, p, h))
                    plt.plot(zcats, heights)
                plt.axis([0, 1.4, 0, 35.0])
                plt.show()
            
    suite = unittest.TestLoader().loadTestsFromTestCase(TerrainTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
    



