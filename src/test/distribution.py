# adjust python path so we may import things from peer packages
import sys
import os.path
if __name__ == '__main__': sys.path.append(os.path.abspath('../'))
from core import damage, scenario, database, csvarray, house

import unittest

class Test(unittest.TestCase):
    def setUp(self):
        damage.configure('../model.db', False)
        s = scenario.Scenario(20, 40.0, 120.0, 60.0, '2')
        s.setHouseName('Group 4 House')
        s.setOpt_SampleSeed(True)
        s.setOpt_CPISample(True)
        s.setOpt_DmgDistribute(True)
        s.setOpt_DmgPlotFragility(False)
        s.setOpt_DmgPlotVuln(False)
        self.s = s
        
    def tearDown(self):
        database.db().close()
        
    def testDamageDistribution(self):
        print '\n'
                
        # find all files named "dist_n.csv" with matching exp file...
        testfilebases = []
        import glob
        
        for filename in glob.glob('./dist_[0-9].csv'):
            basename = os.path.splitext(filename)[0]
            if os.path.isfile(basename + '_exp.csv'):
                testfilebases.append(basename)

        for filename in glob.glob('./dist_[0-9][0-9].csv'):
            basename = os.path.splitext(filename)[0]
            if os.path.isfile(basename + '_exp.csv'):
                testfilebases.append(basename)

        # for each test found
        for base in testfilebases:
            print 'running test: ', base

            mySim = damage.WindDamageSimulator(self.s, None)
            mySim.sample_house_and_wind_params()

            # damage conns from input
            ctgname = ''
            infile = base + '.csv'
            linecount = 0
            for line in open(infile, 'r'):
                line = line.rstrip()
                if linecount == 0:
                    ctgname = line
                else:
                    house.connByZoneTypeMap[line][ctgname].result_damaged = True
                linecount += 1

            # distribute
            mySim.redistribute_damage(house.ctgMap[ctgname])

            # check results (might be better to walk whole grid?)
            resfile = base + '_exp.csv'
            x, linecount = csvarray.readArrayFromCSV(resfile, "S50,f4")
            for row in x:
                conn = house.connByZoneTypeMap[row[0]][ctgname]
                msg = '%s expected k=%f got %f' % (row[0], row[1], conn.result_ksunk)
                self.assertAlmostEqual(conn.result_ksunk, row[1], places=7, msg=msg)

        
if __name__ == "__main__":
    unittest.main()











