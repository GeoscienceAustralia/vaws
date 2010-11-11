# adjust python path so we may import things from peer packages
import sys
import os.path
if __name__ == '__main__': sys.path.append(os.path.abspath('../'))
from core import damage, scenario, database
from matplotlib.pyplot import show, plot

import unittest

loopCount = 0

def damage_callback(V, di, percLoops):
    global loopCount
    loopCount += 1
    if loopCount % 30 == 0:
        print '%.2f perc' % percLoops
    return True

class Test(unittest.TestCase):
    def setUp(self):
        damage.configure('../model.db', False)
        s = scenario.Scenario(20, 40.0, 120.0, 60.0, '2')
        s.setHouseName('Group 4 House')
        s.setOpt_SampleSeed(True)
        s.setOpt_DmgDistribute(True)
        s.setOpt_DmgPlotFragility(False)
        s.setOpt_DmgPlotVuln(False)
        self.s = s
        
    def tearDown(self):
        database.db().close()
        
    def testHouseNumberSensitivityOnCurve(self):
        print '\n'
        final1 = []
        final2 = []
        houseNums = [1,10,30]
        
        for numHouses in houseNums:
            print 'running simulator over %d houses' % numHouses
            self.s.num_iters = numHouses
            mySim = damage.WindDamageSimulator(self.s, None)
            runTime = mySim.simulator_mainloop()
            self.assertNotEqual(runTime, None)
            mySim.fit_vuln_curve()
            print 'Simulation ran in %s', runTime
            print 'Curve coeffs were: %.4f, %.4f' % (mySim.A_final[0], mySim.A_final[1])
            
            final1.append(mySim.A_final[0])
            final2.append(mySim.A_final[1])
            
            #scatter(mySim.A_final[0], mySim.A_final[1], s=20, c='r', marker='o', label="Means")
            #mySim.plot_vulnerability(None, False, False, 'N=%d'%numHouses)
            
        plot(houseNums, final1)
        show()  
        plot(houseNums, final2)
        show()
            
        #output.plot_wind_event_show(self.s.num_iters, self.s.wind_speed_min, self.s.wind_speed_max, None)

if __name__ == "__main__":
    unittest.main()











