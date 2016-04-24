"""
    Scenario module - user input to model run (file or gui)

    Note: regional shielding may be indirectly disabled by setting
    'regional_shielding_factor' to 1.0
    Note: differential shielding may be indirectly disabled by setting
    'building_spacing' to 0
"""
import house
import database
import pandas as pd
import debris
from numpy.random import random_integers


class Scenario(object):
    # lookup table mapping (0-7) to wind direction desc
    dirs = ['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE', 'Random']

    def __init__(self, ni, wsmin, wsmax, wsnumsteps, tcat):
        self.house = None
        self.region = None
        self.num_iters = ni
        self.wind_speed_max = wsmax
        self.wind_speed_min = wsmin
        self.wind_speed_num_steps = wsnumsteps
        self.terrain_category = tcat

        # FIXME HARDCODED
        self.construction_levels = [['low', 0.33, 0.9, 0.58],
                                    ['medium', 0.33, 1.0, 0.58],
                                    ['high', 0.33, 1.1, 0.58]]

        # FIXME HARDCODED
        self.fragility_thresholds = {'slight': 0.15, 
                                     'medium': 0.45, 
                                     'severe': 0.6, 
                                     'complete': 0.9}
        self.source_items = 100
        self.regional_shielding_factor = 1.0
        self.building_spacing = 20
        self.wind_dir_index = 8
        self.debris_radius = 100
        self.debris_angle = 45
        self.debris_extension = 0
        self.flighttime_mean = 2.0
        self.flighttime_stddev = 0.8
        self.red_V = 40.0
        self.blue_V = 80.0
        self.flags = {'SCEN_SEED_RANDOM' : False,
                      'SCEN_DMG_DISTRIBUTE' : False,
                      'SCEN_DMG_PLOT_VULN' : True,
                      'SCEN_DMG_PLOT_FRAGILITY' : True,
                      'SCEN_DEBRIS' : True,
                      'SCEN_DEBRIS_STAGGERED_SOURCES' : False,
                      'SCEN_DIFF_SHIELDING' : False,
                      'SCEN_CONSTRUCTION_LEVELS' : True,
                      'SCEN_WATERINGRESS' : True,
                      'SCEN_VULN_FITLOG' : False}

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def updateModel(self):
        for ctg in self.house.conn_type_groups:
            ctg.enabled = True if ctg.distribution_order >= 0 else False
            if ctg.enabled:
                ctg.enabled = self.getOptCTGEnabled(ctg.group_name)

    def getOptCTGEnabled(self, ctg_name):
        key_name = 'ctg_%s' % ctg_name
        return self.flags.get(key_name, True)

    def setOptCTGEnabled(self, ctg_name, opt):
        key_name = 'ctg_%s' % ctg_name
        self.flags[key_name] = opt

    def setWindDirection(self, windDirStr):
        if windDirStr in Scenario.dirs:
            self.wind_dir_index = Scenario.dirs.index(windDirStr)

    def getConstructionLevel(self, name):
        for level in self.construction_levels:
            if level[0] == name:
                return level[1], level[2], level[3]

    def setConstructionLevel(self, name, prob, mf, cf):
        for level in self.construction_levels:
            if level[0] == name:
                level[1] = prob
                level[2] = mf
                level[3] = cf
                break

    def sampleConstructionLevel(self):
        d100 = random_integers(0, 100)
        cumprob = 0.0
        for clevel in self.construction_levels:
            cumprob += (clevel[1] * 100.0)
            if d100 <= cumprob:
                break
        return clevel[0], clevel[2], clevel[3]

    def getWindDirIndex(self):
        if self.wind_dir_index == 8:
            return random_integers(0,7)
        else:
            return self.wind_dir_index

    def setRegionalShielding(self, rsf):
        if rsf > 0:
            self.regional_shielding_factor = rsf
        
    def setBuildingSpacing(self, bs):
        if bs > 0:
            self.building_spacing = bs
        
    def setDebrisRadius(self, v):
        self.debris_radius = v
        
    def setDebrisAngle(self, v):
        self.debris_angle = v
        
    def setRegionName(self, regionName):
        self.region = debris.qryDebrisRegionByName(regionName)
        
    def setHouseName(self, house_name):
        self.house = house.queryHouseWithName(house_name)
        
    def getHouseHeight(self):
        return self.house.height
    
    def getOpt_SampleSeed(self):
        return self.flags['SCEN_SEED_RANDOM']

    def setOpt_SampleSeed(self, b=True):
        self.flags['SCEN_SEED_RANDOM'] = b

    def getOpt_DmgDistribute(self):
        return self.flags['SCEN_DMG_DISTRIBUTE']

    def setOpt_DmgDistribute(self, b=True):
        self.flags['SCEN_DMG_DISTRIBUTE'] = b

    def getOpt_DmgPlotVuln(self):
        return self.flags['SCEN_DMG_PLOT_VULN']

    def setOpt_DmgPlotVuln(self, b=True):
        self.flags['SCEN_DMG_PLOT_VULN'] = b

    def getOpt_DmgPlotFragility(self):
        return self.flags['SCEN_DMG_PLOT_FRAGILITY']

    def setOpt_DmgPlotFragility(self, b=True):
        self.flags['SCEN_DMG_PLOT_FRAGILITY'] = b

    def getOpt_Debris(self):
        return self.flags['SCEN_DEBRIS']

    def setOpt_Debris(self, b=True):
        self.flags['SCEN_DEBRIS'] = b

    def getOpt_DebrisStaggeredSources(self):
        return self.flags['SCEN_DEBRIS_STAGGERED_SOURCES']

    def setOpt_DebrisStaggeredSources(self, b=True):
        self.flags['SCEN_DEBRIS_STAGGERED_SOURCES'] = b

    def getOpt_DiffShielding(self):
        return self.flags['SCEN_DIFF_SHIELDING']

    def setOpt_DiffShielding(self, b=True):
        self.flags['SCEN_DIFF_SHIELDING'] = b

    def getOpt_ConstructionLevels(self):
        return self.flags['SCEN_CONSTRUCTION_LEVELS']

    def setOpt_ConstructionLevels(self, b=True):
        self.flags['SCEN_CONSTRUCTION_LEVELS'] = b

    def getOpt_WaterIngress(self):
        return self.flags['SCEN_WATERINGRESS']

    def setOpt_WaterIngress(self, b=True):
        self.flags['SCEN_WATERINGRESS'] = b

    def getOpt_VulnFitLog(self):
        return self.flags['SCEN_VULN_FITLOG']

    def setOpt_VulnFitLog(self, b=True):
        self.flags['SCEN_VULN_FITLOG'] = b

    def storeToCSV(self, fileName):
        try:
            f = open(fileName, 'w')
            lines=[]
            lines.append('N,%d\n' % self.num_iters)
            lines.append('house_name,%s\n' % self.house.house_name)
            if self.region is not None:
                lines.append('region_name,%s\n' % self.region.name)
            lines.append('wind_speed_min,%f\n' % self.wind_speed_min)
            lines.append('wind_speed_max,%f\n' % self.wind_speed_max)
            lines.append('wind_speed_steps,%d\n' % self.wind_speed_num_steps)
            lines.append('terrain_cat,%s\n' % self.terrain_category)
            lines.append('wind_fixed_dir,%s\n' % self.dirs[self.wind_dir_index])
            lines.append('source_items,%d\n' % self.source_items)
            lines.append('regional_shielding_factor,%f\n' % self.regional_shielding_factor)
            lines.append('building_spacing,%f\n' % self.building_spacing)
            lines.append('debris_radius,%f\n' % self.debris_radius)
            lines.append('debris_angle,%f\n' % self.debris_angle)
            lines.append('debris_extension,%f\n' % self.debris_extension)
            lines.append('flighttime_mean,%f\n' % self.flighttime_mean)
            lines.append('flighttime_stddev,%f\n' % self.flighttime_stddev)
            lines.append('red_V,%f\n' % self.red_V)
            lines.append('blue_V,%f\n' % self.blue_V)
            
            for level in self.construction_levels:
                lines.append('level_%s_probability,%f\n' % (level[0], level[1]))
                lines.append('level_%s_mean_factor,%f\n' % (level[0], level[2]))
                lines.append('level_%s_cov_factor,%f\n' % (level[0], level[3]))
                
            for level_key in self.fragility_thresholds:
                lines.append('fragthresh_%s,%f\n' % (level_key, self.fragility_thresholds[level_key]))
                
            for key in self.flags.keys():
                lines.append(key + ',' + str(eval("self.flags['" + key + "']")) + '\n')
            f.writelines(lines)
            f.flush()
            f.close()
        except Exception, e:
            print 'import_model(): %s' % e


def loadFromCSV(fileName):
    """
    read them all in as strings into a simple dict
    Args:
        fileName: file containing scenario information

    Returns: an instance of Scenario class

    """

    args = pd.read_csv(fileName, header=None).set_index(0).to_dict()[1]
    # for key, value in args.iteritems():
    #     try:
    #         args[key] = float(value)
    #     except ValueError:
    #         try:
    #             args[Key] = value == 'True'
    #         pass

    mandatory_keys = ['house_name', 'wind_speed_min', 'wind_speed_max',
                      'wind_speed_steps', 'N', 'terrain_cat']
    for key in mandatory_keys:
        if key not in args:
            msg = "Invalid Scenario File Format - {} is missing".format(key)
            raise Exception(msg)

    s = Scenario(int(args['N']),
                 float(args['wind_speed_min']), 
                 float(args['wind_speed_max']), 
                 int(args['wind_speed_steps']),
                 args['terrain_cat'])
    
    if 'wind_fixed_dir' in args:
        s.setWindDirection(args['wind_fixed_dir'])
        del args['wind_fixed_dir']
        
    if 'region_name' in args:
        s.setRegionName(args['region_name'])
        del args['region_name']
    else:
        s.setRegionName('Capital_city')
        
    for level in s.construction_levels:
        key = 'level_%s_probability' % level[0]
        if key in args:
            level[1] = float(args[key])
            del args[key]
        key = 'level_%s_mean_factor' % level[0]
        if key in args:
            level[2] = float(args[key])
            del args[key]
        key = 'level_%s_cov_factor' % level[0]
        if key in args:
            level[3] = float(args[key])
            del args[key]
        
    for level_key in s.fragility_thresholds:
        key = 'fragthresh_%s' % level_key 
        if key in args:
            s.fragility_thresholds[level_key] = float(args[key]) 
        
    if 'source_items' in args:
        s.source_items = int(args['source_items'])
        del args['source_items'] 
        
    if 'regional_shielding_factor' in args:
        rsf = float(args['regional_shielding_factor'])
        if rsf == 0: rsf = 1.0
        s.setRegionalShielding(rsf)
        del args['regional_shielding_factor']
        
    if 'building_spacing' in args:
        bs = float(args['building_spacing'])
        if bs == 0: bs = 20
        s.setBuildingSpacing(bs)
        del args['building_spacing']
        
    if 'debris_radius' in args:
        s.setDebrisRadius(float(args['debris_radius']))
        del args['debris_radius']
        
    if 'debris_angle' in args:
        s.setDebrisAngle(float(args['debris_angle']))
        del args['debris_angle']
        
    if 'debris_extension' in args:
        s.debris_extension = float(args['debris_extension'])
        del args['debris_extension']
        
    if 'flighttime_mean' in args:
        s.flighttime_mean = float(args['flighttime_mean'])
        del args['flighttime_mean']
        
    if 'flighttime_stddev' in args:
        s.flighttime_stddev = float(args['flighttime_stddev'])
        del args['flighttime_stddev']
        
    if 'red_V' in args:
        s.red_V = float(args['red_V'])
        del args['red_V']
        
    if 'blue_V' in args:
        s.blue_V = float(args['blue_V'])
        del args['blue_V']
        
    s.setHouseName(args['house_name'])
    
    for mkey in mandatory_keys:
        del args[mkey]
       
    for key in args:
        if args[key].lower() == 'true':
            s.flags[key] = True
        elif args[key].lower() == 'false':
            s.flags[key] = False
        else:
            s.flags[key] = args[key]
    
    s.updateModel()        
    return s


if __name__ == '__main__': 
    import unittest
    import os

    database.configure()

    path_, _ = os.path.split(os.path.abspath(__file__))

    class MyTestCase(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.file1 = os.path.abspath(os.path.join(path_,
                                                     '../scenarios/carl1.csv'))
            cls.file2 = os.path.abspath(os.path.join(path_,
                                                     '../scenarios/carl2.csv'))
            cls.file3 = os.path.abspath(os.path.join(path_,
                                                     '../test/temp.csv'))

        def test_nocomments(self):
            s1 = loadFromCSV(self.file1)
            self.assertEquals(s1.getWindDirIndex(), 3)
            
        def test_equals_op(self):
            s1 = loadFromCSV(self.file1)
            s2 = loadFromCSV(self.file2)
            self.assertNotEquals(s1, s2)
            
        def test_debrisopt(self):
            s1 = loadFromCSV(self.file1)
            s1.storeToCSV(self.file3)
            self.assertEquals(s1.getOpt_Debris(), True)
            
        def test_wind_directions(self):
            s = loadFromCSV(self.file1)
            s.setWindDirection('Random')
            dirs = []
            for i in range(100):
                dirs.append(s.getWindDirIndex())
            wd1 = dirs[0]
            for wd in dirs:
                if wd != wd1:
                    break
            self.assertNotEqual(wd, wd1)
            s.setWindDirection('SW')
            self.assertEqual(s.getWindDirIndex(), 1)
            
        def test_wateringress(self):
            s1 = loadFromCSV(self.file1)
            self.assertTrue(s1.getOpt_WaterIngress())
            s1.setOpt_WaterIngress(False)
            self.assertFalse(s1.getOpt_WaterIngress())
            
        def test_ctgenables(self):
            s = loadFromCSV(self.file1)
            self.assertTrue(s.getOptCTGEnabled('fred'))
            s.setOptCTGEnabled('batten', False)
            self.assertFalse(s.getOptCTGEnabled('batten'))

            s.storeToCSV(self.file3)
            s2 = loadFromCSV(self.file3)
            self.assertFalse(s2.getOptCTGEnabled('batten'))
            self.assertTrue(s2.getOptCTGEnabled('sheeting'))
            
        def test_construction_levels(self):
            s1 = loadFromCSV(self.file1)
            s1.setConstructionLevel('low', 0.33, 0.75, 0.78)
            counts = {'low': 0, 'medium': 0, 'high': 0}
            for i in range(1000):
                level, mf, cf = s1.sampleConstructionLevel()
                if level == 'low':
                    self.assertAlmostEquals(mf, 0.75)
                    self.assertAlmostEquals(cf, 0.78)
                counts[level] += 1
            s1.setConstructionLevel('low', 0.33, 0.42, 0.78)

            s1.storeToCSV(self.file3)
            s = loadFromCSV(self.file3)
            self.assertAlmostEquals(s.construction_levels[0][2], 0.42)            
            
    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
