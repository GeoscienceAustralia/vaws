"""
    Scenario module - user input to model run (file or gui)

    Note: regional shielding may be indirectly disabled by setting
    'regional_shielding_factor' to 1.0
    Note: differential shielding may be indirectly disabled by setting
    'building_spacing' to 0
"""
import os
import sys
import ConfigParser
import numpy as np

import house
import database
import debris


class Scenario(object):

    # lookup table mapping (0-7) to wind direction desc
    dirs = ['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE', 'Random']

    def __init__(self, ni, wsmin, wsmax, wsnumsteps, tcat):
        self.num_iters = ni
        self.wind_speed_max = wsmax
        self.wind_speed_min = wsmin
        self.wind_speed_num_steps = wsnumsteps
        self.terrain_category = tcat

        self._house = None
        self._region = None
        self._construction_levels = dict()
        self._fragility_thresholds = dict()

        self._source_items = None
        self._regional_shielding_factor = None
        self._building_spacing = None
        self._wind_dir_index = None
        self._debris_radius = None
        self._debris_angle = None
        self._debris_extension = None
        self._flight_time_mean = None
        self._flight_time_stddev = None

        # self.red_V = 40.0
        # self.blue_V = 80.0
        self._flags = dict()

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def updateModel(self):
        for ctg in self.house.conn_type_groups:
            if ctg.distribution_order >= 0:
                ctg_name = 'ctg_{}'.format(ctg.group_name)
                ctg.enabled = self.flags.get(ctg_name, True)
            else:
                ctg.enabled = False

    def setOptCTGEnabled(self, ctg_name, opt):
        key_name = 'ctg_{}'.format(ctg_name)
        self.flags[key_name] = opt

    def getConstructionLevel(self, name):
        if name in self.construction_levels:
            return (self.construction_levels[name]['probability'],
                    self.construction_levels[name]['mean_factor'],
                    self.construction_levels[name]['cov_factor'])

    def setConstructionLevel(self, name, prob, mf, cf):
        if name in self.construction_levels:
            self.construction_levels[name] = dict(zip(
                ['probability', 'mean_factor', 'cov_factor'],
                [prob, mf, cf]))

    def sampleConstructionLevel(self):
        rv = np.random.random_integers(0, 100)
        cumprob = 0.0
        for key, value in self.construction_levels.iteritems():
            cumprob += value['probability'] * 100.0
            if rv <= cumprob:
                break
        return key, value['mean_factor'], value['cov_factor']

    def get_wind_dir_index(self):
        if self.wind_dir_index == 8:
            return np.random.random_integers(0, 7)
        else:
            return self.wind_dir_index

    @property
    def regional_shielding_factor(self):
        return self._regional_shielding_factor

    @regional_shielding_factor.setter
    def regional_shielding_factor(self, value):
        self._regional_shielding_factor = value

    @property
    def building_spacing(self):
        return self._building_spacing

    @building_spacing.setter
    def building_spacing(self, value):
        self._building_spacing = value

    @property
    def flight_time_mean(self):
        return self._flight_time_mean

    @flight_time_mean.setter
    def flight_time_mean(self, value):
        self._flight_time_mean = value

    @property
    def flight_time_stddev(self):
        return self._flight_time_stddev

    @flight_time_stddev.setter
    def flight_time_stddev(self, value):
        self._flight_time_stddev = value

    @property
    def debris_radius(self):
        return self._debris_radius

    @debris_radius.setter
    def debris_radius(self, value):
        self._debris_radius = value

    @property
    def debris_angle(self):
        return self._debris_angle

    @debris_angle.setter
    def debris_angle(self, value):
        self._debris_angle = value

    @property
    def debris_extension(self):
        return self._debris_extension

    @debris_extension.setter
    def debris_extension(self, value):
        self._debris_extension = value

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, region_name):
        if region_name in ['Capital_city', 'Tropical_town']:
            self._region = debris.qryDebrisRegionByName(region_name)
        else:
            self._region = debris.qryDebrisRegionByName('Capital_city')
            print('8(for Random) is set for wind_dir_index by default')

    @property
    def construction_levels(self):
        return self._construction_levels

    @construction_levels.setter
    def construction_levels(self, value):
        assert isinstance(value, dict)
        self._construction_levels = value

    @property
    def fragility_thresholds(self):
        return self._fragility_thresholds

    @fragility_thresholds.setter
    def fragility_thresholds(self, value):
        assert isinstance(value, dict)
        self._fragility_thresholds = value

    @property
    def flags(self):
        return self._flags

    @flags.setter
    def flags(self, value):
        assert isinstance(value, dict)
        self._flags = value

    @property
    def house(self):
        return self._house

    @house.setter
    def house(self, house_name):
        self._house = house.queryHouseWithName(house_name)

    @property
    def source_items(self):
        return self._source_items

    @source_items.setter
    def source_items(self, value):
        self._source_items = value

    @property
    def wind_dir_index(self):
        return self._wind_dir_index

    @wind_dir_index.setter
    def wind_dir_index(self, wind_dir_str):
        if wind_dir_str in type(self).dirs:
            self._wind_dir_index = Scenario.dirs.index(wind_dir_str)
        else:
            print('8(for Random) is set for wind_dir_index by default')
            self._wind_dir_index = 8

    def setOpt_SampleSeed(self, b=True):
        self.flags['random_seed'] = b

    def setOpt_DmgDistribute(self, b=True):
        self.flags['dmg_distribute'] = b

    def setOpt_DmgPlotVuln(self, b=True):
        self.flags['dmg_plot_vuln'] = b

    def setOpt_DmgPlotFragility(self, b=True):
        self.flags['dmg_plot_fragility'] = b

    def setOpt_Debris(self, b=True):
        self.flags['debris'] = b

    def setOpt_DebrisStaggeredSources(self, b=True):
        self.flags['debris_staggered_sources'] = b

    def setOpt_DiffShielding(self, b=True):
        self.flags['diff_shielding'] = b

    def setOpt_ConstructionLevels(self, b=True):
        self.flags['construction_levels'] = b

    def setOpt_WaterIngress(self, b=True):
        self.flags['water_ingress'] = b

    def setOpt_VulnFitLog(self, b=True):
        self.flags['vul_fit_log'] = b

    def storeToCSV(self, cfg_file):

        config = ConfigParser.RawConfigParser()

        # When adding sections or items, add them in the reverse order of
        # how you want them to be displayed in the actual file.
        # In addition, please note that using RawConfigParser's and the raw
        # mode of ConfigParser's respective set functions, you can assign
        # non-string values to keys internally, but will receive an error
        # when attempting to write to a file or when you get it in non-raw
        # mode. SafeConfigParser does not allow such assignments to take place.

        key = 'main'
        config.add_section(key)
        config.set(key, 'no_simulations', self.num_iters)
        config.set(key, 'wind_speed_min', self.wind_speed_min)
        config.set(key, 'wind_speed_max', self.wind_speed_max)
        config.set(key, 'wind_speed_steps', self.wind_speed_num_steps)
        config.set(key, 'terrain_cat', self.terrain_category)
        config.set(key, 'house_name', self.house.house_name)
        config.set(key, 'regional_shielding_factor',
                   self.regional_shielding_factor)
        config.set(key, 'wind_fixed_dir', type(self).dirs[self.wind_dir_index])
        config.set(key, 'region_name', self.region.name)

        key = 'options'
        config.add_section(key)
        for sub_key in self.flags:
            config.set(key, sub_key, self.flags.get(sub_key))

        key = 'fragility_thresholds'
        config.add_section(key)
        config.set(key, 'states', ', '.join(self.fragility_thresholds.keys()))
        config.set(key, 'thresholds',
                   ', '.join(str(x) for x in
                             self.fragility_thresholds.values()))

        key = 'debris'
        config.add_section(key)
        config.set(key, 'source_items', self.source_items)
        config.set(key, 'building_spacing', self.building_spacing)
        config.set(key, 'debris_radius', self.debris_radius)
        config.set(key, 'debris_angle', self.debris_angle)
        config.set(key, 'debris_extension', self.debris_extension)
        config.set(key, 'flight_time_mean', self.flight_time_mean)
        config.set(key, 'flight_time_stddev', self.flight_time_stddev)

        key = 'construction_levels'
        config.add_section(key)
        _levels = []
        _probabilities = []
        _mean_factor = []
        _cov_factor = []
        for sub_key, value in self.construction_levels.iteritems():
            _levels.append(sub_key)
            _probabilities.append(str(value['probability']))
            _mean_factor.append(str(value['mean_factor']))
            _cov_factor.append(str(value['cov_factor']))

        config.set(key, 'levels', ', '.join(_levels))
        config.set(key, 'probabilities', ', '.join(_probabilities))
        config.set(key, 'mean_factors', ', '.join(_mean_factor))
        config.set(key, 'cov_factors', ', '.join(_cov_factor))

        with open(cfg_file, 'wb') as configfile:
            config.write(configfile)


def loadFromCSV(cfg_file):
    """
    read them all in as strings into a simple dict
    Args:
        cfg_file: file containing scenario information

    Returns: an instance of Scenario class

    """

    if not os.path.isfile(cfg_file):
        msg = 'Error: file {} not found'.format(cfg_file)
        sys.exit(msg)

    cfg = ConfigParser.ConfigParser()
    cfg.optionxform = str
    cfg.read(cfg_file)

    key = 'main'
    s = Scenario(cfg.getint(key, 'no_simulations'),
                 cfg.getfloat(key, 'wind_speed_min'),
                 cfg.getfloat(key, 'wind_speed_max'),
                 cfg.getint(key, 'wind_speed_steps'),
                 cfg.get(key, 'terrain_cat'))

    s.house = cfg.get(key, 'house_name')
    s.regional_shielding_factor = cfg.getfloat(key, 'regional_shielding_factor')
    s.wind_dir_index = cfg.get(key, 'wind_fixed_dir')
    s.region = cfg.get(key, 'region_name')

    key = 'options'
    for sub_key, value in cfg.items('options'):
        s.flags[sub_key] = cfg.getboolean(key, sub_key)

    key = 'construction_levels'
    if s.flags[key]:
        levels = [x.strip() for x in cfg.get(key, 'levels').split(',')]
        probabilities = [float(x) for x in cfg.get(key,
                                                   'probabilities').split(',')]
        mean_factors = [float(x) for x in cfg.get(key,
                                                  'mean_factors').split(',')]
        cov_factors = [float(x) for x in cfg.get(key, 'cov_factors').split(',')]

        for i, level in enumerate(levels):
            s.construction_levels.setdefault(
                level, {})['probability'] = probabilities[i]
            s.construction_levels[level]['mean_factor'] = mean_factors[i]
            s.construction_levels[level]['cov_factor'] = cov_factors[i]
    else:
        s.construction_levels = {'low': {'probability': 0.33,
                                         'mean_factor': 0.9,
                                         'cov_factor': 0.58},
                                 'medium': {'probability': 0.34,
                                            'mean_factor': 1.0,
                                            'cov_factor': 0.58},
                                 'high': {'probability': 0.33,
                                          'mean_factor': 1.1,
                                          'cov_factor': 0.58}}
        print('default construction level distribution is used')

    key = 'fragility_thresholds'
    if cfg.has_section(key):
        states = [x.strip() for x in cfg.get(key, 'states').split(',')]
        thresholds = [float(x) for x in cfg.get(key, 'thresholds').split(',')]
        s.fragility_thresholds = dict(zip(states, thresholds))
    else:
        s.fragility_thresholds = {'slight': 0.15,
                                  'medium': 0.45,
                                  'severe': 0.6,
                                  'complete': 0.9}
        print('default fragility thresholds is used')

    key = 'debris'
    if s.flags[key]:
        s.source_items = cfg.getint(key, 'source_items')
        s.building_spacing = cfg.getfloat(key, 'building_spacing')
        s.debris_radius = cfg.getfloat(key, 'debris_radius')
        s.debris_angle = cfg.getfloat(key, 'debris_angle')
        s.debris_extension = cfg.getfloat(key, 'debris_extension')
        s.flight_time_mean = cfg.getfloat(key, 'flight_time_mean')
        s.flight_time_stddev = cfg.getfloat(key, 'flight_time_stddev')

    # if 'red_V' in args:
    #     s.red_V = float(args['red_V'])
    #     del args['red_V']
    #
    # if 'blue_V' in args:
    #     s.blue_V = float(args['blue_V'])
    #     del args['blue_V']

    # for ctg in s.house.conn_type_groups:
    #     print('{}:{}'.format(ctg.enabled, ctg.group_name))

    s.updateModel()

    # for ctg in s.house.conn_type_groups:
    #     print('{}:{}'.format(ctg.enabled, ctg.group_name))

    return s


if __name__ == '__main__': 

    import unittest

    database.configure()

    path_, _ = os.path.split(os.path.abspath(__file__))

    class MyTestCase(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.file1 = os.path.abspath(os.path.join(path_,
                                                     '../scenarios/carl1.cfg'))
            cls.file2 = os.path.abspath(os.path.join(path_,
                                                     '../scenarios/carl2.cfg'))
            cls.file3 = os.path.abspath(os.path.join(path_,
                                                     '../test/temp.cfg'))

        def test_nocomments(self):
            s1 = loadFromCSV(self.file1)
            self.assertEquals(s1.wind_dir_index, 3)
            
        def test_equals_op(self):
            s1 = loadFromCSV(self.file1)
            s2 = loadFromCSV(self.file2)
            self.assertNotEquals(s1, s2)

        def test_debrisopt(self):
            s1 = loadFromCSV(self.file1)
            # s1.storeToCSV(self.file3)
            self.assertEquals(s1.flags['debris'], True)

        def test_wind_directions(self):
            s = loadFromCSV(self.file1)
            s.wind_dir_index = 'Random'
            dirs = []
            for i in range(100):
                dirs.append(s.get_wind_dir_index())
            wd1 = dirs[0]
            for wd in dirs:
                if wd != wd1:
                    break
            self.assertNotEqual(wd, wd1)
            s.wind_dir_index = 'SW'
            self.assertEqual(s.wind_dir_index, 1)

        def test_wateringress(self):
            s1 = loadFromCSV(self.file1)
            self.assertTrue(s1.flags['water_ingress'])
            s1.flags['water_ingress'] = False
            self.assertFalse(s1.flags['water_ingress'])

        def test_ctgenables(self):
            s = loadFromCSV(self.file1)
            self.assertTrue(s.flags['ctg_{}'.format('rafter')])
            s.setOptCTGEnabled('batten', False)
            self.assertFalse(s.flags['ctg_{}'.format('batten')])

            s.storeToCSV(self.file3)
            s2 = loadFromCSV(self.file3)
            self.assertFalse(s2.flags['ctg_{}'.format('batten')])
            self.assertTrue(s2.flags['ctg_{}'.format('sheeting')])

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
            self.assertAlmostEquals(
                s.construction_levels['low']['mean_factor'], 0.42)
            
    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
