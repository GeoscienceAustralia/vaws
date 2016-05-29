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
import pandas as pd
from collections import OrderedDict

# import database
import terrain


class Scenario(object):

    # lookup table mapping (0-7) to wind direction desc
    dirs = ['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE', 'Random']

    def __init__(self, no_sims, wind_min, wind_max, wind_steps, terrain_cat):

        self.no_sims = no_sims
        self.wind_speed_min = wind_min
        self.wind_speed_max = wind_max
        self.wind_speed_num_steps = wind_steps
        self.terrain_category = terrain_cat

        self._parallel = None
        self._house_name = None
        self._region_name = None
        self._db_file = None
        self._construction_levels = OrderedDict()
        self._fragility_thresholds = None

        self._source_items = None
        self._regional_shielding_factor = None
        self._building_spacing = None
        self._wind_dir_index = None
        self._debris_radius = None
        self._debris_angle = None
        self._debris_extension = None
        self._flight_time_mean = None
        self._flight_time_stddev = None

        self._file_cpis = None
        self._file_debris = None
        self._file_frag = None
        self._file_water = None
        self._file_damage = None
        self._file_dmg = None
        self._file_dmg_idx = None

        self._wind_profile = None

        self._speeds = None

        # self.red_V = 40.0
        # self.blue_V = 80.0
        self._flags = dict()

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    # def updateModel(self):
    #     for ctg in self.house.conn_type_groups:
    #         if ctg.distribution_order >= 0:
    #             ctg_name = 'ctg_{}'.format(ctg.group_name)
    #             ctg.enabled = self.flags.get(ctg_name, True)
    #         else:
    #             ctg.enabled = False

    def setOptCTGEnabled(self, ctg_name, opt):
        key_name = 'ctg_{}'.format(ctg_name)
        self.flags[key_name] = opt

    def getConstructionLevel(self, name):
        try:
            return (self.construction_levels[name]['probability'],
                    self.construction_levels[name]['mean_factor'],
                    self.construction_levels[name]['cov_factor'])
        except KeyError:
            msg = '{} not found in the construction_levels'.format(name)
            raise KeyError(msg)

    def setConstructionLevel(self, name, prob, mf, cf):
        self.construction_levels[name] = OrderedDict(zip(
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
    def parallel(self):
        return self._parallel

    @parallel.setter
    def parallel(self, value):
        assert isinstance(value, bool)
        self._parallel = value

    @property
    def db_file(self):
        return self._db_file

    @db_file.setter
    def db_file(self, value):
        assert isinstance(value, str)
        self._db_file = value

    @property
    def speeds(self):
        return self._speeds

    @speeds.setter
    def speeds(self, value):
        assert isinstance(value, np.ndarray)
        self._speeds = value

    @property
    def wind_profile(self):
        return self._wind_profile

    @wind_profile.setter
    def wind_profile(self, value):
        assert isinstance(value, dict)
        self._wind_profile = value

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
    def region_name(self):
        return self._region_name

    @region_name.setter
    def region_name(self, value):
        try:
            assert value in ['Capital_city', 'Tropical_town']
        except AssertionError:
            self._region_name = 'Capital_city'
            print('Capital_city is set for region_name by default')
        else:
            self._region_name = value

    @property
    def construction_levels(self):
        return self._construction_levels

    @construction_levels.setter
    def construction_levels(self, value):
        assert isinstance(value, OrderedDict)
        self._construction_levels = value

    @property
    def fragility_thresholds(self):
        return self._fragility_thresholds

    @fragility_thresholds.setter
    def fragility_thresholds(self, value):
        assert isinstance(value, pd.DataFrame)
        self._fragility_thresholds = value

    @property
    def flags(self):
        return self._flags

    @flags.setter
    def flags(self, value):
        assert isinstance(value, dict)
        self._flags = value

    @property
    def house_name(self):
        return self._house_name

    @house_name.setter
    def house_name(self, house_name):
        self._house_name = house_name

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
        try:
            self._wind_dir_index = Scenario.dirs.index(wind_dir_str.upper())
        except ValueError:
            print('8(i.e., Random) is set for wind_dir_index by default')
            self._wind_dir_index = 8

    @property
    def file_cpis(self):
        return self._file_cpis

    @file_cpis.setter
    def file_cpis(self, file_name):
        self._file_cpis = open(file_name, 'w')
        self._file_cpis.write('Simulated House #, Cpi Changed At\n')
        self._file_cpis.close()
        self._file_cpis = open(file_name, 'a')

    @property
    def file_debris(self):
        return self._file_debris

    @file_debris.setter
    def file_debris(self, file_name):
        self._file_debris = open(file_name, 'w')
        header = ('Wind Speed(m/s),% Houses Internally Pressurized,'
                  '% Debris Damage Mean\n')
        self._file_debris.write(header)
        self._file_debris.close()
        self._file_debris = open(file_name, 'a')

    @property
    def file_dmg_idx(self):
        return self._file_dmg_idx

    @file_dmg_idx.setter
    def file_dmg_idx(self, file_name):
        self._file_dmg_idx = file_name

    @property
    def file_damage(self):
        return self._file_damage

    @file_damage.setter
    def file_damage(self, file_name):
        self._file_damage = open(file_name, 'w')
        header = 'Simulated House #,Wind Speed(m/s),Wind Direction,'
        # list_ = []
        # for ctg in self.house.conn_type_groups:
        #     if ctg.enabled:
        #         for ct in ctg.conn_types:
        #             list_.append(ct.connection_type)
        # header += ','.join(list_)
        # header += '\n'
        self._file_damage.write(header)

    @property
    def file_dmg(self):
        return self._file_dmg

    @file_dmg.setter
    def file_dmg(self, file_name):
        self._file_dmg = open(file_name, 'w')

    @property
    def file_water(self):
        return self._file_water

    @file_water.setter
    def file_water(self, file_name):
        self._file_water = open(file_name, 'w')
        header_ = ('V,Envelope DI,Water Damage,Damage Scenario,'
                   'Water Damage Cost,WaterCosting\n')
        self._file_water.write(header_)
        self._file_water.close()
        self._file_water = open(file_name, 'a')

    @property
    def file_frag(self):
        return self._file_frag

    @file_frag.setter
    def file_frag(self, file_name):
        # self._file_frag = open(file_name, 'w')
        # header = ('Slight Median,Slight Beta,Medium Median,Median Beta,'
        #            'Severe Median,Severe Beta,Complete Median,Complete Beta\n')
        # self._file_frag.write(header)
        self._file_frag = file_name

    '''
    # used by main.pyw

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
    '''

    def storeToCSV(self, cfg_file):

        config = ConfigParser.RawConfigParser()

        key = 'main'
        config.add_section(key)
        config.set(key, 'db_file', self.db_file)
        config.set(key, 'parallel', self.parallel)
        config.set(key, 'no_simulations', self.no_sims)
        config.set(key, 'wind_speed_min', self.wind_speed_min)
        config.set(key, 'wind_speed_max', self.wind_speed_max)
        config.set(key, 'wind_speed_steps', self.wind_speed_num_steps)
        config.set(key, 'terrain_cat', self.terrain_category)
        config.set(key, 'house_name', self.house_name)
        config.set(key, 'regional_shielding_factor',
                   self.regional_shielding_factor)
        config.set(key, 'wind_fixed_dir', type(self).dirs[self.wind_dir_index])
        config.set(key, 'region_name', self.region_name)

        key = 'options'
        config.add_section(key)
        for sub_key in self.flags:
            config.set(key, sub_key, self.flags.get(sub_key))

        key = 'fragility_thresholds'
        config.add_section(key)
        config.set(key, 'states', ', '.join(self.fragility_thresholds.index))
        config.set(key, 'thresholds',
                   ', '.join(str(x) for x in
                             self.fragility_thresholds['threshold'].values))

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

    conf = ConfigParser.ConfigParser()
    conf.optionxform = str
    conf.read(cfg_file)

    path_cfg_file = os.path.dirname(os.path.realpath(cfg_file))

    key = 'main'
    s = Scenario(conf.getint(key, 'no_simulations'),
                 conf.getfloat(key, 'wind_speed_min'),
                 conf.getfloat(key, 'wind_speed_max'),
                 conf.getint(key, 'wind_speed_steps'),
                 conf.get(key, 'terrain_cat'))

    s.speeds = np.linspace(s.wind_speed_min,
                           s.wind_speed_max,
                           s.wind_speed_num_steps)

    s.db_file = os.path.join(path_cfg_file, conf.get(key, 'db_file'))

    s.parallel = conf.getboolean(key, 'parallel')
    s.house_name = conf.get(key, 'house_name')
    s.regional_shielding_factor = conf.getfloat(key,
                                                'regional_shielding_factor')
    s.wind_dir_index = conf.get(key, 'wind_fixed_dir')
    s.region_name = conf.get(key, 'region_name')

    key = 'options'
    for sub_key, value in conf.items('options'):
        s.flags[sub_key] = conf.getboolean(key, sub_key)

    key = 'construction_levels'
    if s.flags[key]:
        levels = [x.strip() for x in conf.get(key, 'levels').split(',')]
        probabilities = [float(x) for x in conf.get(key,
                                                    'probabilities').split(',')]
        mean_factors = [float(x) for x in conf.get(key,
                                                   'mean_factors').split(',')]
        cov_factors = [float(x) for x in conf.get(key,
                                                  'cov_factors').split(',')]

        for i, level in enumerate(levels):
            s.construction_levels.setdefault(
                level, {})['probability'] = probabilities[i]
            s.construction_levels[level]['mean_factor'] = mean_factors[i]
            s.construction_levels[level]['cov_factor'] = cov_factors[i]
    else:
        s.construction_levels = OrderedDict()
        s.construction_levels.setdefault('low', {})['probability'] = 0.33
        s.construction_levels.setdefault('medium', {})['probability'] = 0.34
        s.construction_levels.setdefault('high', {})['probability'] = 0.33

        s.construction_levels['low']['mean_factor'] = 0.9
        s.construction_levels['medium']['mean_factor'] = 1.0
        s.construction_levels['high']['mean_factor'] = 1.1

        s.construction_levels['low']['cov_factor'] = 0.58
        s.construction_levels['medium']['cov_factor'] = 0.58
        s.construction_levels['high']['cov_factor'] = 0.58

        print('default construction level distribution is used')

    key = 'fragility_thresholds'
    if conf.has_section(key):
        states = [x.strip() for x in conf.get(key, 'states').split(',')]
        thresholds = [float(x) for x in conf.get(key, 'thresholds').split(',')]
    else:
        states = ['slight', 'medium', 'severe', 'complete']
        thresholds = [0.15, 0.45, 0.6, 0.9]
        print('default fragility thresholds is used')

    s.fragility_thresholds = pd.DataFrame(thresholds, index=states,
                                          columns=['threshold'])
    s.fragility_thresholds['color'] = ['b', 'g', 'y', 'r']

    key = 'debris'
    if s.flags[key]:
        s.source_items = conf.getint(key, 'source_items')
        s.building_spacing = conf.getfloat(key, 'building_spacing')
        s.debris_radius = conf.getfloat(key, 'debris_radius')
        s.debris_angle = conf.getfloat(key, 'debris_angle')
        s.debris_extension = conf.getfloat(key, 'debris_extension')
        s.flight_time_mean = conf.getfloat(key, 'flight_time_mean')
        s.flight_time_stddev = conf.getfloat(key, 'flight_time_stddev')

    s.wind_profile = terrain.populate_wind_profile_by_terrain()

    # if 'red_V' in args:
    #     s.red_V = float(args['red_V'])
    #     del args['red_V']
    #
    # if 'blue_V' in args:
    #     s.blue_V = float(args['blue_V'])
    #     del args['blue_V']

    # for ctg in s.house.conn_type_groups:
    #     print('{}:{}'.format(ctg.enabled, ctg.group_name))

    # s.updateModel()

    # for ctg in s.house.conn_type_groups:
    #     print('{}:{}'.format(ctg.enabled, ctg.group_name))

    return s


if __name__ == '__main__': 

    import unittest

    # database.configure()

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

        # def test_equals_op(self):
        #     s1 = loadFromCSV(self.file1)
        #     s2 = loadFromCSV(self.file2)
        #     self.assertNotEquals(s1, s2)

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
