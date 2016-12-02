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
    dirs = ['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE', 'RANDOM']

    def __init__(self, cfg_file=None, output_path=None):

        self.cfg_file = cfg_file
        self.output_path = output_path

        self.no_sims = None
        self.wind_speed_min = None
        self.wind_speed_max = None
        self.wind_speed_num_steps = None
        self.speeds = None
        self.idx_speeds = None
        self.terrain_category = None

        self.db_file = None
        self.parallel = None
        self.house_name = None
        self._region_name = None
        self.construction_levels = OrderedDict()
        self.fragility_thresholds = None

        self.source_items = None
        self.regional_shielding_factor = None
        self.building_spacing = None
        self._wind_dir_index = None
        self.debris_radius = None
        self.debris_angle = None
        self.debris_extension = None
        self.flight_time_mean = None
        self.flight_time_stddev = None

        self._file_dmg_freq_by_conn_type = None

        self._file_house_cpi = None  #
        self._file_wind_debris = None
        self.file_dmg_idx = None
        self.file_dmg_map_by_conn_type = None
        self.file_frag = None
        self._file_water = None
        self.file_dmg_pct_by_conn_type = None
        self.file_dmg_area_by_conn_grp = None
        self.file_repair_cost_by_conn_grp = None
        self.file_dmg_by_conn = None
        self.file_strength_by_conn = None
        self.file_deadload_by_conn = None
        self.file_dmg_dist_by_conn = None
        self.file_rnd_parameters = None

        self.wind_profile = None

        # self._rnd_state = None

        self._list_conn = None
        self._list_conn_type = None
        self._list_conn_type_group = None

        self.red_v = 54.0
        self.blue_v = 95.0
        self.flags = dict()

        if not os.path.isfile(cfg_file):
            msg = 'Error: file {} not found'.format(cfg_file)
            sys.exit(msg)
        else:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            self.read_config()

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
        key_name = 'conn_type_group_{}'.format(ctg_name)
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

    # def sampleConstructionLevel(self):
    #     rv = np.random.random_integers(0, 100)
    #     cumprob = 0.0
    #     for key, value in self.construction_levels.iteritems():
    #         cumprob += value['probability'] * 100.0
    #         if rv <= cumprob:
    #             break
    #     return key, value['mean_factor'], value['cov_factor']

    def get_wind_dir_index(self):
        if self.wind_dir_index == 8:
            return np.random.random_integers(0, 7)
            # return self.rnd_state.random_integers(0, 7)
        else:
            return self.wind_dir_index

    def read_config(self):

        conf = ConfigParser.ConfigParser()
        conf.optionxform = str
        conf.read(self.cfg_file)

        path_cfg_file = os.path.dirname(os.path.realpath(self.cfg_file))

        key = 'main'
        self.no_sims = conf.getint(key, 'no_simulations')
        self.wind_speed_min = conf.getfloat(key, 'wind_speed_min')
        self.wind_speed_max = conf.getfloat(key, 'wind_speed_max')
        self.wind_speed_num_steps = conf.getint(key, 'wind_speed_steps')
        self.terrain_category = conf.get(key, 'terrain_cat')

        self.speeds = np.linspace(self.wind_speed_min,
                                  self.wind_speed_max,
                                  self.wind_speed_num_steps)

        self.idx_speeds = range(self.wind_speed_num_steps)

        self.db_file = os.path.join(path_cfg_file, conf.get(key, 'db_file'))

        self.parallel = conf.getboolean(key, 'parallel')
        self.house_name = conf.get(key, 'house_name')
        self.regional_shielding_factor = conf.getfloat(
            key, 'regional_shielding_factor')
        self.wind_dir_index = conf.get(key, 'wind_fixed_dir')
        self.region_name = conf.get(key, 'region_name')

        key = 'options'
        for sub_key, value in conf.items('options'):
            self.flags[sub_key] = conf.getboolean(key, sub_key)

        key = 'construction_levels'
        if self.flags[key]:
            levels = [x.strip() for x in conf.get(key, 'levels').split(',')]
            probabilities = [float(x) for x in conf.get(
                key, 'probabilities').split(',')]
            mean_factors = [float(x) for x in conf.get(
                key, 'mean_factors').split(',')]
            cov_factors = [float(x) for x in conf.get(
                key, 'cov_factors').split(',')]

            for i, level in enumerate(levels):
                self.construction_levels.setdefault(
                    level, {})['probability'] = probabilities[i]
                self.construction_levels[level]['mean_factor'] = mean_factors[i]
                self.construction_levels[level]['cov_factor'] = cov_factors[i]
        else:
            self.construction_levels = OrderedDict()
            self.construction_levels.setdefault('low', {})['probability'] = 0.33
            self.construction_levels.setdefault('medium', {})['probability'] = 0.34
            self.construction_levels.setdefault('high', {})['probability'] = 0.33

            self.construction_levels['low']['mean_factor'] = 0.9
            self.construction_levels['medium']['mean_factor'] = 1.0
            self.construction_levels['high']['mean_factor'] = 1.1

            self.construction_levels['low']['cov_factor'] = 0.58
            self.construction_levels['medium']['cov_factor'] = 0.58
            self.construction_levels['high']['cov_factor'] = 0.58

            print('default construction level distribution is used')

        key = 'fragility_thresholds'
        if conf.has_section(key):
            states = [x.strip() for x in conf.get(key, 'states').split(',')]
            thresholds = [float(x) for x in conf.get(key, 'thresholds').split(',')]
        else:
            states = ['slight', 'medium', 'severe', 'complete']
            thresholds = [0.15, 0.45, 0.6, 0.9]
            print('default fragility thresholds is used')

        self.fragility_thresholds = pd.DataFrame(thresholds, index=states,
                                              columns=['threshold'])
        self.fragility_thresholds['color'] = ['b', 'g', 'y', 'r']

        key = 'debris'
        if self.flags[key]:
            self.source_items = conf.getint(key, 'source_items')
            self.building_spacing = conf.getfloat(key, 'building_spacing')
            self.debris_radius = conf.getfloat(key, 'debris_radius')
            self.debris_angle = conf.getfloat(key, 'debris_angle')
            self.debris_extension = conf.getfloat(key, 'debris_extension')
            self.flight_time_mean = conf.getfloat(key, 'flight_time_mean')
            self.flight_time_stddev = conf.getfloat(key, 'flight_time_stddev')

        self.wind_profile = terrain.populate_wind_profile_by_terrain()

        key = 'heatmap'
        try:
            self.red_v = conf.getfloat(key, 'red_V')
            self.blue_v = conf.getfloat(key, 'blue_V')
        except ConfigParser.NoSectionError:
            print('default value is used for heatmap')

        if self.output_path:

            # wind speed at pressurised failure
            self.file_house_cpi = os.path.join(self.output_path, 'house_cpi.csv')

            # failure frequency by connection type with wind speed ?
            # previously houses_damaged_at_v.csv
            self.file_dmg_freq_by_conn_type = os.path.join(self.output_path,
                                                        'dmg_freq_by_conn_type.csv')
            # previously houses_damage_map.csv
            self.file_dmg_map_by_conn_type = os.path.join(self.output_path,
                                                       'dmg_map_by_conn_type.csv')
            self.file_frag = os.path.join(self.output_path, 'fragilities.csv')
            self.file_water = os.path.join(self.output_path, 'wateringress.csv')
            # previously house_damage.csv
            self.file_dmg_pct_by_conn_type = os.path.join(self.output_path,
                                                       'dmg_pct_by_conn_type.csv')
            self.file_wind_debris = os.path.join(self.output_path, 'wind_debris.csv')
            self.file_dmg_idx = os.path.join(self.output_path, 'house_dmg_idx.csv')

            self.file_dmg_area_by_conn_grp = os.path.join(self.output_path,
                                                       'dmg_area_by_conn_grp.csv')
            self.file_repair_cost_by_conn_grp = os.path.join(self.output_path,
                                                          'repair_cost_by_conn_grp.csv')
            self.file_dmg_by_conn = os.path.join(self.output_path, 'dmg_by_conn.csv')

            self.file_strength_by_conn = os.path.join(self.output_path,
                                                   'strength_by_conn.csv')

            self.file_deadload_by_conn = os.path.join(self.output_path,
                                                   'deadload_by_conn.csv')

            self.file_dmg_dist_by_conn = os.path.join(self.output_path,
                                                   'dmg_dist_by_conn.csv')
            self.file_rnd_parameters = os.path.join(self.output_path,
                                                 'random_parameters.csv')

        else:
            print 'output path is not assigned'

    @property
    def list_conn_type_group(self):
        return self._list_conn_type_group

    @list_conn_type_group.setter
    def list_conn_type_group(self, value):
        assert isinstance(value, set)
        self._list_conn_type_group = value

    @property
    def list_conn_type(self):
        return self._list_conn_type

    @list_conn_type.setter
    def list_conn_type(self, value):
        assert isinstance(value, set)
        self._list_conn_type = value

    @property
    def list_conn(self):
        return self._list_conn

    @list_conn.setter
    def list_conn(self, value):
        assert isinstance(value, set)
        self._list_conn = value

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
    def wind_dir_index(self):
        return self._wind_dir_index

    @wind_dir_index.setter
    def wind_dir_index(self, wind_dir_str):
        try:
            self._wind_dir_index = Scenario.dirs.index(wind_dir_str.upper())
        except ValueError:
            print('8(i.e., RANDOM) is set for wind_dir_index by default')
            self._wind_dir_index = 8

    @property
    def file_house_cpi(self):
        return self._file_house_cpi

    @file_house_cpi.setter
    def file_house_cpi(self, file_name):
        self._file_house_cpi = open(file_name, 'w')
        self._file_house_cpi.write('Simulated House #, Cpi Changed At\n')
        self._file_house_cpi.close()
        self._file_house_cpi = open(file_name, 'a')

    @property
    def file_wind_debris(self):
        return self._file_wind_debris

    @file_wind_debris.setter
    def file_wind_debris(self, file_name):
        self._file_wind_debris = open(file_name, 'w')
        header = ('Wind Speed(m/s),% Houses Internally Pressurized,'
                  '% Debris Damage Mean\n')
        self._file_wind_debris.write(header)
        self._file_wind_debris.close()
        self._file_wind_debris = open(file_name, 'a')

    @property
    def file_dmg_freq_by_conn_type(self):
        return self._file_dmg_freq_by_conn_type

    @file_dmg_freq_by_conn_type.setter
    def file_dmg_freq_by_conn_type(self, file_name):
        self._file_dmg_freq_by_conn_type = open(file_name, 'w')
        self._file_dmg_freq_by_conn_type.write('Number of Damaged Houses\n')
        self._file_dmg_freq_by_conn_type.write('Num Houses,{:d}\n'.format(self.no_sims))
        self._file_dmg_freq_by_conn_type.write('Wind Direction,{}\n'.format(
            self.dirs[self.wind_dir_index]))
        self._file_dmg_freq_by_conn_type.close()
        self._file_dmg_freq_by_conn_type = open(file_name, 'a')

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


if __name__ == '__main__':

    import unittest

    path_, _ = os.path.split(os.path.abspath(__file__))

    class MyTestCase(unittest.TestCase):

        @classmethod
        def setUpClass(cls):

            cls.output_path = './output'
            cls.scenario_filename1 = os.path.abspath(os.path.join(path_,
                                                     '../scenarios/carl1.cfg'))

            cls.scenario_filename2 = os.path.abspath(os.path.join(path_,
                                                     '../scenarios/carl2.cfg'))

            cls.scenario_filename3 = os.path.abspath(os.path.join(path_,
                                                     '../test/temp.cfg'))

        def test_nocomments(self):
            s1 = Scenario(cfg_file=self.scenario_filename1)
            self.assertEquals(s1.wind_dir_index, 3)

        # def test_equals_op(self):
        #     s1 = loadFromCSV(self.file1)
        #     s2 = loadFromCSV(self.file2)
        #     self.assertNotEquals(s1, s2)

        def test_debrisopt(self):
            s1 = Scenario(cfg_file=self.scenario_filename1)
            # s1.storeToCSV(self.file3)
            self.assertEquals(s1.flags['debris'], True)

        def test_wind_directions(self):
            s = Scenario(cfg_file=self.scenario_filename1)
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
            s1 = Scenario(cfg_file=self.scenario_filename1)
            self.assertTrue(s1.flags['water_ingress'])
            s1.flags['water_ingress'] = False
            self.assertFalse(s1.flags['water_ingress'])

        def test_ctgenables(self):
            s = Scenario(cfg_file=self.scenario_filename1)
            self.assertTrue(s.flags['conn_type_group_{}'.format('rafter')])
            s.setOptCTGEnabled('batten', False)
            self.assertFalse(s.flags['conn_type_group_{}'.format('batten')])

            s.storeToCSV(self.scenario_filename3)
            s2 = Scenario(cfg_file=self.scenario_filename3)
            self.assertFalse(s2.flags['conn_type_group_{}'.format('batten')])
            self.assertTrue(s2.flags['conn_type_group_{}'.format('sheeting')])

        def test_construction_levels(self):
            s1 = Scenario(cfg_file=self.scenario_filename1)
            s1.setConstructionLevel('low', 0.33, 0.42, 0.78)

            s1.storeToCSV(self.scenario_filename3)
            s = Scenario(cfg_file=self.scenario_filename3)
            self.assertAlmostEquals(
                s.construction_levels['low']['mean_factor'], 0.42)


    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
