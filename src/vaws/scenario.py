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

from stats import compute_logarithmic_mean_stddev
from debris import Debris


class Scenario(object):

    # lookup table mapping (0-7) to wind direction (8: random)
    wind_dir = ['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE']
    terrain_categories = ['2', '2.5', '3', 'non_cyclonic']
    heights = [3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 17.0, 20.0, 25.0, 30.0]

    house_attributes = ['replace_cost', 'height', 'cpe_cov', 'cpe_k',
                        'cpe_str_cov', 'length', 'width', 'roof_cols',
                        'roof_rows']
    zone_attributes = ['area', 'cpi_alpha', 'wall_dir']
    group_attributes = ['dist_order', 'dist_dir', 'damage_scenario',
                        'trigger_collapse_at', 'patch_dist',
                        'set_zone_to_zero', 'water_ingress_order']
    type_attributes = ['costing_area', 'dead_load_mean', 'dead_load_std',
                       'group_name', 'strength_mean', 'strength_std']
    conn_attributes = ['edge', 'type_name', 'zone_loc']

    def __init__(self, cfg_file=None, output_path=None):

        self.cfg_file = cfg_file
        self.path_cfg = os.path.dirname(os.path.realpath(cfg_file))
        self.output_path = output_path

        self.no_sims = None
        self.wind_speed_min = None
        self.wind_speed_max = None
        self.wind_speed_num_steps = None
        self.speeds = None
        self.idx_speeds = None
        self.terrain_category = None
        self.wind_profile = None
        self._debris_regions = None
        self._debris_types = None

        self.house_name = None
        self.path_datafile = None
        self.table_house = None
        self.parallel = None
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
        self.flight_time_mu = None
        self.flight_time_std = None
        self.debris_sources = None

        # house data
        self.df_house = None
        self.df_zones = None
        self.df_zones_cpe_mean = None
        self.df_zones_cpe_eave_mean = None
        self.df_zones_cpe_str_mean = None
        self.df_zones_edge = None

        self.df_groups = None
        self.df_types = None
        self.df_conns = None

        self.df_damage_costing = None

        self.dic_influences = None
        self.dic_damage_factorings = None
        self.dic_patches = None

        self.outfile_model = None
        self.outfile_group = None
        self.outfile_type = None
        self.outfile_conn = None
        self.outfile_zone = None

        self.red_v = 54.0
        self.blue_v = 95.0
        self.flags = dict()

        if not os.path.isfile(cfg_file):
            msg = 'Error: file {} not found'.format(cfg_file)
            sys.exit(msg)
        else:
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

    def read_config(self):

        conf = ConfigParser.ConfigParser()
        conf.optionxform = str
        conf.read(self.cfg_file)

        key = 'main'
        self.no_sims = conf.getint(key, 'no_simulations')
        self.wind_speed_min = conf.getfloat(key, 'wind_speed_min')
        self.wind_speed_max = conf.getfloat(key, 'wind_speed_max')
        self.wind_speed_num_steps = conf.getint(key, 'wind_speed_steps')
        self.terrain_category = conf.get(key, 'terrain_cat')

        try:
            path_wind_profiles = conf.get(key, 'path_wind_profiles')
        except ConfigParser.NoOptionError:
            path_wind_profiles = '../data/gust_envelope_profiles'
        self.set_wind_profile(path_wind_profiles)

        self.speeds = np.linspace(start=self.wind_speed_min,
                                  stop=self.wind_speed_max,
                                  num=self.wind_speed_num_steps)

        self.idx_speeds = range(self.wind_speed_num_steps)
        self.path_datafile = os.path.join(self.path_cfg,
                                          conf.get(key, 'path_datafile'))
        # house data files

        file_house = os.path.join(self.path_datafile, 'house_data.csv')
        file_zones = os.path.join(self.path_datafile, 'zones.csv')
        file_zones_cpe_mean = os.path.join(self.path_datafile,
                                                'zones_cpe_mean.csv')
        file_zones_cpe_str_mean = os.path.join(self.path_datafile,
                                                    'zones_cpe_str_mean.csv')
        file_zones_cpe_eave_mean = os.path.join(self.path_datafile,
                                                     'zones_cpe_eave_mean.csv')
        file_zones_edge = os.path.join(self.path_datafile,
                                            'zones_edge.csv')

        self.df_house = pd.read_csv(file_house)
        self.df_zones = pd.read_csv(file_zones, index_col='name',
                                    dtype={'cpi_alpha': float,
                                           'area': float,
                                           'wall_dir': int})

        names_ = ['name'] + range(8)
        self.df_zones_cpe_mean = pd.read_csv(file_zones_cpe_mean,
                                             names=names_,
                                             index_col='name',
                                             skiprows=1)
        self.df_zones_cpe_str_mean = pd.read_csv(file_zones_cpe_str_mean,
                                                 names=names_,
                                                 index_col='name',
                                                 skiprows=1)
        self.df_zones_cpe_eave_mean = pd.read_csv(file_zones_cpe_eave_mean,
                                                  names=names_,
                                                  index_col='name',
                                                  skiprows=1)
        self.df_zones_edge = pd.read_csv(file_zones_edge,
                                         names=names_,
                                         index_col='name',
                                         skiprows=1)

        file_groups = os.path.join(self.path_datafile, 'conn_groups.csv')
        file_types = os.path.join(self.path_datafile, 'conn_types.csv')
        file_conns = os.path.join(self.path_datafile, 'connections.csv')

        self.df_groups = pd.read_csv(file_groups, index_col='group_name')
        self.df_types = pd.read_csv(file_types, index_col='type_name')

        file_damage_costing = os.path.join(self.path_datafile,
                                           'damage_costing_data.csv')
        try:
            self.df_damage_costing = pd.read_csv(file_damage_costing,
                                             index_col='name')
        except ValueError:
            print('Error in reading {}'.format(file_damage_costing))

        # change arithmetic mean, std to logarithmic mean, std
        self.df_types['lognormal_strength'] = self.df_types.apply(
            lambda row: compute_logarithmic_mean_stddev(row['strength_mean'],
                                                        row['strength_std']),
            axis=1)

        self.df_types['lognormal_dead_load'] = self.df_types.apply(
            lambda row: compute_logarithmic_mean_stddev(row['dead_load_mean'],
                                                        row['dead_load_std']),
            axis=1)

        self.df_conns = pd.read_csv(file_conns, index_col='conn_name')
        self.df_conns['group_name'] = self.df_types.loc[
            self.df_conns['type_name'], 'group_name'].values

        file_influences = os.path.join(self.path_datafile, 'influences.csv')
        self.dic_influences = self.read_influences(file_influences)

        file_damage_factorings = os.path.join(self.path_datafile, 'damage_factorings.csv')
        self.dic_damage_factorings = \
            self.read_damage_factorings(file_damage_factorings)

        file_influence_patches = os.path.join(self.path_datafile, 'influence_patches.csv')
        self.dic_influence_patches = self.read_influences(file_influence_patches)

        self.parallel = conf.getboolean(key, 'parallel')
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

            try:
                self.debris_regions = conf.get(key, 'file_debris_regions')
            except ConfigParser.NoOptionError:
                self.debris_regions = '../data/debris_regions.csv'

            try:
                self.debris_types = conf.get(key, 'file_debris_types')
            except ConfigParser.NoOptionError:
                self.debris_types = '../data/debris_types.csv'

            self.source_items = conf.getint(key, 'source_items')
            self.building_spacing = conf.getfloat(key, 'building_spacing')
            self.debris_radius = conf.getfloat(key, 'debris_radius')
            self.debris_angle = conf.getfloat(key, 'debris_angle')
            self.debris_extension = conf.getfloat(key, 'debris_extension')
            flight_time_mean = conf.getfloat(key, 'flight_time_mean')
            flight_time_stddev = conf.getfloat(key, 'flight_time_stddev')
            self.flight_time_mu, self.flight_time_std = \
                compute_logarithmic_mean_stddev(flight_time_mean,
                                                flight_time_stddev)

            self.debris_sources = Debris.create_sources(self.debris_radius,
                                                        self.debris_angle,
                                                        self.building_spacing,
                                                        self.flags['debris_staggered_sources'])

        key = 'heatmap'
        try:
            self.red_v = conf.getfloat(key, 'red_V')
            self.blue_v = conf.getfloat(key, 'blue_V')
        except ConfigParser.NoSectionError:
            print('default value is used for heatmap')

        if self.output_path:

            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            print 'output directory: {}'.format(self.output_path)

            """
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

            self.file_dead_load_by_conn = os.path.join(self.output_path,
                                                   'dead_load_by_conn.csv')

            self.file_dmg_dist_by_conn = os.path.join(self.output_path,
                                                   'dmg_dist_by_conn.csv')
            """

            self.outfile_model = os.path.join(self.output_path, 'results_model.h5')
            self.outfile_group = os.path.join(self.output_path, 'results_group.h5')
            self.outfile_type = os.path.join(self.output_path, 'results_type.h5')
            self.outfile_conn = os.path.join(self.output_path, 'results_conn.h5')
            self.outfile_zone = os.path.join(self.output_path, 'results_zone.h5')

        else:
            print 'output path is not assigned'

    @staticmethod
    def read_damage_factorings(filename):
        """

        Args:
            filename: influences.csv
            connection_name, zone1_name, zone1_infl, (.....)
        Returns: dictionary

        """
        _dic = dict()
        with open(filename, 'r') as f:
            next(f)  # skip the first line
            for line in f:
                fields = line.strip().rstrip(',').split(',')
                _dic.setdefault(fields[0], []).append(fields[1])

        return _dic

    @staticmethod
    def read_influences(filename):
        """

        Args:
            filename: influences.csv
            connection_name, zone1_name, zone1_infl, (.....)
        Returns: dictionary

        """
        _dic = dict()
        with open(filename, 'r') as f:
            next(f)  # skip the first line
            for line in f:
                key = None
                sub_key = None
                fields = line.strip().rstrip(',').split(',')
                for i, value in enumerate(fields):
                    if i == 0:
                        try:
                            key = int(value)
                        except ValueError:
                            key = value
                    elif i % 2:
                        try:
                            sub_key = int(value)
                        except ValueError:
                            sub_key = value
                    elif key and sub_key:
                        _dic.setdefault(key, {})[sub_key] = float(value)

        return _dic

    @staticmethod
    def read_influence_patches(filename):
        """

        Args:
            filename: influence_patches.csv
            damaged_conn, target_conn, conn_name, conn_infl, (.....)
        Returns: dictionary

        """
        _dic = dict()
        with open(filename, 'r') as f:
            next(f)  # skip the first line
            for line in f:
                damaged_conn = None
                target_conn = None
                sub_key = None
                fields = line.strip().rstrip(',').split(',')
                for i, value in enumerate(fields):
                    if i == 0:
                        damaged_conn = int(value)
                    elif i == 1:
                        target_conn = int(value)
                    elif i % 2 == 0:
                        sub_key = int(value)
                    elif damaged_conn and target_conn and sub_key:
                        _dic.setdefault(damaged_conn, {}).setdefault(target_conn, {})[sub_key] = float(value)

        return _dic

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

    def set_wind_profile(self, path_wind_profiles):

        _path = os.path.join(self.path_cfg, path_wind_profiles)

        try:
            float(self.terrain_category)
        except ValueError:
            _file = 'non_cyclonic.csv'
        else:
            _file = 'cyclonic_terrain_cat{}.csv'.format(self.terrain_category)

        self.wind_profile = pd.read_csv(os.path.join(_path, _file),
                                        skiprows=1,
                                        header=None,
                                        index_col=0).to_dict('list')

    @property
    def debris_regions(self):
        return self._debris_regions

    @debris_regions.setter
    def debris_regions(self, file_debris_regions):

        path_cfg_file = os.path.dirname(os.path.realpath(self.cfg_file))
        file_ = os.path.join(path_cfg_file, file_debris_regions)
        self._debris_regions = pd.read_csv(file_, index_col=0).to_dict('index')

    @property
    def debris_types(self):
        return self._debris_types

    @debris_types.setter
    def debris_types(self, file_debris_types):

        path_cfg_file = os.path.dirname(os.path.realpath(self.cfg_file))
        file_ = os.path.join(path_cfg_file, file_debris_types)

        self._debris_types = pd.read_csv(file_, index_col=0).to_dict('index')

        for key in self._debris_types:

            tmp = self.debris_regions[self.region_name]
            self._debris_types.setdefault(key, {})['ratio'] = tmp['{}_ratio'.format(key)]

            for item in ['frontalarea', 'mass']:
                _mean = tmp['{0}_{1}_mean'.format(key, item)]
                _std = tmp['{0}_{1}_stddev'.format(key, item)]
                mu_lnx, std_lnx = compute_logarithmic_mean_stddev(_mean, _std)
                self._debris_types.setdefault(key, {})['{}_mu'.format(item)] = mu_lnx
                self._debris_types.setdefault(key, {})['{}_std'.format(item)] = std_lnx

    @property
    def wind_dir_index(self):
        return self._wind_dir_index

    @wind_dir_index.setter
    def wind_dir_index(self, wind_dir_str):
        try:
            self._wind_dir_index = Scenario.wind_dir.index(wind_dir_str.upper())
        except ValueError:
            print('8(i.e., RANDOM) is set for wind_dir_index by default')
            self._wind_dir_index = 8

    def storeToCSV(self, cfg_file):

        config = ConfigParser.RawConfigParser()

        key = 'main'
        config.add_section(key)
        config.set(key, 'path_datafile', self.path_datafile)
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
        # FIXME!!! DO NOT KNOW WHAT debris_extension DOES
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


def delete_keys_from_dict(dict_):
    for key in dict_.keys():
        if key.startswith('_'):
            dict_.pop(key)

    for v in dict_.values():
        if isinstance(v, dict):
            delete_keys_from_dict(v)

    return dict_


if __name__ == '__main__':

    import unittest

    path = '/'.join(__file__.split('/')[:-1])

    class MyTestCase(unittest.TestCase):

        @classmethod
        def setUpClass(cls):

            # cls.output_path = './output'
            scenario_filename1 = os.path.abspath(
                os.path.join(path, '../../scenarios/test_scenario1.cfg'))

            cls.cfg = Scenario(cfg_file=scenario_filename1)

            # cls.scenario_filename3 = os.path.abspath(os.path.join(path,
            #                                          '../../scenarios/test.cfg'))

        def test_debris(self):
            # s1.storeToCSV(self.file3)
            self.assertEquals(self.cfg.flags['debris'], False)

        def test_wind_directions(self):
            self.cfg.wind_dir_index = 'Random'
            self.assertEqual(self.cfg.wind_dir_index, 8)

            self.cfg.wind_dir_index = 'SW'
            self.assertEqual(self.cfg.wind_dir_index, 1)

        def test_water_ingress(self):
            self.assertFalse(self.cfg.flags['water_ingress'])
            self.cfg.flags['water_ingress'] = True
            self.assertTrue(self.cfg.flags['water_ingress'])

        def test_ctgenables(self):
            self.assertTrue(
                self.cfg.flags['conn_type_group_{}'.format('sheeting')])
            self.cfg.setOptCTGEnabled('batten', False)
            self.assertFalse(
                self.cfg.flags['conn_type_group_{}'.format('batten')])

            # s.storeToCSV(self.scenario_filename3)
            # s2 = Scenario(cfg_file=self.scenario_filename3)
            # self.assertFalse(s2.flags['conn_type_group_{}'.format('batten')])
            # self.assertTrue(s2.flags['conn_type_group_{}'.format('sheeting')])

        # def test_construction_levels(self):
        #     s1 = Scenario(cfg_file=self.scenario_filename1)
        #     s1.setConstructionLevel('low', 0.33, 0.42, 0.78)

            # s1.storeToCSV(self.scenario_filename3)
            # s = Scenario(cfg_file=self.scenario_filename3)
            # self.assertAlmostEquals(
            #     s.construction_levels['low']['mean_factor'], 0.42)


    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
