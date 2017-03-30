"""
    Configuration module - user input to model run (file or gui)

    Note: regional shielding may be indirectly disabled by setting
    'regional_shielding_factor' to 1.0
    Note: differential shielding may be indirectly disabled by setting
    'building_spacing' to 0
"""
import os
import sys
import logging
import ConfigParser
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
from scipy.stats import norm

from vaws.stats import compute_logarithmic_mean_stddev
from vaws.damage_costing import Costing, WaterIngressCosting


class Config(object):

    # lookup table mapping (0-7) to wind direction (8: random)
    wind_dir = ['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE', 'Random']
    terrain_categories = ['2', '2.5', '3', 'non_cyclonic']
    heights = [3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 17.0, 20.0, 25.0, 30.0]
    region_names = ['Capital_city', 'Tropical_town']
    house_attributes = ['replace_cost', 'height', 'cpe_cov', 'cpe_k',
                        'cpe_str_cov', 'length', 'width', 'roof_cols',
                        'roof_rows']
    # zone_attributes = ['area', 'cpi_alpha', 'wall_dir']
    # group_attributes = ['dist_order', 'dist_dir', 'damage_scenario',
    #                     'trigger_collapse_at', 'patch_dist',
    #                     'set_zone_to_zero', 'water_ingress_order']
    # type_attributes = ['costing_area', 'dead_load_mean', 'dead_load_std',
    #                    'group_name', 'strength_mean', 'strength_std']
    # connection_attributes = ['edge', 'type_name', 'zone_loc']

    # model dependent attributes
    list_house_bucket = ['profile', 'wind_orientation', 'construction_level',
                          'mzcat', 'str_mean_factor', 'str_cov_factor']

    # model and wind dependent attributes
    list_compnents = ['group', 'connection', 'zone']

    list_house_damage_bucket = ['qz', 'Ms', 'cpi', 'cpi_wind_speed', 'collapse',
                                'di', 'di_except_water', 'repair_cost',
                                'water_ingress_cost']

    list_debris_bucket = ['no_items', 'no_touched', 'breached', 'damaged_area']

    list_group_bucket = ['damaged_area']

    list_connection_bucket = ['damaged', 'capacity', 'load', 'strength',
                              'dead_load']

    list_zone_bucket = ['pressure', 'cpe', 'cpe_str', 'cpe_eave']

    dic_obj_for_fitting = {'weibull': 'vulnerability_weibull',
                           'lognorm': 'vulnerability_lognorm'}

    def __init__(self, cfg_file=None):

        self.cfg_file = cfg_file
        self.path_cfg = os.path.dirname(os.path.realpath(cfg_file))
        self.output_path = None

        self.no_sims = None
        self.wind_speed_min = 0.0
        self.wind_speed_max = 0.0
        self.wind_speed_increment = None
        self.speeds = None
        self.terrain_category = None
        self.path_wind_profiles = None
        self.wind_profile = None
        self.debris_types = None
        self.debris_regions = None

        self.path_datafile = None
        self.table_house = None
        self.house_name = None
        self.parallel = None
        self.region_name = None
        self.construction_levels = OrderedDict()
        self.fragility_thresholds = None
        self.water_ingress_given_di = None

        self.source_items = 0
        self.regional_shielding_factor = 1.0
        self.building_spacing = None
        self.wind_dir_index = None
        self.debris_radius = 0.0
        self.debris_angle = 0.0
        self.flight_time_mean = 0.0
        self.flight_time_stddev = 0.0
        self.flight_time_log_mu = None
        self.flight_time_log_std = None
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
        self.df_connections = None
        self.dic_influences = None
        self.dic_influence_patches = None

        self.list_groups = None
        self.list_types = None
        self.list_connections = None
        self.list_zones = None

        self.dic_costings = None
        self.damage_order_by_water_ingress = None
        self.dic_costing_to_group = None
        self.dic_water_ingress_costings = None
        self.dic_damage_factorings = None
        self.df_footprint = None
        self.dic_front_facing_walls = None
        self.df_coverages = None
        self.dic_walls = None

        self.file_house = None
        self.file_group = None
        self.file_type = None
        self.file_connection = None
        self.file_zone = None
        self.file_curve = None

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

    def get_att(self, att_name, default=0):
        try:
            return getattr(self, att_name)
        except AttributeError:
            return default

    def get_flag(self, key, default=0):
        try:
            return self.flags[key]
        except KeyError:
            return default

    def set_flag(self, flag_name, flag_value):
        self.flags[flag_name] = flag_value

    @staticmethod
    def conf_float(conf, key, option, default):
        value = conf.get(key, option)
        if value:
            return float(value)
        else:
            return default

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
        self.house_name = conf.get(key, 'house_name')
        self.parallel = conf.getboolean(key, 'parallel')
        self.no_sims = conf.getint(key, 'no_simulations')

        self.wind_speed_min = conf.getfloat(key, 'wind_speed_min')
        self.wind_speed_max = conf.getfloat(key, 'wind_speed_max')
        self.wind_speed_steps = conf.getint(key, 'wind_speed_steps')
        self.speeds = np.linspace(start=self.wind_speed_min,
                                  stop=self.wind_speed_max,
                                  num=self.wind_speed_steps)
        self.set_wind_dir_index(conf.get(key, 'wind_fixed_dir'))
        self.regional_shielding_factor = conf.getfloat(
            key, 'regional_shielding_factor')

        self.set_terrain_category(conf.get(key, 'terrain_cat'))

        default_path_wind_profiles = os.path.join(
            self.path_cfg, 'input/gust_envelope_profiles')
        try:
            self.path_wind_profiles = os.path.join(
                self.path_cfg, conf.get(key, 'path_wind_profiles'))

            if not os.path.exists(self.path_wind_profiles):
                self.path_wind_profiles = default_path_wind_profiles

        except ConfigParser.NoOptionError:
            self.path_wind_profiles = default_path_wind_profiles

        self.set_wind_profile(self.path_wind_profiles)

        default_path_datafile = os.path.join(self.path_cfg, 'input/house')
        try:
            self.path_datafile = os.path.join(self.path_cfg,
                                              conf.get(key, 'path_datafile'))
            if not os.path.exists(self.path_datafile):
                self.path_datafile = default_path_datafile

        except ConfigParser.NoOptionError:
            self.path_datafile = default_path_datafile

        try:
            self.output_path = os.path.join(self.path_cfg,
                                            conf.get(key, 'output_path'))
        except ConfigParser.NoOptionError:
            self.output_path = os.path.join(self.path_cfg, 'output')

        key = 'options'
        for sub_key, value in conf.items('options'):
            self.flags[sub_key] = conf.getboolean(key, sub_key)

        self.read_house_data()

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

            logging.info('default construction level distribution is used')

        key = 'fragility_thresholds'
        if conf.has_section(key):
            states = [x.strip() for x in conf.get(key, 'states').split(',')]
            thresholds = [float(x) for x in conf.get(key, 'thresholds').split(',')]
        else:
            states = ['slight', 'medium', 'severe', 'complete']
            thresholds = [0.15, 0.45, 0.6, 0.9]
            print('default fragility thresholds is used')

        self.fragility_thresholds = pd.DataFrame(thresholds,
                                                 index=states,
                                                 columns=['threshold'])
        self.fragility_thresholds['color'] = ['b', 'g', 'y', 'r']

        key = 'debris'
        if self.flags[key]:

            from debris import Debris

            self.set_region_name(conf.get(key, 'region_name'))
            self.source_items = conf.getint(key, 'source_items')
            self.building_spacing = conf.getfloat(key, 'building_spacing')
            self.debris_radius = conf.getfloat(key, 'debris_radius')
            self.debris_angle = conf.getfloat(key, 'debris_angle')
            self.flight_time_mean = conf.getfloat(key, 'flight_time_mean')
            self.flight_time_stddev = conf.getfloat(key, 'flight_time_stddev')

            self.flight_time_log_mu, self.flight_time_log_std = \
                compute_logarithmic_mean_stddev(self.flight_time_mean,
                                                self.flight_time_stddev)

            self.debris_sources = Debris.create_sources(self.debris_radius,
                                                        self.debris_angle,
                                                        self.building_spacing,
                                                        self.flags['debris_staggered_sources'])

            default_path_debris = os.path.join(self.path_cfg, 'input/debris')
            try:
                path_debris = os.path.join(self.path_cfg,
                                           conf.get(key, 'path_debris'))
                if not os.path.exists(path_debris):
                    path_debris = default_path_debris

            except ConfigParser.NoOptionError:
                path_debris = default_path_debris

            file_debris_regions = os.path.join(path_debris,
                                               'debris_regions.csv')
            file_debris_types = os.path.join(path_debris,
                                             'debris_types.csv')
            self.set_debris_types(file_debris_types, file_debris_regions)

        key = 'heatmap'
        try:
            self.red_v = conf.getfloat(key, 'red_V')
            self.blue_v = conf.getfloat(key, 'blue_V')
        except ConfigParser.NoSectionError:
            print('default value is used for heatmap')

        key = 'water_ingress'
        if conf.has_section(key):
            thresholds = [float(x) for x in conf.get(key, 'thresholds').split(',')]
            lower = [float(x) for x in conf.get(key, 'lower').split(',')]
            upper = [float(x) for x in conf.get(key, 'upper').split(',')]
        else:
            thresholds = [0.1, 0.2, 0.5, 2.0]
            lower = [40.0, 35.0, 0.0, -20.0]
            upper = [60.0, 55.0, 40.0, 20.0]
            print('default water ingress thresholds is used')

        self.water_ingress_given_di = pd.DataFrame(np.array([lower, upper]).T,
                                                   index=thresholds,
                                                   columns=['lower', 'upper'])

        self.water_ingress_given_di['wi'] = self.water_ingress_given_di.apply(
            self.return_norm_cdf, axis=1)

        if self.output_path:

            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
            print 'output directory: {}'.format(self.output_path)

            self.file_house = os.path.join(self.output_path, 'results_house.h5')
            self.file_group = os.path.join(self.output_path, 'results_group.h5')
            self.file_type = os.path.join(self.output_path, 'results_type.h5')
            self.file_connection = os.path.join(self.output_path, 'results_connection.h5')
            self.file_zone = os.path.join(self.output_path, 'results_zone.h5')
            self.file_curve = os.path.join(self.output_path, 'results_curve.csv')
        else:
            print 'output path is not assigned'

    @staticmethod
    def return_norm_cdf(row):
        """

        Args:
            row:

        Returns:

        """

        _mean = (row['upper'] + row['lower']) / 2.0
        _sd = (row['upper'] - row['lower']) / 6.0

        return norm(loc=_mean, scale=_sd).cdf

    def read_house_data(self):

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

        file_groups = os.path.join(self.path_datafile, 'conn_groups.csv')
        file_types = os.path.join(self.path_datafile, 'conn_types.csv')
        file_connections = os.path.join(self.path_datafile, 'connections.csv')
        file_damage_costing = os.path.join(self.path_datafile,
                                           'damage_costing_data.csv')

        file_influences = os.path.join(self.path_datafile, 'influences.csv')
        file_influence_patches = os.path.join(self.path_datafile,
                                              'influence_patches.csv')
        file_damage_factorings = os.path.join(self.path_datafile,
                                              'damage_factorings.csv')

        file_water_ingress_costing = os.path.join(
            self.path_datafile, 'water_ingress_costing_data.csv')

        self.df_house = pd.read_csv(file_house)
        self.df_zones = pd.read_csv(file_zones, index_col='name',
                                    dtype={'cpi_alpha': float,
                                           'area': float,
                                           'wall_dir': int})
        self.list_zones = self.df_zones.index.tolist()

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

        self.df_groups = pd.read_csv(file_groups, index_col='group_name')
        self.list_groups = self.df_groups.index.tolist()

        self.df_types = pd.read_csv(file_types, index_col='type_name')
        self.list_types = self.df_types.index.tolist()

        # change arithmetic mean, std to logarithmic mean, std
        self.df_types['lognormal_strength'] = self.df_types.apply(
            lambda row: compute_logarithmic_mean_stddev(row['strength_mean'],
                                                        row['strength_std']),
            axis=1)

        self.df_types['lognormal_dead_load'] = self.df_types.apply(
            lambda row: compute_logarithmic_mean_stddev(row['dead_load_mean'],
                                                        row['dead_load_std']),
            axis=1)

        self.df_connections = pd.read_csv(file_connections,
                                          index_col='conn_name')
        if 'section' not in self.df_connections:
            self.df_connections['section'] = 0

        self.list_connections = self.df_connections.index.tolist()

        self.df_connections['group_name'] = self.df_types.loc[
            self.df_connections['type_name'], 'group_name'].values

        self.dic_influences = self.read_influences(file_influences)
        self.dic_damage_factorings = self.read_damage_factorings(
            file_damage_factorings)
        self.dic_influence_patches = self.read_influence_patches(
            file_influence_patches)

        self.dic_costings, self.dic_costing_to_group, self.damage_order_by_water_ingress = \
            self.read_damage_costing_data(file_damage_costing, self.df_groups)

        self.dic_water_ingress_costings = self.read_water_ingress_costing_data(
            file_water_ingress_costing, self.df_groups)

        # if self.flags['debris']:
        file_footprint = os.path.join(self.path_datafile, 'footprint.csv')
        file_coverages = os.path.join(self.path_datafile, 'coverages.csv')
        file_coverage_types = os.path.join(self.path_datafile,
                                           'coverage_types.csv')
        file_wall = os.path.join(self.path_datafile, 'walls.csv')
        file_front_facing_walls = os.path.join(self.path_datafile,
                                               'front_facing_walls.csv')

        self.df_footprint = pd.read_csv(file_footprint,
                                        skiprows=1,
                                        header=None)

        self.dic_front_facing_walls = self.read_front_facing_walls(
            file_front_facing_walls)

        dic_coverage_types = pd.read_csv(
            file_coverage_types, index_col='Name').to_dict('index')

        try:
            self.dic_walls = pd.read_csv(
                file_wall, index_col='wall_name').to_dict()['wall_area']
        except TypeError:
            self.dic_walls = dict()

        self.df_coverages = pd.read_csv(file_coverages)

        if dic_coverage_types:
            self.df_coverages['log_failure_momentum'] = \
                self.df_coverages.apply(self.get_lognormal_tuple,
                                        args=(dic_coverage_types,), axis=1)

    @staticmethod
    def read_damage_costing_data(file_damage_costing, df_groups):
        dic_costing = {}
        df_damage_costing = pd.read_csv(file_damage_costing)
        for _, item in df_damage_costing.iterrows():
            if df_groups['damage_scenario'].isin([item['name']]).any():
                _name = item['name']
                dic_costing[_name] = Costing(costing_name=_name, **item)

        dic_costing_to_group = defaultdict(list)
        for key, value in df_groups['damage_scenario'].to_dict().iteritems():
            dic_costing_to_group[value].append(key)

        _a = df_damage_costing.loc[df_damage_costing[
            'water_ingress_order'].sort_values().index, 'name']
        damage_order_by_water_ingress = []
        for i, value in _a.iteritems():
            if df_damage_costing.loc[i, 'water_ingress_order'] and \
                    df_groups['damage_scenario'].isin([value]).any():
                damage_order_by_water_ingress.append(value)

        return dic_costing, dic_costing_to_group, damage_order_by_water_ingress

    @staticmethod
    def read_water_ingress_costing_data(file_water_ingress_costing, df_groups):
        dic_ = {}
        tmp = pd.read_csv(file_water_ingress_costing)
        for key, grouped in tmp.groupby('name'):
            if df_groups['damage_scenario'].isin([key]).any():
                grouped = grouped.set_index('water_ingress')
                grouped['costing'] = grouped.apply(
                    lambda row: WaterIngressCosting(costing_name=key, **row),
                    axis=1)
                dic_[key] = grouped
        return dic_

    @staticmethod
    def read_front_facing_walls(filename):
        _dic = dict()
        with open(filename, 'r') as f:
            next(f)  # skip the first line
            for line in f:
                fields = line.strip().rstrip(',').split(',')
                _dic[fields[0]] = [int(x) for x in fields[1:]]
        return _dic

    @staticmethod
    def get_lognormal_tuple(row, dic_):
        _type = row['coverage_type']
        _mean = dic_[_type]['failure_momentum_mean']
        _sd = dic_[_type]['failure_momentum_std']
        return compute_logarithmic_mean_stddev(_mean, _sd)

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
                        try:
                            damaged_conn = int(value)
                        except ValueError:
                            damaged_conn = value
                    elif i == 1:
                        try:
                            target_conn = int(value)
                        except ValueError:
                            target_conn = value
                    elif i % 2 == 0:
                        try:
                            sub_key = int(value)
                        except ValueError:
                            sub_key = value
                    elif damaged_conn and target_conn and sub_key:
                        _dic.setdefault(damaged_conn, {}
                                        ).setdefault(target_conn, {}
                                                     )[sub_key] = float(value)

        return _dic

    def set_region_name(self, value):
        try:
            assert value in self.region_names
        except AssertionError:
            self.region_name = 'Capital_city'
            print('Capital_city is set for region_name by default')
        else:
            self.region_name = value

    def set_terrain_category(self, value):
        try:
            assert value in self.terrain_categories
        except AssertionError:
            print('Invalid terrain category: {}'.format(value))
        else:
            self.terrain_category = value

    def set_wind_profile(self, path_wind_profiles):

        try:
            float(self.terrain_category)
        except ValueError:
            _file = 'non_cyclonic.csv'
        else:
            _file = 'cyclonic_terrain_cat{}.csv'.format(self.terrain_category)

        self.wind_profile = pd.read_csv(os.path.join(path_wind_profiles, _file),
                                        skiprows=1,
                                        header=None,
                                        index_col=0).to_dict('list')

    def set_debris_types(self, file_debris_types, file_debris_regions):

        self.debris_types = pd.read_csv(
            file_debris_types, index_col=0).to_dict('index')
        self.debris_regions = pd.read_csv(
            file_debris_regions, index_col=0).to_dict('index')[self.region_name]

        for key in self.debris_types:

            self.debris_types.setdefault(key, {})['ratio'] = \
                self.debris_regions['{}_ratio'.format(key)]

            for item in ['frontalarea', 'mass']:
                _mean = self.debris_regions['{0}_{1}_mean'.format(key, item)]
                _std = self.debris_regions['{0}_{1}_stddev'.format(key, item)]
                mu_lnx, std_lnx = compute_logarithmic_mean_stddev(_mean, _std)
                self.debris_types.setdefault(key, {})['{}_mu'.format(item)] = mu_lnx
                self.debris_types.setdefault(key, {})['{}_std'.format(item)] = std_lnx

    def set_wind_dir_index(self, wind_dir_str):
        try:
            self.wind_dir_index = Config.wind_dir.index(wind_dir_str.upper())
        except ValueError:
            print('8(i.e., RANDOM) is set for wind_dir_index by default')
            self.wind_dir_index = 8

    def save_config(self, ):

        config = ConfigParser.RawConfigParser()

        key = 'main'
        config.add_section(key)
        config.set(key, 'path_datafile', self.path_datafile)
        config.set(key, 'parallel', self.parallel)
        config.set(key, 'no_simulations', self.no_sims)
        config.set(key, 'wind_speed_min', self.wind_speed_min)
        config.set(key, 'wind_speed_max', self.wind_speed_max)
        config.set(key, 'wind_speed_steps', self.wind_speed_steps)
        config.set(key, 'terrain_cat', self.terrain_category)
        config.set(key, 'house_name', self.house_name)
        config.set(key, 'regional_shielding_factor',
                   self.regional_shielding_factor)
        config.set(key, 'wind_fixed_dir', Config.wind_dir[self.wind_dir_index])
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

        with open(self.cfg_file, 'wb') as configfile:
            config.write(configfile)


def delete_keys_from_dict(dict_):
    for key in dict_.keys():
        if key.startswith('_'):
            dict_.pop(key)

    for v in dict_.values():
        if isinstance(v, dict):
            delete_keys_from_dict(v)

    return dict_

