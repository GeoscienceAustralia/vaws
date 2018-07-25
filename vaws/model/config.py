"""Config module

    This module defines basic parameter values for simulation.

    Attributes:

        OUTPUT_DIR: "output"
        INPUT_DIR: "input"
        DEBRIS_DATA: "input/debris"
        GUST_PROFILES_DATA: "input/gust_envelope_profiles"
        HOUSE_DATA: "input/house"

        FILE_HOUSE_DATA: 'house_data.csv'
        FILE_CONN_GROUPS: 'conn_groups.csv'
        FILE_CONN_TYPES: 'conn_types.csv'
        FILE_CONNECTIONS: 'connections.csv'
        FILE_ZONES: 'zones.csv'

        FILE_COVERAGE_TYPES: 'coverage_types.csv'
        FILE_COVERAGES: 'coverages.csv'
        FILE_COVERAGES_CPE: 'coverages_cpe.csv'

        FILE_INFLUENCES: 'influences.csv'
        FILE_INFLUENCE_PATCHES: 'influence_patches.csv'

        FILE_DAMAGE_COSTING_DATA: 'damage_costing_data.csv'
        FILE_DAMAGE_FACTORINGS: 'damage_factorings.csv'
        FILE_WATER_INGRESS_COSTING_DATA: 'water_ingress_costing_data.csv'

        FILE_FOOTPRINT: 'footprint.csv'
        FILE_FRONT_FACING_WALLS: 'front_facing_walls.csv'

        # debris data
        FILE_DEBRIS: 'debris.csv'

        # results
        FILE_RESULTS: 'results.h5'


    Note:
        * regional shielding may be indirectly disabled by setting 'regional_shielding_factor' to 1.0
        * differential shielding may be indirectly disabled by setting 'building_spacing' to 0
"""

from __future__ import division, print_function

import os
import sys
import logging
import ConfigParser
import pandas as pd

from collections import OrderedDict, defaultdict
import numpy as np
from scipy import stats
from matplotlib import patches
from shapely import geometry

from vaws.model.constants import (WIND_DIR, FLAGS_PRESSURE, FLAGS_DIST_DIR,
                                  DEBRIS_TYPES_KEYS, DEBRIS_TYPES_ATTS,
                                  COVERAGE_FAILURE_KEYS, COSTING_FORMULA_TYPES,
                                  BLDG_SPACING)
from vaws.model.stats import compute_logarithmic_mean_stddev, calc_big_a_b_values
from vaws.model.damage_costing import Costing, WaterIngressCosting
from vaws.model.zone import get_grid_from_zone_location

OUTPUT_DIR = "output"
INPUT_DIR = "input"
DEBRIS_DATA = os.path.join(INPUT_DIR, "debris")
GUST_PROFILES_DATA = os.path.join(INPUT_DIR, "gust_envelope_profiles")
HOUSE_DATA = os.path.join(INPUT_DIR, "house")

# house data files
FILE_HOUSE_DATA = 'house_data.csv'
FILE_CONN_GROUPS = 'conn_groups.csv'
FILE_CONN_TYPES = 'conn_types.csv'
FILE_CONNECTIONS = 'connections.csv'
FILE_ZONES = 'zones.csv'
FILE_COVERAGE_TYPES = 'coverage_types.csv'
FILE_COVERAGES = 'coverages.csv'
FILE_COVERAGES_CPE = 'coverages_cpe.csv'
FILE_INFLUENCES = 'influences.csv'
FILE_INFLUENCE_PATCHES = 'influence_patches.csv'
FILE_DAMAGE_COSTING_DATA = 'damage_costing_data.csv'
FILE_DAMAGE_FACTORINGS = 'damage_factorings.csv'
FILE_WATER_INGRESS_COSTING_DATA = 'water_ingress_costing_data.csv'
FILE_FOOTPRINT = 'footprint.csv'
FILE_FRONT_FACING_WALLS = 'front_facing_walls.csv'
FILE_DEBRIS = 'debris.csv'
FILE_RESULTS = 'results.h5'


class Config(object):
    """ Config class to set configuration for simulation

    Attributes:

        # model dependent attributes
        house_bucket = (list):

        att_non_float = (list):

        house_bucket = (list):

        debris_bucket = (list):

        # model and wind dependent attributes
        list_components = (list):

        group_bucket = (list):

        connection_bucket (list):

        zone_bucket (list):

        coverage_bucket (list):

        dic_obj_for_fitting (dict):

        att_time_invariant (list):


    """
    # lookup table mapping (0-7) to wind direction (8: random)
    header_for_cpe = ['name', 0, 1, 2, 3, 4, 5, 6, 7]

    # model dependent attributes
    # time variant: 1
    house_bucket = [('profile_index', 0),
                    ('wind_dir_index', 0),
                    ('construction_level', 0),
                    ('terrain_height_multiplier', 0),
                    ('shielding_multiplier', 0),
                    ('mean_factor', 0),
                    ('cv_factor', 0),
                    ('qz', 1),
                    ('cpi', 1),
                    ('collapse', 0),
                    ('di', 1),
                    ('di_except_water', 1),
                    ('repair_cost', 1),
                    ('water_ingress_cost', 1),
                    ('window_breached', 1),
                    ('no_debris_items', 1),
                    ('no_debris_impacts', 1),
                    ('breached_area', 1),
                    ('mean_no_debris_items', 1)]

    att_non_float = ['construction_level']

    # model and wind dependent attributes
    list_components = ['group', 'connection', 'zone', 'coverage']

    group_bucket = [('damaged_area', 1), ('prop_damaged', 1)]

    connection_bucket = [('damaged', 1),
                         ('capacity', 0),
                         ('load', 1),
                         ('strength', 0),
                         ('dead_load', 0)]

    zone_bucket = [('pressure_cpe', 1),
                   ('pressure_cpe_str', 1),
                   ('cpe', 0),
                   ('cpe_str', 0),
                   ('cpe_eave', 0),
                   ('differential_shielding', 0)]

    coverage_bucket = [('strength_negative', 0),
                       ('strength_positive', 0),
                       ('momentum_capacity', 0),
                       ('capacity', 0),
                       ('load', 1),
                       ('breached', 1),
                       ('breached_area', 1)]

    # debris_bucket = [('mass', 1),
    #                  ('frontal_area', 0),
    #                  ('flight_time', 0),
    #                  ('momentum', 0),
    #                  ('flight_distance', 1),
    #                  ('impact', 1),
    #                  ('landing.x', 1),
    #                  ('landing.y', 1)]

    dic_obj_for_fitting = {'weibull': 'vulnerability_weibull',
                           'lognorm': 'vulnerability_lognorm'}

    def __init__(self, file_cfg=None, logger=None):
        """ Initialise instance of Config class

        Args:
            file_cfg:
        """
        self.file_cfg = file_cfg
        self.logger = logger or logging.getLogger(__name__)

        self.model_name = None  # only used for gui display may be deleted later
        self.no_models = None
        self.random_seed = 0
        self.wind_direction = None
        self.wind_speed_min = 0.0
        self.wind_speed_max = 0.0
        self.wind_speed_increment = 0.0
        self.wind_speed_steps = None  # set_wind_speeds
        self.wind_speeds = None  # set_wind_speeds
        self.regional_shielding_factor = 1.0
        self.file_wind_profiles = None
        self.wind_profiles = None
        self.profile_heights = None
        self.wind_dir_index = None

        self.construction_levels = {}
        self.construction_levels_levels = ['medium']
        self.construction_levels_probs = [1.0]
        self.construction_levels_mean_factors = [1.0]
        self.construction_levels_cv_factors = [0.58]

        self.fragility = None
        self.fragility_i_states = ['slight', 'medium', 'severe', 'complete']
        self.fragility_i_thresholds = [0.15, 0.45, 0.6, 0.9]

        self.water_ingress = None
        self.water_ingress_i_thresholds = [0.1, 0.2, 0.5]
        self.water_ingress_i_speed_at_zero_wi = [50.0, 35.0, 0.0, -20.0]
        self.water_ingress_i_speed_at_full_wi = [75.0, 55.0, 40.0, 20.0]

        # debris related
        self.region_name = None
        self.staggered_sources = None
        self.source_items = 0
        self.boundary_radius = 0.0
        self.impact_boundary = None
        self.building_spacing = None
        self.debris_radius = 0.0
        self.debris_angle = 0.0
        self.debris_sources = None
        self.debris_regions = None
        self.debris_types = {}
        self.debris_types_ratio = []
        self.footprint = None

        # house data
        self.house = None
        self.zones = None
        self.coverages = None

        self.groups = None
        self.types = None
        self.connections = None
        self.damage_grid_by_sub_group = None
        self.influences = None
        self.influence_patches = None

        self.list_groups = None
        self.list_subgroups = None
        self.list_connections = None
        self.list_zones = None
        self.list_coverages = None

        # damage costing
        self.costings = None
        self.damage_order_by_water_ingress = None
        self.costing_to_group = None
        self.water_ingress_costings = None
        self.damage_factorings = None

        # debris related
        self.front_facing_walls = None
        self.coverages_area = 0.0

        self.file_results = None

        self.heatmap_vmin = 54.0
        self.heatmap_vmax = 95.0
        self.heatmap_vstep = 21

        self.flags = {}

        if not os.path.isfile(file_cfg):
            msg = 'Error: {} not found'.format(file_cfg)
            sys.exit(msg)
        else:
            self.path_cfg = os.sep.join(os.path.abspath(file_cfg).split(os.sep)[:-1])
            self.path_output = os.path.join(self.path_cfg, OUTPUT_DIR)
            self.path_house_data = os.path.join(self.path_cfg, HOUSE_DATA)
            self.path_wind_profiles = os.path.join(self.path_cfg,
                                                   GUST_PROFILES_DATA)
            self.path_debris = os.path.join(self.path_cfg, DEBRIS_DATA)

            self.file_house_data = os.path.join(
                self.path_house_data, FILE_HOUSE_DATA)
            self.file_conn_groups = os.path.join(
                self.path_house_data, FILE_CONN_GROUPS)
            self.file_conn_types = os.path.join(
                self.path_house_data, FILE_CONN_TYPES)
            self.file_connections = os.path.join(
                self.path_house_data, FILE_CONNECTIONS)
            self.file_zones = os.path.join(self.path_house_data, FILE_ZONES)
            for item in ['cpe_mean', 'cpe_str_mean', 'cpe_eave_mean', 'edge']:
                setattr(self, 'file_zones_{}'.format(item),
                        os.path.join(self.path_house_data,
                                     'zones_{}.csv'.format(item)))

            self.file_influences = os.path.join(
                self.path_house_data, FILE_INFLUENCES)
            self.file_influence_patches = os.path.join(
                self.path_house_data, FILE_INFLUENCE_PATCHES)

            self.file_coverage_types = os.path.join(
                self.path_house_data, FILE_COVERAGE_TYPES)
            self.file_coverages = os.path.join(
                self.path_house_data, FILE_COVERAGES)
            self.file_coverages_cpe = os.path.join(
                self.path_house_data, FILE_COVERAGES_CPE)
            self.file_damage_costing_data = os.path.join(
                self.path_house_data, FILE_DAMAGE_COSTING_DATA)
            self.file_damage_factorings = os.path.join(
                self.path_house_data, FILE_DAMAGE_FACTORINGS)
            self.file_water_ingress_costing_data = os.path.join(
                self.path_house_data, FILE_WATER_INGRESS_COSTING_DATA)

            self.file_front_facing_walls = os.path.join(
                self.path_house_data, FILE_FRONT_FACING_WALLS)
            self.file_footprint = os.path.join(
                self.path_house_data, FILE_FOOTPRINT)

            self.read_config()
            self.process_config()
            self.set_output_files()

    def set_output_files(self):
        if not os.path.exists(self.path_output):
            os.makedirs(self.path_output)
        self.file_results = os.path.join(self.path_output, FILE_RESULTS)

    def read_config(self):

        conf = ConfigParser.ConfigParser()
        conf.optionxform = str
        conf.read(self.file_cfg)

        self.read_main(conf, key='main')
        self.read_options(conf, key='options')
        self.read_heatmap(conf, key='heatmap')
        self.read_construction_levels(conf, key='construction_levels')
        self.read_fragility_thresholds(conf, key='fragility_thresholds')
        self.read_debris(conf, key='debris')
        self.read_water_ingress(conf, key='water_ingress')

    def process_config(self):

        self.set_wind_speeds()  # speeds, wind_speed_steps
        self.set_wind_dir_index()  # wind_dir_index
        self.set_wind_profiles()  # profile_heights, wind_profiles

        self.set_house()
        df_groups = self.set_groups()
        df_types = self.set_types()
        self.set_connections(df_types, df_groups)
        self.set_zones()
        self.set_coverages()

        self.set_influences()
        self.set_influence_patches()
        self.set_costings(df_groups)

        if self.flags['debris']:

            from vaws.model.debris import create_sources

            self.debris_sources = create_sources(
                self.debris_radius,
                self.debris_angle,
                self.building_spacing,
                self.staggered_sources)

        self.set_debris_types()

        self.set_wawter_ingress()
        self.set_fragility_thresholds()
        self.set_construction_levels()

    def read_options(self, conf, key):
        for sub_key, value in conf.items(key):
            self.flags[sub_key] = conf.getboolean(key, sub_key)

    def read_main(self, conf, key):
        """
        
        Args:
            conf: 
            key: 

        Returns:

        """
        self.model_name = conf.get(key, 'model_name')
        self.no_models = conf.getint(key, 'no_models')
        try:
            self.random_seed = conf.getint(key, 'random_seed')
        except ConfigParser.NoOptionError:
            self.random_seed = 0
        self.wind_speed_min = conf.getfloat(key, 'wind_speed_min')
        self.wind_speed_max = conf.getfloat(key, 'wind_speed_max')
        self.wind_speed_increment = conf.getfloat(key, 'wind_speed_increment')
        # self.set_wind_speeds()
        self.wind_direction = conf.get(key, 'wind_direction')
        self.regional_shielding_factor = conf.getfloat(
            key, 'regional_shielding_factor')
        self.file_wind_profiles = conf.get(key, 'wind_profiles')

    def set_wind_speeds(self):
        self.wind_speed_steps = int(
            (self.wind_speed_max - self.wind_speed_min) /
            self.wind_speed_increment) + 1
        self.wind_speeds = np.linspace(start=self.wind_speed_min,
                                       stop=self.wind_speed_max,
                                       num=self.wind_speed_steps,
                                       endpoint=True)

    def read_water_ingress(self, conf, key):
        """
        read water ingress related parameters
        Args:
            conf:
            key:

        Returns:

        TODO:
        """
        if conf.has_section(key):
            for k in ['thresholds', 'speed_at_zero_wi', 'speed_at_full_wi']:
                setattr(self, 'water_ingress_i_{}'.format(k),
                        self.read_column_separated_entry(conf.get(key, k)))
        else:
            self.logger.info('default water ingress thresholds is used')

    def set_wawter_ingress(self):
        thresholds = [x for x in self.water_ingress_i_thresholds]
        thresholds.append(1.1)
        self.water_ingress = pd.DataFrame(
            np.array([self.water_ingress_i_speed_at_zero_wi,
                      self.water_ingress_i_speed_at_full_wi]).T,
            columns=['speed_at_zero_wi', 'speed_at_full_wi'],
            index=thresholds)
        self.water_ingress['wi'] = self.water_ingress.apply(
            self.return_norm_cdf, axis=1)

    def set_coverages(self):

        self.set_front_facing_walls()

        try:
            coverage_types = pd.read_csv(
                self.file_coverage_types, index_col=0)
        except IOError as msg:
            self.logger.error(msg, exc_info=True)
        else:
            # Check that each failure_strength_out_mean entry is <=0.
            not_good = coverage_types.loc[
                coverage_types['failure_strength_out_mean'] > 0].index.tolist()
            if not_good:
                raise ValueError(
                    'Invalid failure_strength_out_mean for coverage(s): {}'.format(not_good))

            # Check that the remaining numerical entries are >=0.
            sub_df = coverage_types.loc[:, coverage_types.columns != 'failure_strength_out_mean']
            not_good = sub_df.loc[(sub_df < 0).any(axis=1)].index.tolist()
            if not_good:
                raise ValueError('Invalid value(s) for coverage(s): {}'.format(
                    not_good))

            coverage_types = coverage_types.to_dict('index')

        try:
            self.coverages = pd.read_csv(self.file_coverages, index_col=0)
        except IOError as msg:
            self.logger.error(msg, exc_info=True)
        else:

            if not self.coverages.empty:

                walls = set([item for sublist in self.front_facing_walls.values()
                             for item in sublist])

                # check coverage_type
                not_good = self.coverages.loc[
                    ~self.coverages['wall_name'].isin(walls)].index.tolist()

                if not_good:
                    raise ValueError(
                        'Invalid wall name for coverages: {}'.format(not_good))

                # check area >= 0
                not_good = self.coverages.loc[
                    self.coverages['area'] < 0.0].index.tolist()
                if not_good:
                    raise ValueError(
                        'Invalid area for coverages: {}'.format(not_good))

                # check coverage_type
                not_good = self.coverages.loc[
                    ~self.coverages['coverage_type'].isin(
                        coverage_types.keys())].index.tolist()

                if not_good:
                    raise ValueError(
                        'Invalid coverage_type for coverages: {}'.format(not_good))

                for key in COVERAGE_FAILURE_KEYS:
                    df = self.coverages.apply(self.get_lognormal_tuple,
                                              args=(coverage_types, key,),
                                              axis=1)
                    self.coverages = self.coverages.merge(df,
                                                          left_index=True,
                                                          right_index=True)
                self.list_coverages = self.coverages.index.tolist()

                self.coverages_area = self.coverages['area'].sum()
            else:
                self.coverages = None

        self.set_coverages_cpe()

    def set_front_facing_walls(self):

        try:
            self.front_facing_walls = self.read_front_facing_walls(
                self.file_front_facing_walls)
        except IOError as msg:
            self.logger.error(msg, exc_info=True)
        # else:
        #     try:
        #         _set = set(self.coverages.wall_name.unique())
        #     except AttributeError:
        #         pass
        #     else:
        #         msg = 'Invalid wall name for {}'
        #         for key, value in self.front_facing_walls.items():
        #             assert set(value).issubset(_set), msg.format(key)

    def set_coverages_cpe(self):
        try:
            coverages_cpe_mean = pd.read_csv(
                self.file_coverages_cpe, names=self.header_for_cpe, index_col=0,
                skiprows=1)
        except (IOError, TypeError) as msg:
            self.logger.warning(msg)
        else:
            if not coverages_cpe_mean.empty:
                msg = 'Invalid value(s) for coverage(s): {} in {}'
                not_good = coverages_cpe_mean.loc[
                    ((coverages_cpe_mean > 5) | (coverages_cpe_mean < -5)).any(
                        axis=1)].index.tolist()
                if not_good:
                    self.logger.warning(msg.format(not_good, self.file_coverages_cpe))
                else:
                    self.coverages['cpe_mean'] = pd.Series(
                        coverages_cpe_mean.to_dict('index'))

    def read_debris(self, conf, key):

        # global data
        _file = os.path.join(self.path_debris, FILE_DEBRIS)
        try:
            self.debris_regions = pd.read_csv(_file, index_col=0).to_dict()
        except IOError as msg:
            self.logger.error(msg, exc_info=True)

        self.staggered_sources = conf.getboolean(key, 'staggered_sources')
        self.source_items = conf.getint(key, 'source_items')
        self.boundary_radius = conf.getfloat(key, 'boundary_radius')
        self.impact_boundary = geometry.Point(0, 0).buffer(self.boundary_radius)
        for item in ['building_spacing', 'debris_radius', 'debris_angle']:
            setattr(self, item, conf.getfloat(key, item))

        try:
            assert self.building_spacing in BLDG_SPACING
        except AssertionError:
            self.logger.error('building_spacing should be either 20 or 40')

        self.set_region_name(conf.get(key, 'region_name'))

        self.set_footprint()

    def set_footprint(self):
        try:
            footprint_xy = pd.read_csv(self.file_footprint, skiprows=1,
                                       header=None).values
        except IOError as msg:
            self.logger.warning(msg)
        else:
            if np.isnan(footprint_xy).any():
                raise ValueError('Invalid coordinates for footprint')
            self.footprint = geometry.Polygon(footprint_xy)

    def read_fragility_thresholds(self, conf, key):
        if conf.has_section(key):
            for k in ['states', 'thresholds']:
                setattr(self, 'fragility_i_{}'.format(k),
                        self.read_column_separated_entry(conf.get(key, k)))
        else:
            self.logger.warning('default fragility thresholds is used')

    def set_fragility_thresholds(self):
        self.fragility = pd.DataFrame(self.fragility_i_thresholds,
                                      index=self.fragility_i_states,
                                      columns=['threshold'])
        self.fragility['color'] = ['b', 'g', 'y', 'r']

    def read_construction_levels(self, conf, key):
        try:
            for k in ['levels', 'probs', 'mean_factors', 'cv_factors']:
                setattr(self, 'construction_levels_{}'.format(k),
                        self.read_column_separated_entry(conf.get(key, k)))
        except ConfigParser.NoSectionError:
            self.logger.warning('construction level medium is used')

    def set_construction_levels(self):
        for level, mean, cov in zip(
                self.construction_levels_levels,
                self.construction_levels_mean_factors,
                self.construction_levels_cv_factors):
            self.construction_levels[level] = {'mean_factor': mean,
                                               'cv_factor': cov}

    @staticmethod
    def read_column_separated_entry(value):
        try:
            return [float(x) for x in value.split(',')]
        except ValueError:
            return [x.strip() for x in value.split(',')]

    def read_heatmap(self, conf, key):
        try:
            for item in ['vmin', 'vmax', 'vstep']:
                setattr(self, 'heatmap_{}'.format(item), conf.getfloat(key, item))
        except ConfigParser.NoSectionError:
            self.logger.warning('default value is used for heatmap')

    @staticmethod
    def return_norm_cdf(row):
        """

        Args:
            row:

        Returns:

        """

        mean = (row['speed_at_zero_wi'] + row['speed_at_full_wi']) / 2.0
        sd = (row['speed_at_full_wi'] - row['speed_at_zero_wi']) / 6.0

        return stats.norm(loc=mean, scale=sd).cdf

    def set_house(self):

        # house data
        try:
            df = pd.read_csv(self.file_house_data, index_col=0, header=None)[1]
        except IOError as msg:
            self.logger.error(msg, exc_info=True)
        else:
            self.house = {}
            for key, value in df.iteritems():
                try:
                    self.house[key] = float(value)
                except ValueError:
                    self.house[key] = value

            assert 0 <= self.house['cpe_cv'] <= 1, \
                "cpe_cv should be between 0 and 1"
            assert 0 <= self.house['cpe_str_cv'] <= 1, \
                "cpe_str_cv should be between 0 and 1"
            assert self.house['cpe_k'] > 0, "cpe_k should be > 0"
            assert self.house['cpe_str_k'] > 0, "cpe_str_k should be > 0"

            self.house['big_a'], self.house['big_b'] = \
                calc_big_a_b_values(shape_k=self.house['cpe_k'])

            self.house['big_a_str'], self.house['big_b_str'] = \
                calc_big_a_b_values(shape_k=self.house['cpe_str_k'])

    def set_groups(self):
        try:
            df_groups = pd.read_csv(
                self.file_conn_groups, index_col=0).fillna('')
        except IOError as msg:
            self.logger.error(msg, exc_info=True)
        else:
            self.groups = df_groups.to_dict('index')
            df_groups.sort_values(by='dist_order', inplace=True)

            if (~df_groups['flag_pressure'].isin(FLAGS_PRESSURE)).sum():
                msg = 'flag_pressure should be either {} or {}'.format(
                    *FLAGS_PRESSURE)
                raise Exception(msg)

            if (~df_groups['dist_dir'].isin(FLAGS_DIST_DIR)).sum():
                msg = 'dist_dir should be either {}, {}, {} or {}'.format(
                    *FLAGS_DIST_DIR)
                raise Exception(msg)

            return df_groups

    def set_types(self):

        try:
            df_types = pd.read_csv(self.file_conn_types, index_col=0)
        except IOError as msg:
            self.logger.error(msg, exc_info=True)
        else:
            # asert check values >= 0
            not_good = df_types.loc[(df_types < 0).any(axis=1)].index.tolist()
            if not_good:
                raise Exception('Invalid value(s) for type(s): {}'.format(
                    not_good))

            # change arithmetic mean, std to logarithmic mean, std
            df_types['lognormal_strength'] = df_types.apply(
                lambda row: compute_logarithmic_mean_stddev(row['strength_mean'],
                                                            row['strength_std']),
                axis=1)

            df_types['lognormal_dead_load'] = df_types.apply(
                lambda row: compute_logarithmic_mean_stddev(row['dead_load_mean'],
                                                            row['dead_load_std']),
                axis=1)
            self.types = df_types.to_dict('index')

            return df_types

    def set_connections(self, df_types, df_groups):

        # assert each group_name entry of types is listed in conn_groups.csv
        not_good = df_types.loc[~df_types.group_name.isin(
            df_groups.index.tolist())].index.tolist()
        if not_good:
            raise Exception(
                'invalid group_name for type(s): {}'.format(not_good))

        # connections
        try:
            dump = self.read_file_connections(self.file_connections)
        except IOError as msg:
            self.logger.error(msg, exc_info=True)
        else:
            self.connections = self.process_read_file_connections(dump, df_types)
            self.list_connections = self.connections.index.tolist()

            self.damage_grid_by_sub_group = self.connections.groupby('sub_group')['grid_max'].apply(
                lambda x: x.unique()[0]).to_dict()
            self.list_groups = df_groups.index.tolist()
            self.connections['group_idx'] = self.connections['group_name'].apply(
                lambda x: self.list_groups.index(x))
            self.connections['flag_pressure'] = self.connections['group_name'].apply(
                lambda x: df_groups.loc[x, 'flag_pressure'])
            self.list_subgroups = self.connections['sub_group'].unique().tolist()

    def set_influences(self):

        # influences
        msg_conn = "Invalid connection name in {}"
        msg_conn_zone = "Invalid zone or connection name in {}"
        try:
            self.influences = self.read_influences(self.file_influences)
        except IOError as msg:
            self.logger.error(msg, exc_info=True)
        else:
            # check conn_name is also listed in connections.csv
            assert set(self.influences.keys()).issubset(self.list_connections), \
                msg_conn.format(self.file_influences)

            # # check zone name is listed in zones.csv
            for _, value in self.influences.items():
                assert set(value.keys()).issubset(self.list_connections + self.list_zones), \
                    msg_conn_zone.format(self.file_influences)

    def set_influence_patches(self):

        msg_conn = "Invalid connection name in {}"
        msg_conn_zone = "Invalid zone or connection name in {}"
        try:
            self.influence_patches = self.read_influence_patches(
                self.file_influence_patches)
        except IOError as msg:
            self.logger.error(msg, exc_info=True)
        else:
            # check conn_name is also listed in connections.csv
            assert set(self.influence_patches.keys()).issubset(self.list_connections), \
                msg_conn.format(self.file_influences)

            for _, value in self.influence_patches.items():
                assert set(value.keys()).issubset(
                    self.list_connections), msg_conn.format(self.file_influences)
                for _, sub_value in value.items():
                    assert set(sub_value.keys()).issubset(
                        self.list_connections + self.list_zones), msg_conn_zone.format(self.file_influences)

    def set_costings(self, df_groups):

        # costing
        self.costings, self.damage_order_by_water_ingress = \
            self.read_damage_costing_data(self.file_damage_costing_data)

        try:
            self.damage_factorings = self.read_damage_factorings(
                self.file_damage_factorings)
        except IOError as msg:
            self.logger.error(msg, exc_info=True)

        self.set_water_ingress_costings()

        self.costing_to_group = defaultdict(list)
        msg = 'damage scenario for {group} is {scenario}, ' \
              'but not defined in {file}'
        for key, value in df_groups['damage_scenario'].to_dict().items():
            if value in self.costings:
                self.costing_to_group[value].append(key)
            else:
                self.logger.warning(msg.format(group=key,
                                               scenario=value,
                                               file=FILE_DAMAGE_COSTING_DATA))

        if self.coverages is not None:
            self.costing_to_group['Wall debris damage'] = ['debris']

        # tidy up
        for key in self.costings.keys():
            if key not in self.costing_to_group:
                del self.costings[key]
                self.damage_order_by_water_ingress.remove(key)

        for key in self.water_ingress_costings.keys():
            if (key != 'WI only') and (key not in self.costing_to_group):
                del self.water_ingress_costings[key]

    def set_water_ingress_costings(self):
        msg = 'Invalid or missing name in {}'
        self.water_ingress_costings = self.read_water_ingress_costing_data(
            self.file_water_ingress_costing_data)
        assert set(self.water_ingress_costings.keys()).difference(
            self.costings.keys()) == {'WI only'}, msg.format(
            self.file_water_ingress_costing_data)

    def set_zones(self):

        try:
            df = self.read_file_zones(self.file_zones)
        except IOError as msg:
            self.logger.error(msg, exc_info=True)
        else:
            dic = df.to_dict('index')
            self.list_zones = df.index.tolist()
            self.zones = OrderedDict((k, dic.get(k)) for k in self.list_zones)

        for item in ['cpe_mean', 'cpe_str_mean', 'cpe_eave_mean', 'edge']:
            _file = getattr(self, 'file_zones_{}'.format(item))
            try:
                value = pd.read_csv(_file, names=self.header_for_cpe,
                                    index_col=0, skiprows=1).fillna(value=np.nan)
            except IOError as msg:
                self.logger.error(msg, exc_info=True)
            else:
                # name is listed in zones
                msg = "Invalid or missing zone names in {}"
                assert set(value.index) == set(df.index), msg.format(_file)

                # check zone has values for 8 directions
                not_good = value.loc[value.isnull().any(axis=1)].index.tolist()
                if not_good:
                    raise ValueError(
                        'Invalid entry(ies) for zone(s): {} in {}'.format(not_good, _file))

                if 'cpe' in item:
                    # Cpe is between -5 and 5
                    not_good = value.loc[
                        ((value < -5.0) | (value > 5.0)).any(axis=1)].index.tolist()
                    if not_good:
                        self.logger.warning(
                            'Invalid Cpe for zone(s): {} in {}'.format(
                                not_good, _file))
                else:
                    # edge value should be either 0 or 1.
                    not_good = value.loc[
                        (~(value.isin([0, 1]))).any(axis=1)].index.tolist()
                    if not_good:
                        raise ValueError(
                            'Invalid value for zone(s): {} in {}'.format(
                                not_good, _file))

                for name, zone in self.zones.items():
                    zone[item] = value.loc[name].tolist()

    @classmethod
    def read_file_zones(cls, file_zones):

        logger = logging.getLogger(__name__)

        msg = 'Coordinates should consist of at least 3 points: {}'
        dump = []
        with open(file_zones, 'rU') as f:
            next(f)  # skip the first line
            for line in f:
                fields = line.strip().rstrip(',').split(',')
                if len(fields) > 3:
                    tmp = [x.strip() for x in fields[:4]]

                    array = np.array([float(x) for x in fields[4:]])
                    if array.size:
                        try:
                            array = array.reshape((-1, 2))
                        except ValueError:
                            logger.warning(
                                'Coordinates are incomplete: {}'.format(tmp[0]))
                        else:
                            if array.size < 6:
                                logger.warning(msg.format(tmp[0]))
                            tmp.append(patches.Polygon(array))
                    else:
                        tmp.append([])
                    dump.append(tmp)

        df = pd.DataFrame(dump, columns=['zone_name', 'area', 'cpi_alpha',
                                         'wall_dir', 'coords'])
        df.set_index('zone_name', drop=True, inplace=True)

        df['centroid'] = df.apply(cls.return_centroid, axis=1)

        df['area'] = df['area'].astype(dtype=float)
        df['cpi_alpha'] = df['cpi_alpha'].astype(dtype=float)
        df['wall_dir'] = df['wall_dir'].astype(dtype=int)

        # check area >= 0
        not_good = df.loc[df['area'] < 0.0].index.tolist()
        if not_good:
            raise ValueError(
                'Invalid area for zone(s): {}'.format(not_good))

        # check 0=< cpi_alpha <=1
        not_good = df.loc[
            (df['cpi_alpha'] < 0.0) | (df['cpi_alpha'] > 1.0)].index.tolist()
        if not_good:
            raise ValueError(
                'Invalid cpi_alpha for zone(s): {}'.format(not_good))

        return df

    @staticmethod
    def return_centroid(row):
        try:
            return row['coords']._get_xy()[:-1].mean(axis=0).tolist()
        except AttributeError:
            return []

    @classmethod
    def read_file_connections(cls, file_connections):

        logger = logging.getLogger(__name__)

        msg = 'Coordinates should consist of at least 3 points: {}'
        dump = []
        with open(file_connections, 'rU') as f:
            next(f)  # skip the first line
            for line in f:
                fields = line.strip().rstrip(',').split(',')
                if len(fields) > 3:
                    tmp = [x.strip() for x in fields[:4]]
                    array = np.array([float(x) for x in fields[4:]])
                    if array.size:
                        try:
                            array = array.reshape((-1, 2))
                        except ValueError:
                            logger.warning(
                                'Coordinates are incomplete: {}'.format(tmp[0]))
                        else:
                            if array.size < 6:
                                logger.warning(msg.format(tmp[0]))
                            tmp.append(patches.Polygon(array))
                    else:
                        tmp.append([])

                    dump.append(tmp)

        return dump

    @classmethod
    def process_read_file_connections(cls, dump, types):
        df = pd.DataFrame(dump, columns=['conn_name', 'type_name', 'zone_loc',
                                         'section', 'coords'])
        df['conn_name'] = df['conn_name'].astype(int)
        df.set_index('conn_name', drop=True, inplace=True)

        df['centroid'] = df.apply(cls.return_centroid, axis=1)

        df['group_name'] = types.loc[df['type_name'], 'group_name'].values

        if df['group_name'].isnull().sum():
            msg = 'Invalid type_name(s) found in connections.csv'
            raise Exception(msg)

        df['sub_group'] = df.apply(
            lambda row: row['group_name'] + row['section'], axis=1)
        df['costing_area'] = df['type_name'].apply(
            lambda x: types.loc[x, 'costing_area'])
        for item in ['costing_area', 'lognormal_strength', 'lognormal_dead_load']:
            df[item] = df['type_name'].apply(lambda x: types.loc[x, item])

        df['grid_raw'] = df['zone_loc'].apply(get_grid_from_zone_location)
        df = df.join(df.groupby('sub_group')['grid_raw'].apply(
            lambda x: tuple(map(min, *x))), on='sub_group', rsuffix='_min')

        df['grid'] = df.apply(cls.get_diff_tuples,
                              args=('grid_raw', 'grid_raw_min'), axis=1)
        df = df.join(df.groupby('sub_group')['grid'].apply(
            lambda x: tuple(map(max, *x))), on='sub_group', rsuffix='_max')

        return df

    @staticmethod
    def get_diff_tuples(row, key1, key2):
        return tuple([row[key1][i] - row[key2][i] for i in range(2)])

    @staticmethod
    def read_damage_costing_data(filename):
        """

        Args:
            filename:

        Returns:

        Note: damage costing data may contain

        """
        logger = logging.getLogger(__name__)
        costing = {}
        damage_order_by_water_ingress = []

        try:
            damage_costing = pd.read_csv(filename, index_col=0)
        except IOError as msg:
            logger.error(msg, exc_info=True)
        else:

            # both envelope_factor and internal_factor formula_type entry are
            # either 1 or 2.
            sel_col = ['envelope_factor_formula_type', 'internal_factor_formula_type']
            not_good = damage_costing.loc[
                (~damage_costing[sel_col].isin(COSTING_FORMULA_TYPES)).any(axis=1)].index.tolist()
            if not_good:
                raise ValueError(
                    'Invalid formula_type for damage scenario(s): {}'.format(not_good))

            for key, item in damage_costing.iterrows():
                # if groups['damage_scenario'].isin([key]).any():
                costing[key] = Costing(name=key, **item)

            for item in damage_costing['water_ingress_order'].sort_values().index:
                # if groups['damage_scenario'].isin([item]).any():
                damage_order_by_water_ingress.append(item)

        return costing, damage_order_by_water_ingress

    @staticmethod
    def read_water_ingress_costing_data(filename):
        logger = logging.getLogger(__name__)
        dic = {}
        names = ['name', 'water_ingress', 'base_cost', 'formula_type', 'coeff1',
                 'coeff2', 'coeff3']
        try:
            tmp = pd.read_csv(filename, names=names, header=0)
        except IOError as msg:
            logger.error(msg, exc_info=True)
        else:
            not_good = tmp.loc[
                (~tmp['formula_type'].isin([1, 2]))].index.tolist()
            if not_good:
                raise ValueError(
                    'Invalid formula_type for water ingress costing: {}'.format(not_good))
            for key, grouped in tmp.groupby('name'):
                # if groups['damage_scenario'].isin([key]).any() or (key == 'WI only'):
                grouped = grouped.set_index('water_ingress')
                grouped['costing'] = grouped.apply(
                    lambda row: WaterIngressCosting(**row), axis=1)
                dic[key] = grouped
        return dic

    @staticmethod
    def read_front_facing_walls(filename):
        _dic = {}
        with open(filename, 'rU') as f:
            next(f)  # skip the first line
            for line in f:
                fields = line.strip().rstrip(',').split(',')
                _dic[fields[0]] = [int(x) for x in fields[1:]]
        return _dic

    @staticmethod
    def get_lognormal_tuple(row, dic_, key):
        _type = row['coverage_type']
        _mean = dic_[_type]['{}_mean'.format(key)]
        _sign = np.sign(_mean)
        _mean = abs(_mean)
        _sd = dic_[_type]['{}_std'.format(key)]
        return pd.Series({'log_{}'.format(key): compute_logarithmic_mean_stddev(_mean, _sd),
                          'sign_{}'.format(key): _sign})

    @staticmethod
    def read_damage_factorings(filename):
        """Read damage factorings

        Args:
            filename: damage_factorings.csv

        Returns: dict

        """
        _dic = {}
        with open(filename, 'rU') as f:
            next(f)  # skip the first line
            for line in f:
                fields = line.strip().rstrip(',').split(',')
                _dic.setdefault(fields[0].strip(), []).append(fields[1].strip())

        return _dic

    @staticmethod
    def read_influences(filename):
        """Read influence coefficients

        Args:
            filename: influences.csv

        Returns: dict

        """
        _dic = {}
        msg = "infl coeff should be between -10 and 10, not {}"
        with open(filename, 'rU') as f:
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
                            key = value.strip()
                    elif i % 2:
                        try:
                            sub_key = int(value)
                        except ValueError:
                            sub_key = value.strip()
                    elif key and sub_key:
                        assert -10.0 < float(value) < 10.0, msg.format(value)
                        _dic.setdefault(key, {})[sub_key] = float(value)

        return _dic

    @staticmethod
    def read_influence_patches(filename):
        """Read influence patches

        Args:
            filename: influence_patches.csv

        Returns: dict

        """
        _dic = {}
        msg = "infl coeff should be between -10 and 10, not {}"
        with open(filename, 'rU') as f:
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
                            damaged_conn = value.strip()
                    elif i == 1:
                        try:
                            target_conn = int(value)
                        except ValueError:
                            target_conn = value.strip()
                    elif i % 2 == 0:
                        try:
                            sub_key = int(value)
                        except ValueError:
                            sub_key = value.strip()
                    elif damaged_conn and target_conn and sub_key:
                        assert -10.0 < float(value) < 10.0, msg.format(value)
                        _dic.setdefault(damaged_conn, {}
                                        ).setdefault(target_conn, {}
                                                     )[sub_key] = float(value)

        return _dic

    def set_region_name(self, value):
        try:
            assert value in self.debris_regions
        except AssertionError:
            self.logger.error('region_name {} is not defined'.format(value))
        else:
            self.region_name = value

    def set_wind_profiles(self):
        _file = os.path.join(self.path_wind_profiles, self.file_wind_profiles)
        try:
            df = pd.read_csv(_file, skiprows=1, header=None, index_col=0)
        except (IOError, ValueError):
            self.logger.error('invalid wind_profiles file: {}'.format(_file))
        else:
            self.profile_heights = df.index.tolist()
            self.wind_profiles = df.to_dict('list')

    def set_debris_types(self):

        debris_region = self.debris_regions[self.region_name]
        self.debris_types_ratio = []
        for key in DEBRIS_TYPES_KEYS:
            self.debris_types[key] = {}

            for item in DEBRIS_TYPES_ATTS:
                if item in ['frontal_area', 'mass', 'flight_time']:
                    mean = debris_region['{0}_{1}_mean'.format(key, item)]
                    std = debris_region['{0}_{1}_stddev'.format(key, item)]
                    mu_lnx, std_lnx = compute_logarithmic_mean_stddev(mean, std)
                    self.debris_types[key]['{}_mu'.format(item)] = mu_lnx
                    self.debris_types[key]['{}_std'.format(item)] = std_lnx
                else:
                    self.debris_types[key][item] = debris_region['{}_{}'.format(key, item)]

            self.debris_types_ratio.append(self.debris_types[key]['ratio'] / 100)

    def set_wind_dir_index(self):
        try:
            self.wind_dir_index = WIND_DIR.index(self.wind_direction.upper())
        except ValueError:
            self.logger.warning('8(i.e., RANDOM) is set for wind_dir_index')
            self.wind_dir_index = 8

    def save_config(self, filename=None):

        config = ConfigParser.RawConfigParser()

        key = 'main'
        config.add_section(key)
        # config.set(key, 'path_datafile', self.path_house_data)
        # config.set(key, 'parallel', self.parallel)
        config.set(key, 'no_models', self.no_models)
        config.set(key, 'model_name', self.model_name)
        config.set(key, 'random_seed', self.random_seed)

        config.set(key, 'wind_direction', WIND_DIR[self.wind_dir_index])
        config.set(key, 'wind_speed_min', self.wind_speed_min)
        config.set(key, 'wind_speed_max', self.wind_speed_max)
        config.set(key, 'wind_speed_increment', self.wind_speed_increment)
        config.set(key, 'wind_profiles', self.file_wind_profiles)
        config.set(key, 'regional_shielding_factor', self.regional_shielding_factor)

        key = 'options'
        config.add_section(key)
        for sub_key in self.flags:
            config.set(key, sub_key, self.flags.get(sub_key))

        key = 'heatmap'
        config.add_section(key)
        config.set(key, 'vmin', self.heatmap_vmin)
        config.set(key, 'vmax', self.heatmap_vmax)
        config.set(key, 'vstep', self.heatmap_vstep)

        key = 'fragility_thresholds'
        config.add_section(key)
        config.set(key, 'states', ', '.join(self.fragility_i_states))
        config.set(key, 'thresholds',
                   ', '.join(str(x) for x in self.fragility_i_thresholds))

        key = 'debris'
        config.add_section(key)
        config.set(key, 'region_name', self.region_name)
        config.set(key, 'staggered_sources', self.staggered_sources)
        config.set(key, 'source_items', self.source_items)
        config.set(key, 'building_spacing', self.building_spacing)
        config.set(key, 'debris_radius', self.debris_radius)
        config.set(key, 'boundary_radius', self.boundary_radius)
        config.set(key, 'debris_angle', self.debris_angle)
        # config.set(key, 'flight_time_mean', self.flight_time_mean)
        # config.set(key, 'flight_time_stddev', self.flight_time_stddev)

        key = 'construction_levels'
        config.add_section(key)
        config.set(key, 'levels',
                   ', '.join(self.construction_levels_levels))
        config.set(key, 'probs',
                   ', '.join(str(x) for x in self.construction_levels_probs))
        config.set(key, 'mean_factors',
                   ', '.join(str(x) for x in self.construction_levels_mean_factors))
        config.set(key, 'cv_factors',
                   ', '.join(str(x) for x in self.construction_levels_cv_factors))

        key = 'water_ingress'
        config.add_section(key)
        config.set(key, 'thresholds',
                   ', '.join(str(x) for x in self.water_ingress_i_thresholds))
        config.set(key, 'speed_at_zero_wi',
                   ', '.join(str(x) for x in self.water_ingress_i_speed_at_zero_wi))
        config.set(key, 'speed_at_full_wi',
                   ', '.join(str(x) for x in self.water_ingress_i_speed_at_full_wi))

        if filename:
            with open(filename, 'w') as configfile:
                config.write(configfile)
                self.logger.info('{} is created'.format(filename))
        else:
            with open(self.file_cfg, 'w') as configfile:
                config.write(configfile)
                self.logger.info('{} is created'.format(self.file_cfg))
