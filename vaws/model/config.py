"""
    Configuration module - user input to model run (file or gui)

    Note: regional shielding may be indirectly disabled by setting
    'regional_shielding_factor' to 1.0
    Note: differential shielding may be indirectly disabled by setting
    'building_spacing' to 0
"""
import copy
import os
import sys
import logging
import ConfigParser
import pandas as pd

from collections import OrderedDict, defaultdict
from numpy import array, linspace, reshape, sign
from scipy.stats import norm
from matplotlib.patches import Polygon

from vaws.model.stats import compute_logarithmic_mean_stddev, calc_big_a_b_values
from vaws.model.damage_costing import Costing, WaterIngressCosting
from vaws.model.zone import Zone

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

# debris data
FILE_DEBRIS = 'debris.csv'

# results
FILE_RESULTS = 'results.h5'
FILE_CURVE = 'results_curve.csv'


class Config(object):

    # lookup table mapping (0-7) to wind direction (8: random)
    wind_dir = ['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE', 'RANDOM']
    debris_types_keys = ['Rod', 'Compact', 'Sheet']
    debris_types_atts = ['mass', 'frontal_area', 'cdav', 'ratio']
    dominant_opening_ratio_thresholds = [0.5, 1.5, 2.5, 6.0]
    cpi_table_for_dominant_opening = \
        {0: {'windward': -0.3, 'leeward': -0.3, 'side1': -0.3, 'side2': -0.3},
         1: {'windward': 0.2, 'leeward': -0.3, 'side1': -0.3, 'side2': -0.3},
         2: {'windward': 0.7, 'leeward': 1.0, 'side1': 1.0, 'side2': 1.0},
         3: {'windward': 0.85, 'leeward': 1.0, 'side1': 1.0, 'side2': 1.0},
         4: {'windward': 1.0, 'leeward': 1.0, 'side1': 1.0, 'side2': 1.0}}

    # model dependent attributes
    house_bucket = ['profile', 'wind_orientation', 'construction_level',
                    'mzcat', 'str_mean_factor', 'str_cov_factor']

    att_non_float = ['construction_level']

    house_damage_bucket = ['qz', 'ms', 'cpi', 'collapse', 'di',
                           'di_except_water', 'repair_cost',
                           'water_ingress_cost', 'breached']

    debris_bucket = ['no_items', 'no_touched', 'damaged_area']

    # model and wind dependent attributes
    list_components = ['group', 'connection', 'zone', 'coverage']

    group_bucket = ['damaged_area']

    connection_bucket = ['damaged', 'capacity', 'load', 'strength', 'dead_load']

    zone_bucket = ['pressure', 'cpe', 'cpe_str', 'cpe_eave']

    coverage_bucket = ['strength_negative', 'strength_positive', 'load',
                       'breached', 'breached_area', 'capacity']

    dic_obj_for_fitting = {'weibull': 'vulnerability_weibull',
                           'lognorm': 'vulnerability_lognorm'}

    att_time_invariant = ['strength', 'strength_negative', 'strength_positive',
                          'dead_load', 'cpe', 'cpe_str', 'cpe_eave', 'capacity',
                          'collapse']

    def __init__(self, cfg_file=None):

        self.cfg_file = cfg_file

        self.house_name = None  # only used for gui display may be deleted later
        self.no_models = None
        self.random_seed = 0
        self.wind_speed_min = 0.0
        self.wind_speed_max = 0.0
        self.wind_speed_increment = 0.0
        self.wind_speed_steps = None  # set_wind_speeds
        self.speeds = None  # set_wind_speeds
        # self.terrain_category = None
        self.regional_shielding_factor = 1.0
        self.file_wind_profiles = None
        self.wind_profiles = None  # set_wind_profiles
        self.profile_heights = None  # set_wind_profiles
        self.wind_dir_index = None

        self.construction_levels = OrderedDict()
        self.fragility_thresholds = None

        # debris related
        self.region_name = None
        self.staggered_sources = None
        self.source_items = 0
        self.building_spacing = None
        self.debris_radius = 0.0
        self.debris_angle = 0.0
        self.flight_time_mean = 0.0
        self.flight_time_stddev = 0.0
        self.flight_time_log_mu = None
        self.flight_time_log_std = None
        self.debris_sources = None
        self.debris_regions = None
        self.debris_types = {}
        self.footprint = None

        # house data
        self.house = None
        self.zones = None
        self.zones_cpe_mean = None
        self.zones_cpe_eave_mean = None
        self.zones_cpe_str_mean = None
        self.zones_edge = None
        self.coverages = None

        self.groups = None
        self.types = None
        self.connections = None
        self.damage_grid_by_sub_group = None
        self.influences = None
        self.influence_patches = None

        self.list_groups = None
        self.list_connections = None
        self.list_zones = None
        self.list_coverages = None

        # damage costing
        self.costings = None
        self.damage_order_by_water_ingress = None
        self.costing_to_group = None
        self.water_ingress_costings = None
        self.damage_factorings = None
        self.water_ingress_given_di = None

        # debris related
        self.front_facing_walls = None
        self.coverages_cpe_mean = None
        self.coverages_area = 0.0

        self.file_results = None
        self.file_curve = None

        self.heatmap_vmin = 54.0
        self.heatmap_vmax = 95.0
        self.heatmap_vstep = 21

        self.flags = {}

        if not os.path.isfile(cfg_file):
            msg = 'Error: {} not found'.format(cfg_file)
            sys.exit(msg)
        else:
            self.path_cfg = os.path.dirname(os.path.realpath(cfg_file))
            self.path_output = os.path.join(self.path_cfg, OUTPUT_DIR)
            self.path_house_data = os.path.join(self.path_cfg, HOUSE_DATA)
            self.path_wind_profiles = os.path.join(self.path_cfg,
                                                   GUST_PROFILES_DATA)
            self.path_debris = os.path.join(self.path_cfg, DEBRIS_DATA)

            self.read_config()
            self.set_output_files()

    def set_output_files(self):
        if not os.path.exists(self.path_output):
            os.makedirs(self.path_output)
        self.file_results = os.path.join(self.path_output, FILE_RESULTS)
        self.file_curve = os.path.join(self.path_output, FILE_CURVE)

    def read_config(self):

        conf = ConfigParser.ConfigParser()
        conf.optionxform = str
        conf.read(self.cfg_file)

        self.read_main(conf, key='main')
        self.read_options(conf, key='options')
        self.read_heatmap(conf, key='heatmap')
        self.read_construction_levels(conf, key='construction_levels')
        self.read_fragility_thresholds(conf, key='fragility_thresholds')
        self.read_debris(conf, key='debris')
        self.read_water_ingress(conf, key='water_ingress')

        self.set_house()
        df_groups, list_groups = self.set_groups()
        df_types = self.set_types()
        self.set_connections(df_types, list_groups)
        self.set_zones()
        self.set_coverages()

        self.set_influences_and_influence_patches()
        self.set_costings(df_groups)

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
        self.house_name = conf.get(key, 'house_name')
        self.no_models = conf.getint(key, 'no_models')
        try:
            self.random_seed = conf.getint(key, 'random_seed')
        except ConfigParser.NoOptionError:
            self.random_seed = 0
        self.wind_speed_min = conf.getfloat(key, 'wind_speed_min')
        self.wind_speed_max = conf.getfloat(key, 'wind_speed_max')
        self.wind_speed_increment = conf.getfloat(key, 'wind_speed_increment')
        self.set_wind_speeds()
        self.set_wind_dir_index(conf.get(key, 'wind_direction'))
        self.regional_shielding_factor = conf.getfloat(
            key, 'regional_shielding_factor')
        self.set_wind_profiles(conf.get(key, 'wind_profiles'))

    def set_wind_speeds(self):
        self.wind_speed_steps = int(
            (self.wind_speed_max - self.wind_speed_min) /
            self.wind_speed_increment) + 1
        self.speeds = linspace(start=self.wind_speed_min,
                               stop=self.wind_speed_max,
                               num=self.wind_speed_steps,
                               endpoint=True)

    def read_water_ingress(self, conf, key):
        if conf.has_section(key):
            thresholds = [float(x) for x in
                          conf.get(key, 'thresholds').split(',')]
            lower = [float(x) for x in conf.get(key, 'lower').split(',')]
            upper = [float(x) for x in conf.get(key, 'upper').split(',')]
        else:
            thresholds = [0.1, 0.2, 0.5, 2.0]
            lower = [40.0, 35.0, 0.0, -20.0]
            upper = [60.0, 55.0, 40.0, 20.0]
            logging.info('default water ingress thresholds is used')
        self.water_ingress_given_di = pd.DataFrame(array([lower, upper]).T,
                                                   index=thresholds,
                                                   columns=['lower', 'upper'])
        self.water_ingress_given_di['wi'] = self.water_ingress_given_di.apply(
            self.return_norm_cdf, axis=1)

    def set_coverages(self):
        # coverages
        _file = os.path.join(self.path_house_data, FILE_COVERAGE_TYPES)
        try:
            coverage_types = pd.read_csv(_file, index_col=0).to_dict('index')
        except IOError as msg:
            logging.warning('{}'.format(msg))

        _file = os.path.join(self.path_house_data, FILE_COVERAGES)
        try:
            self.coverages = pd.read_csv(_file, index_col=0)
        except IOError as msg:
            logging.warning('{}'.format(msg))
        else:

            if not self.coverages.empty:

                _list = coverage_types[coverage_types.keys()[0]].keys()
                failure_keys = [s.replace('_mean', '')
                                for s in _list if '_mean' in s]

                for _key in failure_keys:
                    _df = self.coverages.apply(self.get_lognormal_tuple,
                                               args=(coverage_types, _key,),
                                               axis=1)
                    self.coverages = self.coverages.merge(_df,
                                                          left_index=True,
                                                          right_index=True)
                self.list_coverages = self.coverages.index.tolist()

                self.coverages_area = self.coverages['area'].sum()
            else:
                self.coverages = None

        names_ = ['name'] + range(8)
        _file = os.path.join(self.path_house_data, FILE_COVERAGES_CPE)
        try:
            self.coverages_cpe_mean = pd.read_csv(
                _file, names=names_, index_col=0, skiprows=1).to_dict('index')
        except (IOError, TypeError) as msg:
            logging.warning('{}'.format(msg))

        _file = os.path.join(self.path_house_data, FILE_FRONT_FACING_WALLS)
        try:
            self.front_facing_walls = self.read_front_facing_walls(_file)
        except IOError as msg:
            logging.warning('{}'.format(msg))

    def read_debris(self, conf, key):

        # global data
        _file = os.path.join(self.path_debris, FILE_DEBRIS)
        try:
            self.debris_regions = pd.read_csv(_file, index_col=0).to_dict()
        except IOError as msg:
            logging.warning('{}'.format(msg))

        self.staggered_sources = conf.getboolean(key, 'staggered_sources')
        self.source_items = conf.getint(key, 'source_items')
        for item in ['building_spacing', 'debris_radius', 'debris_angle',
                     'flight_time_mean', 'flight_time_stddev']:
            setattr(self, item, conf.getfloat(key, item))

        self.set_flight_time_log()
        self.set_region_name(conf.get(key, 'region_name'))

        _file = os.path.join(self.path_house_data, FILE_FOOTPRINT)
        try:
            self.footprint = pd.read_csv(_file, skiprows=1, header=None).values
        except IOError as msg:
            logging.warning('{}'.format(msg))

        if self.flags[key]:

            from vaws.model.debris import Debris

            self.debris_sources = Debris.create_sources(
                self.debris_radius,
                self.debris_angle,
                self.building_spacing,
                self.staggered_sources)

            self.set_debris_types()

    def set_flight_time_log(self):
        self.flight_time_log_mu, self.flight_time_log_std = \
            compute_logarithmic_mean_stddev(self.flight_time_mean,
                                            self.flight_time_stddev)

    def read_fragility_thresholds(self, conf, key):
        if conf.has_section(key):
            states = [x.strip() for x in conf.get(key, 'states').split(',')]
            thresholds = [float(x) for x in
                          conf.get(key, 'thresholds').split(',')]
        else:
            states = ['slight', 'medium', 'severe', 'complete']
            thresholds = [0.15, 0.45, 0.6, 0.9]
            logging.info('default fragility thresholds is used')
        self.fragility_thresholds = pd.DataFrame(thresholds,
                                                 index=states,
                                                 columns=['threshold'])
        self.fragility_thresholds['color'] = ['b', 'g', 'y', 'r']

    def get_construction_level(self, name):
        try:
            return (self.construction_levels[name]['probability'],
                    self.construction_levels[name]['mean_factor'],
                    self.construction_levels[name]['cov_factor'])
        except KeyError:
            msg = '{} not found in the construction_levels'.format(name)
            raise KeyError(msg)

    def set_construction_level(self, name, prob, mf, cf):
        self.construction_levels[name] = OrderedDict(zip(
            ['probability', 'mean_factor', 'cov_factor'],
            [prob, mf, cf]))

    def read_construction_levels(self, conf, key):
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
            self.construction_levels = {'medium': {'probability': 1.0,
                                                   'mean_factor': 1.0,
                                                   'cov_factor': 0.58}}
            logging.info('construction level is set to be medium')

    def read_heatmap(self, conf, key):
        for item in ['vmin', 'vmax', 'vstep']:
            try:
                setattr(self, 'heatmap_{}'.format(item), conf.getfloat(key, item))
            except ConfigParser.NoSectionError:
                logging.info('default value is used for heatmap')

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

    def set_house(self):

        # house data
        _file = os.path.join(self.path_house_data, FILE_HOUSE_DATA)
        try:
            tmp = pd.read_csv(_file, index_col=0, header=None).to_dict()[1]
        except IOError as msg:
            logging.error('{}'.format(msg))
        else:
            self.house = {}
            for k, v in tmp.iteritems():
                try:
                    self.house[k] = float(v)
                except ValueError:
                    self.house[k] = v

            self.house['big_a'], self.house['big_b'] = \
                calc_big_a_b_values(shape_k=self.house['cpe_k'])

            self.house['big_a_str'], self.house['big_b_str'] = \
                calc_big_a_b_values(shape_k=self.house['cpe_str_k'])

    def set_groups(self):
        _file = os.path.join(self.path_house_data, FILE_CONN_GROUPS)
        try:
            df_groups = pd.read_csv(_file, index_col=0).fillna('')
        except IOError as msg:
            logging.error('{}'.format(msg))
        else:
            self.groups = df_groups.to_dict('index')
            list_groups = df_groups.index.tolist()
        return df_groups, list_groups

    def set_types(self):

        _file = os.path.join(self.path_house_data, FILE_CONN_TYPES)
        try:
            df_types = pd.read_csv(_file, index_col=0)
        except IOError as msg:
            logging.error('{}'.format(msg))
        else:
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

    def set_connections(self, types, list_groups):
        # connections
        _file = os.path.join(self.path_house_data, FILE_CONNECTIONS)
        try:
            _connections = self.read_file_connections(_file, types)
        except IOError as msg:
            logging.error('{}'.format(msg))
        else:
            self.connections = _connections
            self.list_connections = self.connections.index.tolist()

            self.damage_grid_by_sub_group = self.connections.groupby('sub_group')['grid_max'].apply(
                lambda x: x.unique()[0]).to_dict()
            self.connections['group_idx'] = self.connections['group_name'].apply(
                lambda x: list_groups.index(x))
            self.list_groups = self.connections['sub_group'].unique().tolist()

    def set_influences_and_influence_patches(self):

        # influences
        try:
            self.influences = self.read_influences(
                os.path.join(self.path_house_data, FILE_INFLUENCES))
        except IOError as msg:
            logging.error('{}'.format(msg))

        try:
            self.influence_patches = self.read_influence_patches(
                os.path.join(self.path_house_data, FILE_INFLUENCE_PATCHES))
        except IOError as msg:
            logging.error('{}'.format(msg))

    def set_costings(self, df_groups):

        # costing
        _file = os.path.join(self.path_house_data, FILE_DAMAGE_COSTING_DATA)
        self.costings, self.damage_order_by_water_ingress = \
            self.read_damage_costing_data(_file)

        _file = os.path.join(self.path_house_data, FILE_DAMAGE_FACTORINGS)
        try:
            self.damage_factorings = self.read_damage_factorings(_file)
        except IOError as msg:
            logging.error('{}'.format(msg))

        _file = os.path.join(self.path_house_data,
                             FILE_WATER_INGRESS_COSTING_DATA)
        self.water_ingress_costings = self.read_water_ingress_costing_data(_file)

        self.costing_to_group = defaultdict(list)
        for key, value in df_groups['damage_scenario'].to_dict().iteritems():
            if value:
                self.costing_to_group[value].append(key)

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

    def set_zones(self):

        _file = os.path.join(self.path_house_data, FILE_ZONES)
        try:
            _df = pd.read_csv(_file, index_col=0, dtype={'cpi_alpha': float,
                                                         'area': float,
                                                         'wall_dir': int})
        except IOError as msg:
            logging.error('{}'.format(msg))
        else:
            _dict = _df.to_dict('index')
            self.list_zones = _df.index.tolist()
            self.zones = OrderedDict((k, _dict.get(k)) for k in self.list_zones)

        names_ = ['name'] + range(8)
        for item in ['cpe_mean', 'cpe_str_mean', 'cpe_eave_mean', 'edge']:
            _file = os.path.join(self.path_house_data,
                                 'zones_{}.csv'.format(item))
            try:
                _value = pd.read_csv(_file, names=names_, index_col=0,
                                     skiprows=1).to_dict('index')
            except IOError as msg:
                logging.error('{}'.format(msg))
            else:
                setattr(self, 'zones_{}'.format(item), _value)

    @classmethod
    def read_file_connections(cls, file_connections, types):

        dump = []
        with open(file_connections, 'r') as f:
            next(f)  # skip the first line
            for line in f:
                fields = line.strip().rstrip(',').split(',')
                tmp = [x.strip() for x in fields[:4]]
                _array = array([float(x) for x in fields[4:]])
                if _array.size:
                    try:
                        _array = reshape(_array, (-1, 2))
                    except ValueError:
                        logging.warning(
                            'Coordinates are incomplete: {}'.format(_array))
                    else:
                        tmp.append(Polygon(_array))
                else:
                    tmp.append([])

                dump.append(tmp)

        _df = pd.DataFrame(dump, columns=['conn_name', 'type_name', 'zone_loc',
                                          'section', 'coords'])
        _df['conn_name'] = _df['conn_name'].astype(int)
        _df.set_index('conn_name', drop=True, inplace=True)

        try:
            _df['centroid'] = _df['coords'].apply(lambda x: x._get_xy().mean(axis=0))
        except AttributeError:
            logging.warning('No coordinates are provided')

        _df['group_name'] = types.loc[_df['type_name'], 'group_name'].values
        _df['sub_group'] = _df.apply(
            lambda row: row['group_name'] + row['section'], axis=1)
        _df['costing_area'] = _df['type_name'].apply(
            lambda x: types.loc[x, 'costing_area'])
        for item in ['costing_area', 'lognormal_strength', 'lognormal_dead_load']:
            _df[item] = _df['type_name'].apply(lambda x: types.loc[x, item])

        _df['grid_raw'] = _df['zone_loc'].apply(Zone.get_grid_from_zone_location)
        _df = _df.join(_df.groupby('sub_group')['grid_raw'].apply(
            lambda x: tuple(map(min, *x))), on='sub_group', rsuffix='_min')

        _df['grid'] = _df.apply(cls.get_diff_tuples,
                                args=('grid_raw', 'grid_raw_min'), axis=1)
        _df = _df.join(_df.groupby('sub_group')['grid'].apply(
            lambda x: tuple(map(max, *x))), on='sub_group', rsuffix='_max')

        return _df

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
        costing = {}
        damage_order_by_water_ingress = []

        try:
            damage_costing = pd.read_csv(filename, index_col=0)
        except IOError as msg:
            logging.error('{}'.format(msg))
        else:
            for key, item in damage_costing.iterrows():
                # if groups['damage_scenario'].isin([key]).any():
                costing[key] = Costing(costing_name=key, **item)

            for item in damage_costing['water_ingress_order'].sort_values().index:
                # if groups['damage_scenario'].isin([item]).any():
                damage_order_by_water_ingress.append(item)

        return costing, damage_order_by_water_ingress

    @staticmethod
    def read_water_ingress_costing_data(filename):
        dic_ = {}
        names = ['name', 'water_ingress', 'base_cost', 'formula_type', 'coeff1',
                 'coeff2', 'coeff3']
        try:
            tmp = pd.read_csv(filename, names=names, header=0)
        except IOError as msg:
            logging.error('{}'.format(msg))
        else:
            for key, grouped in tmp.groupby('name'):
                # if groups['damage_scenario'].isin([key]).any() or (key == 'WI only'):
                grouped = grouped.set_index('water_ingress')
                grouped['costing'] = grouped.apply(
                    lambda row: WaterIngressCosting(costing_name=key, **row),
                    axis=1)
                dic_[key] = grouped
        return dic_

    @staticmethod
    def read_front_facing_walls(filename):
        _dic = {}
        with open(filename, 'r') as f:
            next(f)  # skip the first line
            for line in f:
                fields = line.strip().rstrip(',').split(',')
                _dic[fields[0]] = [int(x) for x in fields[1:]]
        return _dic

    @staticmethod
    def get_lognormal_tuple(row, dic_, key):
        _type = row['coverage_type']
        _mean = dic_[_type]['{}_mean'.format(key)]
        _sign = sign(_mean)
        _mean = abs(_mean)
        _sd = dic_[_type]['{}_std'.format(key)]
        return pd.Series({'log_{}'.format(key): compute_logarithmic_mean_stddev(_mean, _sd),
                          'sign_{}'.format(key): _sign})

    @staticmethod
    def read_damage_factorings(filename):
        """

        Args:
            filename: damage_factorings.csv
            connection_name, zone1_name, zone1_infl, (.....)
        Returns: dictionary

        """
        _dic = {}
        with open(filename, 'r') as f:
            next(f)  # skip the first line
            for line in f:
                fields = line.strip().rstrip(',').split(',')
                _dic.setdefault(fields[0].strip(), []).append(fields[1].strip())

        return _dic

    @staticmethod
    def read_influences(filename):
        """

        Args:
            filename: influences.csv
            connection_name, zone1_name, zone1_infl, (.....)
        Returns: dictionary

        """
        _dic = {}
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
                            key = value.strip()
                    elif i % 2:
                        try:
                            sub_key = int(value)
                        except ValueError:
                            sub_key = value.strip()
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
        _dic = {}
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
                        _dic.setdefault(damaged_conn, {}
                                        ).setdefault(target_conn, {}
                                                     )[sub_key] = float(value)

        return _dic

    def set_region_name(self, value):
        try:
            assert value in self.debris_regions
        except AssertionError:
            logging.error('region_name {} is not defined'.format(value))
        else:
            self.region_name = value

    def set_wind_profiles(self, value):
        _file = os.path.join(self.path_wind_profiles, value)
        try:
            _df = pd.read_csv(_file, skiprows=1, header=None, index_col=0)
        except IOError:
            logging.error('invalid wind_profiles file: {}'.format(_file))
        else:
            self.file_wind_profiles = value
            self.profile_heights = _df.index.tolist()
            self.wind_profiles = _df.to_dict('list')

    def set_debris_types(self):

        _debris_region = self.debris_regions[self.region_name]

        for key in self.__class__.debris_types_keys:

            self.debris_types[key] = {}
            for item in self.__class__.debris_types_atts:

                if item in ['frontal_area', 'mass']:
                    _mean = _debris_region['{0}_{1}_mean'.format(key, item)]
                    _std = _debris_region['{0}_{1}_stddev'.format(key, item)]
                    mu_lnx, std_lnx = compute_logarithmic_mean_stddev(_mean, _std)
                    self.debris_types[key]['{}_mu'.format(item)] = mu_lnx
                    self.debris_types[key]['{}_std'.format(item)] = std_lnx
                else:
                    self.debris_types[key][item] = _debris_region['{}_{}'.format(key, item)]

    def set_wind_dir_index(self, wind_dir_str):
        try:
            self.wind_dir_index = self.__class__.wind_dir.index(wind_dir_str.upper())
        except ValueError:
            logging.warning('8(i.e., RANDOM) is set for wind_dir_index')
            self.wind_dir_index = 8

    def save_config(self):

        config = ConfigParser.RawConfigParser()

        key = 'main'
        config.add_section(key)
        # config.set(key, 'path_datafile', self.path_house_data)
        # config.set(key, 'parallel', self.parallel)
        config.set(key, 'no_models', self.no_models)
        config.set(key, 'house_name', self.house_name)
        config.set(key, 'random_seed', self.random_seed)

        config.set(key, 'wind_direction', self.__class__.wind_dir[self.wind_dir_index])
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
        config.set(key, 'states', ', '.join(self.fragility_thresholds.index))
        config.set(key, 'thresholds',
                   ', '.join(str(x) for x in
                             self.fragility_thresholds['threshold'].values))

        key = 'debris'
        config.add_section(key)
        config.set(key, 'region_name', self.region_name)
        config.set(key, 'staggered_sources', self.staggered_sources)
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

        key = 'water_ingress'
        config.add_section(key)
        config.set(key, 'thresholds', ', '.join(
            [str(x) for x in self.water_ingress_given_di.index.values]))

        for item in ['lower', 'upper']:
            config.set(key, item, ', '.join(
                [str(x) for x in self.water_ingress_given_di[item].values]))

        with open(self.cfg_file, 'wb') as configfile:
            config.write(configfile)
            print('{} is created'.format(self.cfg_file))

