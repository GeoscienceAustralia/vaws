"""
    House Module - reference storage for House type information.
        - loaded from database
        - imported from '../data/houses/subfolder' (so we need python constr)
"""

import copy
from numpy import array, interp
from numpy.random import RandomState
import logging
from shapely.geometry import Polygon
from collections import OrderedDict

from vaws.connection import ConnectionTypeGroup
from vaws.zone import Zone
from vaws.config import Config
from vaws.debris import Debris, Coverage


class House(object):

    def __init__(self, cfg, rnd_state):

        assert isinstance(cfg, Config)
        assert isinstance(rnd_state, RandomState)

        self.cfg = cfg
        self.rnd_state = rnd_state

        # attributes assigned by self.read_house_data
        self.name = None
        self.width = None
        self.height = None
        self.length = None
        self.replace_cost = None
        self.cpe_cov = None
        self.cpe_str_cov = None
        self.cpe_k = None
        self.roof_rows = None
        self.roof_cols = None
        self.big_a = None
        self.big_b = None

        # debris related
        self.debris = None
        self.footprint = None

        # random variables
        self.wind_orientation = None  # 0 to 7
        self.construction_level = None
        self.profile = None
        self.str_mean_factor = None
        self.str_cov_factor = None
        self.mzcat = None

        self.groups = OrderedDict()  # list of conn type groups
        self.connections = {}  # dict of connections with name
        self.zones = {}  # dict of zones with id
        self.coverages = None  # pd.dataframe of coverages
        # self.zone_by_grid = {}  # dict of zones with zone loc grid in tuple

        # init house
        self.read_house_data()

        # house is consisting of connections, coverages, and zones
        self.set_house_wind_params()
        self.set_coverages()
        self.set_zones()
        self.set_connections()

        if self.cfg.flags['debris']:
            self.set_debris()

    def read_house_data(self):

        for key, value in self.cfg.house.iteritems():
            setattr(self, key, value)

    def set_zones(self):

        for _name, item in self.cfg.zones.iteritems():

            dic_zone = copy.deepcopy(item)
            dic_zone['cpe_mean'] = self.cfg.zones_cpe_mean[_name]
            dic_zone['cpe_str_mean'] = self.cfg.zones_cpe_str_mean[_name]
            dic_zone['cpe_eave_mean'] = self.cfg.zones_cpe_eave_mean[_name]
            dic_zone['is_roof_edge'] = self.cfg.zones_edge[_name]

            _zone = Zone(zone_name=_name, **dic_zone)

            _zone.sample_cpe(
                wind_dir_index=self.wind_orientation,
                cpe_cov=self.cpe_cov,
                cpe_k=self.cpe_k,
                cpe_str_cov=self.cpe_str_cov,
                big_a=self.big_a,
                big_b=self.big_b,
                rnd_state=self.rnd_state)

            self.zones[_name] = _zone
            # self.zone_by_grid[_zone.grid] = _zone

    def set_connections(self):

        for (_, sub_group_name), connections_by_sub_group in \
                self.cfg.connections.groupby(by=['group_idx', 'sub_group']):

            # sub_group
            group_name = connections_by_sub_group['group_name'].values[0]
            dic_group = copy.deepcopy(self.cfg.groups[group_name])
            dic_group['sub_group'] = sub_group_name

            _group = ConnectionTypeGroup(group_name=group_name,
                                         **dic_group)
            self.groups[sub_group_name] = _group

            _group.damage_grid = self.cfg.damage_grid_by_sub_group[sub_group_name]
            _group.costing = self.assign_costing(dic_group['damage_scenario'])
            _group.connections = connections_by_sub_group.to_dict('index')

            # linking with connections
            for connection_name, _connection in _group.connections.iteritems():

                self.connections[connection_name] = _connection
                self.set_connection_property(_connection)

                try:
                    _group.damage_grid[_connection.grid] = 0  # intact
                except IndexError:
                    logging.warning(
                        'conn grid {} does not exist within group grid {}'.format(
                            _connection.grid, _group.damage_grid))

                _group.connection_by_grid = _connection.grid, _connection

                self.link_connection_to_influence(_connection)

    def link_connection_to_influence(self, _connection):
        # linking connections either zones or connections
        for _inf in _connection.influences.itervalues():
            try:
                _inf.source = self.zones[_inf.name]
            except KeyError:
                try:
                    _inf.source = self.connections[_inf.name]
                except KeyError:
                    logging.warning('unable to associate {} with {} wrt influence'.format(
                        _connection.name, _inf.name))

    def set_connection_property(self, _connection):
        """

        Args:
            _connection: instance of Connection class

        Returns:

        """
        _connection.sample_strength(mean_factor=self.str_mean_factor,
                                    cov_factor=self.str_cov_factor,
                                    rnd_state=self.rnd_state)
        _connection.sample_dead_load(rnd_state=self.rnd_state)

        _connection.influences = self.cfg.influences[_connection.name]

        # influence_patches
        if _connection.name in self.cfg.influence_patches:
            _connection.influence_patch = \
                self.cfg.influence_patches[_connection.name]
        else:
            _connection.influence_patch = {}

    def assign_costing(self, key):
        """

        Args:
            key:

        Returns:

        """

        if key in self.cfg.costings:
            return self.cfg.costings[key]
        else:
            logging.warning('{} not in cfg.costings'.format(key))

    def set_debris(self):

        self.debris = Debris(self.cfg)

        points = []
        for item in self.cfg.footprint:
            points.append((item[0], item[1]))
        self.footprint = Polygon(points)

        self.debris.footprint = self.footprint, self.wind_orientation

        self.debris.rnd_state = self.rnd_state

        self.debris.coverages = self.coverages

    def set_coverages(self):

        if self.cfg.coverages is not None:

            self.coverages = self.cfg.coverages.copy()

            self.coverages['direction'] = self.coverages['wall_name'].apply(
                self.assign_windward)

            for _name, item in self.coverages.iterrows():

                item['cpe_mean'] = self.cfg.coverages_cpe_mean[_name]
                _coverage = Coverage(coverage_name=_name, **item)

                _coverage.sample_cpe(
                    wind_dir_index=self.wind_orientation,
                    cpe_cov=self.cpe_cov,
                    cpe_k=self.cpe_k,
                    cpe_str_cov=self.cpe_str_cov,
                    big_a=self.big_a,
                    big_b=self.big_b,
                    rnd_state=self.rnd_state)

                _coverage.sample_strength(rnd_state=self.rnd_state)

                self.coverages.loc[_name, 'coverage'] = _coverage

    def assign_windward(self, wall_name):

        windward_dir = self.cfg.wind_dir[self.wind_orientation]
        windward = self.cfg.front_facing_walls[windward_dir]

        leeward = self.cfg.front_facing_walls[self.cfg.wind_dir[
            (self.wind_orientation + 4) % 8]]

        side1, side2 = None, None
        if len(windward_dir) == 1:
            side1 = self.cfg.front_facing_walls[self.cfg.wind_dir[
                (self.wind_orientation + 2) % 8]]
            side2 = self.cfg.front_facing_walls[self.cfg.wind_dir[
                (self.wind_orientation + 6) % 8]]

        # assign windward, leeward, side
        if wall_name in windward:
            return 'windward'
        elif wall_name in leeward:
            return 'leeward'
        elif wall_name in side1:
            return 'side1'
        elif wall_name in side2:
            return 'side2'

    def assign_cpi(self):

        self.coverages['breached_area'] = \
            self.coverages['coverage'].apply(lambda x: x.breached_area)

        # need to confirm whether max by coverage or wall

        breached_area_by_wall = \
            self.coverages.groupby('direction')['breached_area'].sum()

        # check if opening is dominant or non-dominant
        max_breached = breached_area_by_wall[breached_area_by_wall ==
                                             breached_area_by_wall.max()]

        # cpi
        if len(max_breached) == 1:  # dominant opening

            area_else = breached_area_by_wall.sum() - max_breached.iloc[0]
            ratio = max_breached.iloc[0] / area_else

            row = (self.cfg.dominant_opening_ratio_thresholds < ratio).sum()
            direction = max_breached.index[0]

            cpi = self.cfg.cpi_table_for_dominant_opening[row][direction]

            if row > 1:  # factor with Cpe

                # cpe where the largest opening
                breached_area = self.coverages.loc[
                    self.coverages['direction'] == direction, 'breached_area']
                max_area = breached_area[breached_area == breached_area.max()]
                cpe_array = array([self.coverages.loc[i, 'coverage'].cpe
                                      for i in max_area.keys()])
                max_cpe = max(cpe_array.min(), cpe_array.max(), key=abs)
                cpi *= max_cpe

            logging.debug('cpi for bldg with dominant opening: {}'.format(cpi))

        elif len(max_breached) == 4:  # all equal openings
            if max_breached.iloc[0]:
                cpi = -0.3
            else:
                cpi = 0.0

            logging.debug('cpi for bldg without dominant opening: {}'.format(cpi))

        else:
            if breached_area_by_wall['windward']:
                cpi = 0.2
            else:
                cpi = -0.3

            logging.debug('cpi for bldg without dominant opening: {}'.format(cpi))

        return cpi

    def set_house_wind_params(self):
        """
        will be constant through wind steps
        Returns:
            wind_orientation,
            construction_level, str_mean_factor, str_cov_factor,
            profile, mzcat

        """
        # set wind_orientation
        if self.cfg.wind_dir_index == 8:
            self.wind_orientation = self.rnd_state.random_integers(0, 7)
        else:
            self.wind_orientation = self.cfg.wind_dir_index

        self.set_construction_level()

        self.set_wind_profile()

    def set_wind_profile(self):
        """

        Returns: profile, mzcat

        """

        self.profile = self.rnd_state.random_integers(1, 10)

        self.mzcat = interp(self.height, self.cfg.profile_heights,
                               self.cfg.wind_profile[self.profile])

    def set_construction_level(self):
        """

        Returns: construction_level, str_mean_factor, str_cov_factor

        """
        rv = self.rnd_state.random_integers(0, 100)
        key, value, cum_prob = None, None, 0.0
        for key, value in self.cfg.construction_levels.iteritems():
            cum_prob += value['probability'] * 100.0
            if rv <= cum_prob:
                break
        self.construction_level = key
        self.str_mean_factor = value['mean_factor']
        self.str_cov_factor = value['cov_factor']
