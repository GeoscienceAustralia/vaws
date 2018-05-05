"""House module

    This module contains House class.

"""

import copy
import numpy as np
import logging
import collections

from vaws.model.connection import ConnectionTypeGroup
from vaws.model.zone import Zone
from vaws.model.config import Config
from vaws.model.debris import Debris
from vaws.model.coverage import Coverage


class House(object):

    def __init__(self, cfg, rnd_state):

        assert isinstance(cfg, Config)
        assert isinstance(rnd_state, np.random.RandomState)

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
        self.cpe_str_k = None
        self.big_a = None
        self.big_b = None
        self.big_a_str = None
        self.big_b_str = None

        # debris related
        self.debris = None

        # random variables
        self.wind_dir_index = None  # 0 to 7
        self.construction_level = None
        self.profile_index = None
        self.terrain_height_multiplier = None
        self.shielding_multiplier = None

        self.groups = collections.OrderedDict()  # list of conn type groups
        self.connections = {}  # dict of connections with name
        self.zones = {}  # dict of zones with id
        self.coverages = None  # pd.dataframe of coverages
        # self.zone_by_grid = {}  # dict of zones with zone loc grid in tuple

        # init house
        self.set_house_data()

        # house is consisting of connections, coverages, and zones
        self.set_coverages()
        self.set_zones()
        self.set_connections()

    @property
    def str_mean_factor(self):
        return self.cfg.construction_levels[
            self.construction_level]['mean_factor']

    @property
    def str_cov_factor(self):
        return self.cfg.construction_levels[
                self.construction_level]['cov_factor']

    def set_terrain_height_multiplier(self):
        self.terrain_height_multiplier = np.interp(
            self.height, self.cfg.profile_heights,
            self.cfg.wind_profiles[self.profile_index])

    def set_shielding_multiplier(self):
        """
        AS4055 (Wind loads for housing) defines the shielding multiplier
        for full, partial and no shielding as 0.85, 0.95 and 1.0, respectively.

        The following percentages are recommended for the shielding of houses
        well inside Australian urban areas:
        Full shielding: 63%, Partial shielding: 15%, No shielding: 22%.
        """
        if self.cfg.regional_shielding_factor <= 0.85:
            idx = (self.cfg.shielding_multiplier_thresholds <=
                   self.rnd_state.random_integers(0, 100)).sum()
            self.shielding_multiplier = self.cfg.shielding_multipliers[idx][1]
        else:
            self.shielding_multiplier = 1.0

    def set_wind_dir_index(self):
        if self.cfg.wind_dir_index == 8:
            self.wind_dir_index = self.rnd_state.random_integers(0, 7)
        else:
            self.wind_dir_index = self.cfg.wind_dir_index

    def set_profile_index(self):
        self.profile_index = self.rnd_state.random_integers(
            1, len(self.cfg.wind_profiles))

    def set_construction_level(self):
        """

        Returns: construction_level, str_mean_factor, str_cov_factor

        """
        rv = self.rnd_state.random_integers(0, 100)
        key, value, cum_prob = None, None, 0.0
        for key, value in self.cfg.construction_levels.items():
            cum_prob += value['probability'] * 100.0
            if rv <= cum_prob:
                break
        self.construction_level = key

    def set_house_data(self):

        for key, value in self.cfg.house.items():
            setattr(self, key, value)

        self.set_wind_dir_index()
        self.set_construction_level()
        self.set_profile_index()
        self.set_terrain_height_multiplier()
        self.set_shielding_multiplier()

    def set_zones(self):

        for _name, item in self.cfg.zones.items():

            _zone = Zone(name=_name, **item)

            _zone.sample_cpe(
                wind_dir_index=self.wind_dir_index,
                cpe_cov=self.cpe_cov,
                cpe_k=self.cpe_k,
                big_a=self.big_a,
                big_b=self.big_b,
                cpe_str_cov=self.cpe_str_cov,
                cpe_str_k=self.cpe_str_k,
                big_a_str=self.big_a_str,
                big_b_str=self.big_b_str,
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

            _group = ConnectionTypeGroup(name=group_name,
                                         **dic_group)
            self.groups[sub_group_name] = _group

            _group.damage_grid = self.cfg.damage_grid_by_sub_group[sub_group_name]
            _group.costing = self.assign_costing(dic_group['damage_scenario'])
            _group.connections = connections_by_sub_group.to_dict('index')

            # linking with connections
            for connection_name, _connection in _group.connections.items():

                self.connections[connection_name] = _connection
                self.set_connection_property(_connection)

                try:
                    _group.damage_grid[_connection.grid] = 0  # intact
                except IndexError:
                    logging.warning(
                        'conn grid {} does not exist within group grid {}'.format(
                            _connection.grid, _group.damage_grid))
                except TypeError:
                    logging.warning(
                        'conn grid does not exist for group {}'.format(_group.name))

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
            logging.warning('group {} undefined in cfg.costings'.format(key))

    def set_debris(self):

        coverages_debris = {}
        if self.coverages is not None and not self.coverages.empty:
            coverages_debris = self.coverages.loc[
                self.coverages.direction == 'windward', 'coverage'].to_dict()

        self.debris = Debris(cfg=self.cfg,
                             wind_dir_idx=self.wind_dir_index,
                             rnd_state=self.rnd_state,
                             coverages=coverages_debris)

    def set_coverages(self):

        if self.cfg.coverages is not None:

            df_coverages = self.cfg.coverages.copy()

            df_coverages['direction'] = df_coverages['wall_name'].apply(
                self.assign_windward)

            self.coverages = df_coverages[['direction', 'wall_name']].copy()
            self.coverages['breached_area'] = np.zeros_like(self.coverages.direction)

            for _name, item in df_coverages.iterrows():

                _coverage = Coverage(name=_name, **item)

                _coverage.sample_cpe(
                    wind_dir_index=self.wind_dir_index,
                    cpe_cov=self.cpe_cov,
                    cpe_k=self.cpe_k,
                    big_a=self.big_a,
                    big_b=self.big_b,
                    cpe_str_cov=self.cpe_str_cov,
                    cpe_str_k=self.cpe_str_k,
                    big_a_str=self.big_a_str,
                    big_b_str=self.big_b_str,
                    rnd_state=self.rnd_state)

                _coverage.sample_strength(rnd_state=self.rnd_state)

                self.coverages.loc[_name, 'coverage'] = _coverage

    def assign_windward(self, wall_name):

        windward_dir = self.cfg.wind_dir[self.wind_dir_index]
        windward = self.cfg.front_facing_walls[windward_dir]

        leeward = self.cfg.front_facing_walls[self.cfg.wind_dir[
            (self.wind_dir_index + 4) % 8]]

        side1, side2 = None, None
        if len(windward_dir) == 1:
            side1 = self.cfg.front_facing_walls[self.cfg.wind_dir[
                (self.wind_dir_index + 2) % 8]]
            side2 = self.cfg.front_facing_walls[self.cfg.wind_dir[
                (self.wind_dir_index + 6) % 8]]

        # assign windward, leeward, side
        if wall_name in windward:
            return 'windward'
        elif wall_name in leeward:
            return 'leeward'
        elif wall_name in side1:
            return 'side1'
        elif wall_name in side2:
            return 'side2'

    def compute_damaged_area_and_assign_cpi(self):

        # self.coverages['breached_area'] = \
        #     self.coverages['coverage'].apply(lambda x: x.breached_area)

        # self.debris.damaged_area = self.coverages['breached_area'].sum()

        breached_area_by_wall = \
            self.coverages.groupby('direction')['breached_area'].sum()

        # check if opening is dominant or non-dominant
        max_breached = breached_area_by_wall[breached_area_by_wall ==
                                             breached_area_by_wall.max()]

        # cpi
        if len(max_breached) == 1:  # dominant opening

            area_else = breached_area_by_wall.sum() - max_breached.iloc[0]
            if area_else:
                ratio = max_breached.iloc[0] / area_else
                row = (self.cfg.dominant_opening_ratio_thresholds <= ratio).sum()
            else:
                row = 4

            direction = max_breached.index[0]

            cpi = self.cfg.cpi_table_for_dominant_opening[row][direction]

            if row > 1:  # factor with Cpe

                # cpe where the largest opening
                breached_area = self.coverages.loc[
                    self.coverages['direction'] == direction, 'breached_area']
                max_area = breached_area[breached_area == breached_area.max()]
                cpe_array = np.array([self.coverages.loc[i, 'coverage'].cpe
                                   for i in max_area.keys()])
                max_cpe = max(cpe_array.min(), cpe_array.max(), key=abs)
                cpi *= max_cpe

            logging.debug('cpi for bldg with dominant opening: {}'.format(cpi))

        elif len(max_breached) == len(breached_area_by_wall):  # all equal openings
            if max_breached.iloc[0]:
                cpi = -0.3
            else:  # in case no opening
                cpi = 0.0

            logging.debug('cpi for bldg without dominant opening: {}'.format(cpi))

        else:
            if breached_area_by_wall['windward']:
                cpi = 0.2
            else:
                cpi = -0.3

            logging.debug('cpi for bldg without dominant opening: {}'.format(cpi))

        return cpi

