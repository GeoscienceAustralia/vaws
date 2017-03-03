"""
    House Module - reference storage for House type information.
        - loaded from database
        - imported from '../data/houses/subfolder' (so we need python constr)
"""

import numpy as np
from shapely.geometry import Polygon
from collections import OrderedDict

from connection import ConnectionTypeGroup
from zone import Zone
from stats import calc_big_a_b_values
from scenario import Scenario


class House(object):

    def __init__(self, cfg, rnd_state):

        assert isinstance(cfg, Scenario)
        assert isinstance(rnd_state, np.random.RandomState)

        self.cfg = cfg
        self.rnd_state = rnd_state

        self.width = None
        self.height = None
        self.length = None
        self.replace_cost = None

        self.cpe_cov = None
        self.cpe_str_cov = None
        self.cpe_k = None

        self.footprint = None
        self.roof_rows = None
        self.roof_cols = None

        self.wind_orientation = None  # 0 to 7
        self.construction_level = None
        self.profile = None
        self.str_mean_factor = None
        self.str_cov_factor = None
        self.mzcat = None
        self.big_a = None
        self.big_b = None

        self.groups = OrderedDict()  # ordered dict of conn type groups
        self.types = dict()  # dict of conn types with id
        self.connections = dict()  # dict of connections with id
        self.zones = dict()  # dict of zones with id
        self.factors_costing = dict()  # dict of damage factoring with id
        self.walls = dict()
        self.patches = dict()

        self.zone_by_name = dict()  # dict of zones with zone name
        self.zone_by_grid = dict()  # dict of zones with zone loc grid in tuple

        # init house
        self.read_house_data()

        # house is consisting of connections, zones, and walls
        self.set_house_wind_params()
        self.set_zones()
        self.set_connections()

    def read_house_data(self):

        for att in self.cfg.house_attributes:
            setattr(self, att, self.cfg.df_house.loc[0, att])

    def set_zones(self):

        for zone_name, item in self.cfg.df_zones.iterrows():

            dic_zone = item.to_dict()
            dic_zone['cpe_mean'] = self.cfg.df_zones_cpe_mean.loc[
                zone_name].to_dict()
            dic_zone['cpe_str_mean'] = self.cfg.df_zones_cpe_str_mean.loc[
                zone_name].to_dict()
            dic_zone['cpe_eave_mean'] = self.cfg.df_zones_cpe_eave_mean.loc[
                zone_name].to_dict()
            dic_zone['is_roof_edge'] = self.cfg.df_zones_edge.loc[
                zone_name].to_dict()

            _zone = Zone(zone_name=zone_name, **dic_zone)

            _zone.sample_zone_cpe(
                wind_dir_index=self.wind_orientation,
                cpe_cov=self.cpe_cov,
                cpe_k=self.cpe_k,
                cpe_str_cov=self.cpe_str_cov,
                big_a=self.big_a,
                big_b=self.big_b,
                rnd_state=self.rnd_state)

            self.zones[zone_name] = _zone
            self.zone_by_grid[_zone.grid] = _zone

    def set_connections(self):

        for group_name, item in self.cfg.df_groups.iterrows():

            _group = ConnectionTypeGroup(group_name=group_name, **item)

            _group.damage_grid = self.roof_cols, self.roof_rows
            idx_costing = self.cfg.df_damage_costing[
                self.cfg.df_damage_costing['name'] ==
                _group.damage_scenario].index.tolist()[0]
            _group.costing = self.cfg.df_damage_costing.loc[idx_costing].to_dict()
            costing_area_by_group = 0.0

            df_selected_types = self.cfg.df_types.loc[
                self.cfg.df_types['group_name'] == group_name]
            _group.types = df_selected_types.to_dict('index')

            # linking with connections
            for type_name, _type in _group.types.iteritems():

                df_selected_conns = self.cfg.df_conns.loc[
                    self.cfg.df_conns['type_name'] == type_name]

                _type.connections = df_selected_conns.to_dict('index')
                _group.no_connections = _type.no_connections

                # linking with connections
                for conn_name, _conn in _type.connections.iteritems():

                    self.connections[conn_name] = _conn

                    _conn.sample_strength(mean_factor=self.str_mean_factor,
                                          cov_factor=self.str_cov_factor,
                                          rnd_state=self.rnd_state)
                    _conn.sample_dead_load(rnd_state=self.rnd_state)

                    _conn.grid = self.zones[_conn.zone_loc].grid
                    _conn.influences = self.cfg.dic_influences[_conn.name]

                    if _conn.name in self.cfg.dic_influence_patches:
                        _conn.influence_patch = \
                            self.cfg.dic_influence_patches[_conn.name]

                    _group.damage_grid[_conn.grid] = 0  # intact
                    costing_area_by_group += _type.costing_area
                    _group.conn_by_grid = _conn.grid, _conn
                    _group.conn_by_name = _conn.name, _conn

                    # linking connections either zones or connections\
                    for _inf in _conn.influences.itervalues():
                        try:
                            _inf.source = self.zones[_inf.name]
                        except KeyError:
                            _inf.source = self.connections[_inf.name]

                self.types[type_name] = _type

            _group.costing_area = costing_area_by_group
            self.groups[group_name] = _group

        # for item in db_house.walls:

        if self.cfg.flags['debris']:
            points = list()
            for _, item in self.cfg.df_footprint.iterrows():
                points.append((item[0], item[1]))
            self.footprint = Polygon(points)

    def set_house_wind_params(self):
        """
        will be constant through wind steps
        Returns:
            big_a, big_b,
            wind_orientation,
            construction_level, str_mean_factor, str_cov_factor,
            profile, mzcat

        """
        self.big_a, self.big_b = calc_big_a_b_values(shape_k=self.cpe_k)

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
        _wind_profile = self.cfg.wind_profile[self.profile]
        self.mzcat = np.interp(self.height, self.cfg.heights, _wind_profile)

    def set_construction_level(self):
        """

        Returns: construction_level, str_mean_factor, str_cov_factor

        """
        if self.cfg.flags['construction_levels']:
            rv = self.rnd_state.random_integers(0, 100)
            key, value, cum_prob = None, None, 0.0
            for key, value in self.cfg.construction_levels.iteritems():
                cum_prob += value['probability'] * 100.0
                if rv <= cum_prob:
                    break
            self.construction_level = key
            self.str_mean_factor = value['mean_factor']
            self.str_cov_factor = value['cov_factor']
        else:
            self.construction_level = 'medium'
            self.str_mean_factor = 1.0
            self.str_cov_factor = 1.0

            # @staticmethod
    # def string_first(value):
    #     """
    #     check if value starting with char otherwise re-arrange.
    #     Args:
    #         value: 'A12' or '12A'
    #
    #     Returns:
    #
    #     """
    #     for i, item in enumerate(value):
    #         try:
    #             float(item)
    #         except ValueError:
    #             return value[i:] + value[:i]
    #         else:
    #             continue



