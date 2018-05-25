"""Connection module

    This module contains Connection class, ConnectionTypeGroup class, and Influence class.

"""
from __future__ import division, print_function

import numpy as np
import logging

from vaws.model.zone import Zone
from vaws.model.stats import compute_arithmetic_mean_stddev, sample_lognormal, \
    sample_lognorm_given_mean_stddev


def compute_load_by_zone(flag_pressure, dic_influences):

    load = 0.0

    for _, _inf in dic_influences.items():

        if isinstance(_inf.source, Zone):
            _pressure = getattr(_inf.source, 'pressure_{}'.format(flag_pressure))
            load += _inf.coeff * _inf.source.area * _pressure

            logging.debug(
                'load by {}: {:.2f} * {:.3f} * {:.3f}'.format(
                    _inf.source.name, _inf.coeff, _inf.source.area, _pressure))

        else:
            _load_by_zone = compute_load_by_zone(flag_pressure, _inf.source.influences)
            load += _inf.coeff * _load_by_zone
            logging.debug(
                'load by {}: {:.2f} * {:.3f}'.format(
                    _inf.source.name, _inf.coeff, _load_by_zone))

    return load


class Connection(object):

    def __init__(self, name=None, **kwargs):
        """

        Args:
            name:
        """

        self.name = name

        self.type_name = None
        self.zone_loc = None
        self.group_name = None
        self.sub_group = None
        self.flag_pressure = None
        self.section = None
        self.grid = None
        self.grid_raw = None
        self.centroid = None
        self.coords = None
        self.costing_area = None
        self.lognormal_strength = None
        self.lognormal_dead_load = None

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._influences = None
        self._influence_patch = None

        self.strength = None
        self.dead_load = None
        self.capacity = -1.0  # default
        self.damaged = 0
        self.load = None

    @property
    def influences(self):
        return self._influences

    @influences.setter
    def influences(self, _dic):
        assert isinstance(_dic, dict)

        self._influences = {}
        for key, value in _dic.items():
            self._influences[key] = Influence(name=key,
                                              coeff=value)

    @property
    def influence_patch(self):
        return self._influence_patch

    @influence_patch.setter
    def influence_patch(self, _dic):
        assert isinstance(_dic, dict)
        self._influence_patch = _dic

    def sample_strength(self, mean_factor, cv_factor, rnd_state):
        """Return a sampled strength from lognormal distribution

        Args:
            mean_factor: factor adjusting arithmetic mean strength
            cv_factor: factor adjusting arithmetic cv
            rnd_state:

        Returns: sample of strength following log normal dist.

        """
        mu, std = compute_arithmetic_mean_stddev(*self.lognormal_strength)

        mu *= mean_factor
        std *= cv_factor * mean_factor

        self.strength = sample_lognorm_given_mean_stddev(mu, std, rnd_state)

    def sample_dead_load(self, rnd_state):
        """

        Args:
            rnd_state:

        Returns: sample of dead load following log normal dist.

        """
        self.dead_load = sample_lognormal(*(self.lognormal_dead_load +
                                            (rnd_state,)))

    def compute_load(self):
        """

        Returns: load

        """

        self.load = 0.0

        if not self.damaged:

            logging.debug('computing load at conn: {}'.format(self.name))

            self.load = compute_load_by_zone(self.flag_pressure, self.influences)

            logging.debug('dead load: {:.3f}'.format(self.dead_load))

            self.load += self.dead_load

    def check_damage(self, wind_speed):
        """

            wind_speed:

        Returns:

        """

        # if load is negative, check failure
        if self.load < -1.0 * self.strength:

            self.damaged = 1
            self.capacity = wind_speed

            logging.info(
                'connection {} of {} damaged at {:.3f} '
                'b/c {:.3f} < {:.3f}'.format(self.name, self.sub_group,
                                             wind_speed, self.strength, self.load))

    def update_influence(self, source_connection, influence_coeff):
        """

        Args:
            source_connection: source connection (sheeting)
            infl_coeff:

        Returns: load

        """

        # looking at influences
        for _id, _influence in source_connection.influences.items():

            # update influence coeff
            updated_coeff = influence_coeff * _influence.coeff

            if _id in self.influences:
                self.influences[_id].coeff += updated_coeff

            else:
                self.influences.update(
                    {_id: Influence(coeff=updated_coeff, name=_id)})
                self.influences[_id].source = _influence.source

        # logging.debug('influences of {}:{:.2f}'.format(self.name,
        #                                                _inf.coeff))


class ConnectionTypeGroup(object):

    grid_idx_by_dist_dir = {'col': 0, 'row': 1}

    def __init__(self, name=None, **kwargs):
        """

        Args:
            inst: instance of database.ConnectionTypeGroup
        """

        self.name = name

        self.dist_order = None
        self.dist_dir = None
        self.damage_dist = None  # flag for damage distribution (0 or 1)
        self.damage_scenario = None
        self.trigger_collapse_at = None

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.no_connections = 0
        self.costing_area = 0

        self._connections = {}
        self._connection_by_grid = {}  # dict of connections with zone loc grid

        self._costing = None
        self._damage_grid = None  # column (chr), row (num)
        # negative: no connection, 0: Intact,  1: Failed

        self.damage_grid_previous = None
        self.damage_grid_index = []  # list of index of damaged connection
        self.damaged_area = 0.0
        self.prop_damaged = 0.0

    @property
    def connections(self):
        return self._connections

    @connections.setter
    def connections(self, _dic):
        assert isinstance(_dic, dict)

        for key, value in _dic.items():
            _connection = Connection(name=key, **value)
            self._connections[key] = _connection
            self.costing_area += _connection.costing_area
            self.no_connections += 1

    @property
    def damage_grid(self):
        return self._damage_grid

    @damage_grid.setter
    def damage_grid(self, _tuple):

        assert isinstance(_tuple, tuple)
        max_row_idx, max_col_idx = _tuple

        if self.dist_dir:
            self._damage_grid = -1 * np.ones(dtype=int,
                                             shape=(max_row_idx + 1,
                                                    max_col_idx + 1))
        else:
            self._damage_grid = None

    @property
    def connection_by_grid(self):
        return self._connection_by_grid

    @connection_by_grid.setter
    def connection_by_grid(self, _tuple):

        assert isinstance(_tuple, tuple)
        _connection_grid, _connection = _tuple

        self._connection_by_grid[_connection_grid] = _connection

    def compute_damaged_area(self):
        """

        Returns: prop_damaged_group, prop_damaged_area

        """

        if self.prop_damaged < 1.0:

            num_damaged = 0
            self.damaged_area = 0.0

            for _, _connection in self.connections.items():
                num_damaged += _connection.damaged
                self.damaged_area += _connection.costing_area * _connection.damaged

            try:
                self.prop_damaged = num_damaged / self.no_connections
            except ZeroDivisionError:
                self.prop_damaged = 0.0

        logging.info('group {}: damaged area: {:.3f}, prop damaged: {:.2f}'.format(
            self.name, self.damaged_area, self.prop_damaged))

    def check_damage(self, wind_speed):
        """
        Args:
            wind_speed: wind speed

        Returns:

        """

        self.damage_grid_previous = self.damage_grid.copy()

        for _connection in self.connections.itervalues():

            # _connection.compute_load()
            _connection.check_damage(wind_speed)

            if _connection.damaged:

                self.damage_grid[_connection.grid] = 1

        self.damage_grid_index = zip(*np.where(
            self.damage_grid - self.damage_grid_previous))

        self.damage_grid_previous = self.damage_grid.copy()

    def update_influence(self, house_inst):
        """
        
        Args:
            house_connections: 

        Returns:

        Notes:
            influence_patch will be applied regardless of patch_dist
            influence coeff will be distributed only if patch_dist == 0
            dist_dir == 'col': coeff will be distributed over the same char.
            dist_dir == 'row': coeff will be distributed over the same number
            row: chr, col: number
        """

        if self.dist_dir == 'patch':

            for row, col in self.damage_grid_index:
                source_connection = self.connection_by_grid[row, col]
                self.update_influence_by_patch(source_connection, house_inst)

        elif self.dist_dir in ['col', 'row']:

            for row, col in self.damage_grid_index:

                source_connection = self.connection_by_grid[row, col]

                intact_grids = []
                if self.dist_dir == 'col':
                    intact = np.where(-self.damage_grid[row, :] == 0)[0]
                    intact_left = intact[np.where(col > intact)[0]]
                    intact_right = intact[np.where(col < intact)[0]]

                    try:
                        intact_grids.append((row, intact_right[0]))
                    except IndexError:
                        pass

                    try:
                        intact_grids.append((row, intact_left[-1]))
                    except IndexError:
                        pass

                else:
                    intact = np.where(-self.damage_grid[:, col] == 0)[0]
                    intact_left = intact[np.where(row > intact)[0]]
                    intact_right = intact[np.where(row < intact)[0]]

                    try:
                        intact_grids.append((intact_right[0], col))
                    except IndexError:
                        pass

                    try:
                        intact_grids.append((intact_left[-1], col))
                    except IndexError:
                        pass

                logging.debug('row/col of intact zones: {}'.format(intact))

                if intact_left.size * intact_right.size > 0:
                    influence_coeff = 0.5
                else:
                    influence_coeff = 1.0

                for intact_grid in intact_grids:
                    try:
                        target_connection = self.connection_by_grid[intact_grid]
                    except IndexError:
                        logging.error('wrong intact_grid {} for {}'.format(
                            intact_grid, self.name))
                    else:
                        target_connection.update_influence(source_connection,
                                                           influence_coeff)
                        logging.debug('Influence of connection {} is updated: '
                                      'connection {} with {:.2f}'.format(target_connection.name,
                                                                   source_connection.name,
                                                                   influence_coeff))

                # empty the influence of source connection
                source_connection.influences.clear()
                source_connection.load = 0.0

                # update influence patch if applicable
                self.update_influence_by_patch(source_connection, house_inst)

    @staticmethod
    def update_influence_by_patch(damaged_connection, house_inst):
        """
        
        Args:
            damaged_connection: damaged_connection

        Returns:

        """
        for _name, _dic in damaged_connection.influence_patch.items():

            try:
                target_connection = house_inst.connections[_name]
            except KeyError:
                logging.error('target connection {} is not found'
                              'when {} is damaged'.format(_name, damaged_connection.name))
            else:
                if target_connection.damaged == 0 or (
                                target_connection.damaged == 1 and damaged_connection.name == _name):
                    target_connection.influences = _dic
                    house_inst.link_connection_to_influence(target_connection)
                    logging.debug(
                        'Influence of connection {} is updated by connection {}'
                        .format(_name, damaged_connection.name))


class Influence(object):
    def __init__(self, name=None, coeff=None):
        """

        Args:
            name:
            coeff:
        """

        self.name = name  # source connection or zone id
        self.coeff = coeff
        self._source = None

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, inst):
        """

        Args:
            inst: instance of either Zone or Connection class

        Returns:

        """
        assert isinstance(inst, Zone) or isinstance(inst, Connection)
        self._source = inst
