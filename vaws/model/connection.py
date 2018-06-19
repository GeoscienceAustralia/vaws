"""Connection module

    This module contains Connection class, ConnectionTypeGroup class, and Influence class.

"""
from __future__ import division, print_function

import numpy as np
import logging

from vaws.model.zone import Zone
from vaws.model.stats import (compute_arithmetic_mean_stddev, sample_lognormal,
                              sample_lognorm_given_mean_stddev)


def compute_load_by_zone(flag_pressure, dic_influences):

    msg1 = 'load by {name}: {coeff:.2f} * {area:.3f} * {pressure:.3f}'
    msg2 = 'load by {name}: {coeff:.2f} * {load:.3f}'

    load = 0.0
    for _, _inf in dic_influences.items():

        if isinstance(_inf.source, Zone):
            _pressure = getattr(_inf.source, 'pressure_{}'.format(flag_pressure))
            load += _inf.coeff * _inf.source.area * _pressure

            logging.debug(msg1.format(name=_inf.source.name,
                                      coeff=_inf.coeff,
                                      area=_inf.source.area,
                                      pressure=_pressure))

        else:
            _load_by_zone = compute_load_by_zone(flag_pressure, _inf.source.influences)
            load += _inf.coeff * _load_by_zone
            logging.debug(msg2.format(name=_inf.source.name,
                                      coeff=_inf.coeff,
                                      load=_load_by_zone))

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
        self.mean_factor = None
        self.cv_factor = None
        self.rnd_state = None

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._influences = None
        self._influence_patch = None

        self._strength = None
        self._dead_load = None
        self.capacity = -1.0  # default
        self.damaged = 0

    @property
    def influences(self):
        return self._influences

    @influences.setter
    def influences(self, _dic):
        assert isinstance(_dic, dict)
        self._influences = {}
        for key, value in _dic.items():
            self._influences[key] = Influence(name=key, coeff=value)

    @property
    def influence_patch(self):
        return self._influence_patch

    @influence_patch.setter
    def influence_patch(self, _dic):
        assert isinstance(_dic, dict)
        self._influence_patch = _dic

    @property
    def strength(self):
        """Return a sampled strength from lognormal distribution

        Args:
            mean_factor: factor adjusting arithmetic mean strength
            cv_factor: factor adjusting arithmetic cv
            rnd_state: None, integer or np.random.RandomState

        Returns: sample of strength following log normal dist.
        """
        if self._strength is None:
            mu, std = compute_arithmetic_mean_stddev(*self.lognormal_strength)
            mu *= self.mean_factor
            std *= self.cv_factor * self.mean_factor
            self._strength = sample_lognorm_given_mean_stddev(
                mu, std, self.rnd_state)
        return self._strength

    @property
    def dead_load(self):
        """
        Returns: sample of dead load following log normal dist.

        """
        if self._dead_load is None:
            self._dead_load = sample_lognormal(*self.lognormal_dead_load,
                                               rnd_state=self.rnd_state)
        return self._dead_load

    @property
    def load(self):
        """

        Returns: load

        """
        msg1 = 'load at conn {}'
        msg2 = 'dead load: {:.3f}'

        load = 0.0
        if not self.damaged:
            logging.debug(msg1.format(self.name))

            load = compute_load_by_zone(self.flag_pressure, self.influences)
            load += self.dead_load

            logging.debug(msg2.format(self.dead_load))

        return load

    def check_damage(self, wind_speed):
        """

            wind_speed:

        Returns:

        """
        msg = 'connection {name} of {group} damaged at {speed:.3f} ' \
              'b/c {strength:.3f} < {load:.3f}'

        # if load is negative, check failure
        if self.load < -1.0 * self.strength:

            logging.info(msg.format(name=self.name,
                                    group=self.sub_group,
                                    speed=wind_speed,
                                    strength=self.strength,
                                    load=self.load))

            self.damaged = 1
            self.capacity = wind_speed

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

        self._prop_damaged = 0
        self._damaged_area = 0

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

    @property
    def damaged_area(self):
        """

        Returns: prop_damaged_group, prop_damaged_area

        """
        if self._damaged_area < self.costing_area and self.no_connections:
            self._damaged_area = sum([x.damaged * x.costing_area for
                                      _, x in self.connections.items()])
        else:
            self._damaged_area = 0.0

        logging.info('group {}: damaged area: {:.3f}'.format(
            self.name, self._damaged_area))

        return self._damaged_area

    @property
    def prop_damaged(self):
        """

        Returns: prop_damaged_group, prop_damaged_area

        """
        if self._prop_damaged < 1 and self.no_connections:
            no_damaged = sum([x.damaged for _, x in self.connections.items()])
            self._prop_damaged = no_damaged / self.no_connections
        else:
            self._prop_damaged = 0

        logging.info('group {}: prop damaged: {:.3f}'.format(
            self.name, self._prop_damaged))

        return self._prop_damaged

    def check_damage(self, wind_speed):
        """
        Args:
            wind_speed: wind speed

        Returns:

        """

        self.damage_grid_previous = self.damage_grid.copy()

        for _, _connection in self.connections.items():

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
                source_connection.damaged = 1

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
