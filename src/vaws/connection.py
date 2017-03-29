"""
    Connection Module -

"""

import numpy as np
import logging

from vaws.zone import Zone
from vaws.stats import compute_arithmetic_mean_stddev, sample_lognormal, \
    sample_lognorm_given_mean_stddev


class Connection(object):

    def __init__(self, connection_name=None, **kwargs):
        """

        Args:
            connection_name:
        """

        assert isinstance(connection_name, int)
        self.name = connection_name

        self.type_name = None
        self.zone_loc = None
        self.group_name = None

        default_attr = {'type_name': self.type_name,
                        'zone_loc': self.zone_loc,
                        'group_name': self.group_name}

        default_attr.update(kwargs)
        for key, value in default_attr.iteritems():
            setattr(self, key, value)

        self._lognormal_strength = None
        self._lognormal_dead_load = None
        self._influences = None
        self._influence_patch = None
        self._grid = None  # zero-based col, row index

        self.strength = None
        self.dead_load = None
        self.capacity = 0
        self.damaged = 0
        self.load = None

    @property
    def lognormal_strength(self):
        return self._lognormal_strength

    @lognormal_strength.setter
    def lognormal_strength(self, value):
        assert isinstance(value, tuple)
        self._lognormal_strength = value

    @property
    def lognormal_dead_load(self):
        return self._lognormal_dead_load

    @lognormal_dead_load.setter
    def lognormal_dead_load(self, value):
        assert isinstance(value, tuple)
        self._lognormal_dead_load = value

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, _tuple):
        assert isinstance(_tuple, tuple)
        self._grid = _tuple

    @property
    def influences(self):
        return self._influences

    @influences.setter
    def influences(self, _dic):
        assert isinstance(_dic, dict)

        self._influences = {}
        for key, value in _dic.iteritems():
            self._influences[key] = Influence(infl_name=key,
                                              infl_coeff=value)

    @property
    def influence_patch(self):
        return self._influence_patch

    @influence_patch.setter
    def influence_patch(self, _dic):
        assert isinstance(_dic, dict)
        self._influence_patch = _dic

    def sample_strength(self, mean_factor, cov_factor, rnd_state):
        """

        Args:
            mean_factor: factor adjusting arithmetic mean strength
            cov_factor: factor adjusting arithmetic cov
            rnd_state:

        Returns: sample of strength following log normal dist.

        """
        mu, std = compute_arithmetic_mean_stddev(*self.lognormal_strength)
        mu *= mean_factor
        std *= cov_factor

        self.strength = sample_lognorm_given_mean_stddev(mu, std, rnd_state)

    def sample_dead_load(self, rnd_state):
        """

        Args:
            rnd_state:

        Returns: sample of dead load following log normal dist.

        """
        self.dead_load = sample_lognormal(*(self.lognormal_dead_load +
                                            (rnd_state,)))

    def cal_load(self):
        """

        Returns: load

        """

        self.load = 0.0

        if not self.damaged:

            logging.debug('computing load of connection {}'.format(self.name))

            for _inf in self.influences.itervalues():

                try:

                    temp = _inf.coeff * _inf.source.area * _inf.source.pressure

                    logging.debug(
                        'load by {}: {:.2f} x {:.3f} x {:.3f}'.format(
                            _inf.source.name, _inf.coeff, _inf.source.area,
                            _inf.source.pressure))

                except AttributeError:

                    temp = _inf.coeff * _inf.source.load

                    logging.debug(
                        'load by {}: {:.2f} times {:.3f}'.format(
                            _inf.source.name, _inf.coeff, _inf.source.load))

                self.load += temp

            logging.debug('dead load: {:.3f}'.format(self.dead_load))

            self.load += self.dead_load

    def check_damage(self, wind_speed):
        """

        Args:
            wind_speed:

        Returns:

        """

        # if load is negative, check failure
        if self.load < -1.0 * self.strength:

            self.damaged = 1
            self.capacity = wind_speed

            logging.info(
                'connection {} at {} damaged at {:.3f} '
                'b/c {:.3f} > {:.3f}'.format(self.name, self.grid, wind_speed,
                                             self.load, self.strength))

    def update_influence(self, source_connection, infl_coeff):
        """

        Args:
            source_connection: source connection (sheeting)
            infl_coeff:

        Returns: load

        """

        # looking at influences
        for _id, _infl in source_connection.influences.iteritems():

            # update influence coeff
            updated_coeff = infl_coeff * _infl.coeff

            if _id in self.influences:
                self.influences[_id].coeff += updated_coeff

            else:
                self.influences.update(
                    {_id: Influence(infl_coeff=updated_coeff, infl_name=_id)})
                self.influences[_id].source = _infl.source

        # logging.debug('influences of {}:{:.2f}'.format(self.name,
        #                                                _inf.coeff))


class ConnectionType(object):
    def __init__(self, type_name=None, **kwargs):
        """

        Args:
        """
        assert isinstance(type_name, str)
        self.name = type_name

        self.costing_area = None
        self.lognormal_dead_load = None
        self.lognormal_strength = None
        self.group_name = None

        default_attr = {'costing_area': self.costing_area,
                        'lognormal_dead_load': self.lognormal_dead_load,
                        'lognormal_strength': self.lognormal_strength,
                        'group_name': self.group_name}

        default_attr.update(kwargs)
        for key, value in default_attr.iteritems():
            setattr(self, key, value)

        self._connections = None
        self.no_connections = None

    @property
    def connections(self):
        return self._connections

    @connections.setter
    def connections(self, _dic):

        assert isinstance(_dic, dict)

        self._connections = {}
        for key, value in _dic.iteritems():

            _connection = Connection(connection_name=key, **value)
            _connection.lognormal_strength = self.lognormal_strength
            _connection.lognormal_dead_load = self.lognormal_dead_load

            self._connections[key] = _connection

        self.no_connections = len(self._connections)


class ConnectionTypeGroup(object):

    def __init__(self, group_name=None, **kwargs):
        """

        Args:
            inst: instance of database.ConnectionTypeGroup
        """

        self.name = group_name

        self.dist_order = None
        self.dist_dir = None
        self.damage_scenario = None
        self.trigger_collapse_at = None
        self.patch_dist = None
        self.set_zone_to_zero = None
        self.water_ingress_order = None

        default_attr = {'dist_order': self.dist_order,
                        'dist_dir': self.dist_dir,
                        'damage_scenario': self.damage_scenario,
                        'trigger_collapse_at': self.trigger_collapse_at,
                        'patch_dist': self.patch_dist,
                        'set_zone_to_zero': self.set_zone_to_zero,
                        'water_ingress_order': self.water_ingress_order}

        default_attr.update(kwargs)
        for key, value in default_attr.iteritems():
            setattr(self, key, value)

        self._costing = None
        self._costing_area = None
        self._no_connections = 0

        self._types = None

        self._damage_grid = None  # column (chr), row (num)
        # negative: no connection, 0: Intact,  1: Failed

        self._connection_by_grid = {}  # dict of connections with zone loc grid
        self._connection_by_name = {}  # dict of connections with conn name

        self.damaged = None
        self.damaged_area = 0.0

    @property
    def no_connections(self):
        return self._no_connections

    @no_connections.setter
    def no_connections(self, value):
        assert isinstance(value, int)
        self._no_connections += value

    @property
    def types(self):
        return self._types

    @types.setter
    def types(self, _dic):
        assert isinstance(_dic, dict)
        self._types = {}
        for key, value in _dic.iteritems():
            _type = ConnectionType(type_name=key, **value)
            self._types[key] = _type

    @property
    def costing_area(self):
        return self._costing_area

    @costing_area.setter
    def costing_area(self, value):
        assert isinstance(value, float)
        self._costing_area = value

    @property
    def damage_grid(self):
        return self._damage_grid

    @damage_grid.setter
    def damage_grid(self, _tuple):

        assert isinstance(_tuple, tuple)
        no_rows, no_cols = _tuple

        if self.dist_dir:
            self._damage_grid = -1 * np.ones(dtype=int,
                                             shape=(no_rows, no_cols))
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
    def connection_by_name(self):
        return self._connection_by_name

    @connection_by_name.setter
    def connection_by_name(self, _tuple):

        assert isinstance(_tuple, tuple)
        _connection_name, _connection = _tuple

        self._connection_by_name[_connection_name] = _connection

    def cal_damaged_area(self):
        """

        Returns: prop_damaged_group, prop_damaged_area

        """

        self.damaged_area = 0.0
        for _type in self.types.itervalues():
            for _connection in _type.connections.itervalues():
                self.damaged_area += _type.costing_area * _connection.damaged

        logging.info('group {}: damaged area: {:.3f}'.format(
            self.name, self.damaged_area))

    def check_damage(self, wind_speed):
        """
        Args:
            wind_speed: wind speed

        Returns:

        """

        self.damaged = []

        for _type in self.types.itervalues():
            for _connection in _type.connections.itervalues():

                _connection.cal_load()
                _connection.check_damage(wind_speed)

                if _connection.damaged:

                    if self.dist_dir == 'col':
                        self.damaged.append(_connection.grid[0])

                    elif self.dist_dir == 'row':
                        self.damaged.append(_connection.grid[1])

                    self.damage_grid[_connection.grid] = 1

    def distribute_damage(self):

        if self.dist_dir == 'row':  # batten
            # distributed over the same number

            # row: index of chr, col: index of number e.g, 0,0 : A1
            # for row, col in zip(*np.where(self.damage_grid)):
            for row, col0 in zip(*np.where(self.damage_grid[:, self.damaged]>0)):

                col = self.damaged[col0]

                intact = np.where(-self.damage_grid[:, col] == 0)[0]

                intact_left = intact[np.where(row > intact)[0]]
                intact_right = intact[np.where(row < intact)[0]]

                logging.debug('cols of intact connections:{}'.format(intact))

                if intact_left.size * intact_right.size > 0:
                    infl_coeff = 0.5  # can be prop. to distance later
                else:
                    infl_coeff = 1.0

                source_connection = self.connection_by_grid[row, col]

                try:
                    target_connection = self.connection_by_grid[intact_right[0], col]
                except IndexError:
                    pass
                else:
                    target_connection.update_influence(source_connection, infl_coeff)
                    logging.debug('Influence of connection {} is updated: '
                                  'connection {} with {:.2f}'.format(target_connection.name,
                                                               source_connection.name,
                                                               infl_coeff))

                try:
                    target_connection = self.connection_by_grid[intact_left[-1], col]
                except IndexError:
                    pass
                else:
                    target_connection.update_influence(source_connection, infl_coeff)
                    logging.debug('Influence of connection {} is updated: '
                                  'connection {} with {:.2f}'.format(target_connection.name,
                                                               source_connection.name,
                                                               infl_coeff))

                # empty the influence of source connection
                source_connection.influences.clear()
                self.connection_by_grid[row, col].load = 0.0

        elif self.dist_dir == 'col' and self.patch_dist:  # rafter
            # distributed over the same char.

            # row: index of chr, col: index of number e.g, 0,0 : A1
            # for row, col in zip(*np.where(group.damage_grid)):

            for row0, col in zip(
                    *np.where(self.damage_grid[self.damaged, :] > 0)):

                row = self.damaged[row0]

                source_connection = self.connection_by_grid[row, col]

                for target_name, infl_dic in source_connection.influence_patch.iteritems():

                    target_connection = self.connection_by_name[target_name]

                    for key, value in infl_dic.iteritems():
                        try:
                            target_connection.influences[key].coeff = value
                        except KeyError:
                            logging.error('{} not found in the influences {}'.
                                          format(key, target_name))

                    logging.debug('Influence of connection {} is updated by connection {}'
                                  .format(target_name, source_connection.name))

        elif self.dist_dir == 'col' and self.patch_dist == 0:  # sheeting

            # distributed over the same char.

            # row: index of chr, col: index of number e.g, 0,0 : A1
            # for row, col in zip(*np.where(group.damage_grid)):

            for row0, col in zip(
                    *np.where(self.damage_grid[self.damaged, :] > 0)):

                row = self.damaged[row0]

                intact = np.where(-self.damage_grid[row, :] == 0)[0]

                intact_left = intact[np.where(col > intact)[0]]
                intact_right = intact[np.where(col < intact)[0]]

                logging.debug('rows of intact zones:{}'.format(intact))

                if intact_left.size * intact_right.size > 0:
                    infl_coeff = 0.5
                else:
                    infl_coeff = 1.0

                source_connection = self.connection_by_grid[row, col]

                try:
                    target_connection = self.connection_by_grid[row, intact_right[0]]
                except IndexError:
                    pass
                else:
                    target_connection.update_influence(source_connection, infl_coeff)
                    logging.debug('Influence of connection {} is updated: '
                                  'connection {} with {:.2f}'.format(target_connection.name,
                                                               source_connection.name,
                                                               infl_coeff))

                try:
                    target_connection = self.connection_by_grid[row, intact_left[-1]]
                except IndexError:
                    pass
                else:
                    target_connection.update_influence(source_connection, infl_coeff)
                    logging.debug('Influence of connection {} is updated: '
                                  'connection {} with {:.2f}'.format(target_connection.name,
                                                               source_connection.name,
                                                               infl_coeff))

                # empty the influence of source connection
                source_connection.influences.clear()
                self.connection_by_grid[row, col].load = 0.0


class Influence(object):
    def __init__(self, infl_name=None, infl_coeff=None):
        """

        Args:
            infl_name:
            infl_coeff:
        """

        self.coeff = infl_coeff
        self.name = infl_name  # source connection or zone id
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
