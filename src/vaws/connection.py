"""
    Connection Module - reference storage Connections
        - loaded from database
        - imported from '../data/houses/subfolder'

    FIXME!!!!! NEED TO CLARIFY MEAN, STD whether it's arithmetic or logarithmic
    and how to deal with zero value
"""

import numpy as np
import logging

from zone import Zone
from damage_costing import Costing
from stats import compute_arithmetic_mean_stddev, sample_lognormal, \
    sample_lognorm_given_mean_stddev


class Connection(object):
    def __init__(self, conn_name=None, **kwargs):
        """

        Args:
            conn_name:
        """

        assert isinstance(conn_name, int)
        self.name = conn_name

        default_attr = dict(edge=None,
                            type_name=None,
                            zone_loc=None,
                            group_name=None)

        default_attr.update(kwargs)
        for key, value in default_attr.iteritems():
            setattr(self, key, value)

        self._lognormal_strength = None
        self._lognormal_dead_load = None
        self._influences = None
        self._grid = None  # zero-based col, row index

        self.strength = None
        self.dead_load = None

        # self.failure_v = 0.0  # FIXME!!
        # self.failure_v_i = 0  # FIXME!!
        self.failure_v_raw = 9999  # dummy value

        self.damaged = False
        self.distributed = None

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

    # @property
    # def group_name(self):
    #     return self._group_name
    #
    # @group_name.setter
    # def group_name(self, value):
    #     assert isinstance(value, str)
    #     self._group_name = value

    @property
    def influences(self):
        return self._influences

    @influences.setter
    def influences(self, _dic):
        assert isinstance(_dic, dict)

        self._influences = dict()
        for key, value in _dic.iteritems():
            self._influences[key] = Influence(infl_name=key,
                                              infl_coeff=value)

    def sample_strength(self, mean_factor, cov_factor, rnd_state):
        """

        Args:
            mean_factor: factor adjusting arithmetic mean strength
            cov_factor: factor adjusting arithmetic cov
            rnd_state:

        Returns: sample of strength following log normal dist.

        """
        mu, std = compute_arithmetic_mean_stddev(self.lognormal_strength[0],
                                                 self.lognormal_strength[1])
        mu *= mean_factor
        std *= cov_factor

        self.strength = sample_lognorm_given_mean_stddev(mu, std, rnd_state)

    def sample_dead_load(self, rnd_state):
        """

        Args:
            rnd_state:

        Returns: sample of dead load following log normal dist.

        """
        self.dead_load = sample_lognormal(self.lognormal_dead_load[0],
                                          self.lognormal_dead_load[1],
                                          rnd_state)

    def cal_load(self):
        """

        Returns: load

        """

        self.load = 0.0

        if not self.damaged:

            logging.debug('computing load of conn {}'.format(self.name))

            if self.group_name == 'sheeting':

                for _inf in self.influences.itervalues():

                    try:
                        temp = _inf.coeff * _inf.source.area * _inf.source.pz

                        logging.debug(
                            'load by {}: {:.2f} x {:.3f} x {:.3f}'.format(
                                _inf.source.name, _inf.coeff, _inf.source.area,
                                _inf.source.pz))

                    except TypeError:
                        temp = 0.0

                    except AttributeError:
                        logging.critical('zone {} at {} has {:.1f}'.format(
                            _inf.source, _inf.id, _inf.coeff))

                    self.load += temp

                logging.debug('dead load: {:.3f}'.format(self.dead_load))

                self.load += self.dead_load

            else:

                # inf.source.load should be pre-computed
                for _inf in self.influences.itervalues():

                    try:
                        self.load += _inf.coeff * _inf.source.load

                        logging.debug(
                            'load by {}: {:.2f} times {:.3f}'.format(
                                _inf.source.name, _inf.coeff, _inf.source.load))

                    except TypeError:

                        logging.debug(
                            'conn {} at {} has {}'.format(
                                _inf.source.name, _inf.source.grid,
                                _inf.source.load))

            logging.debug('load of conn {}: {:.3f}'.format(self.name, self.load))

    def set_damage(self, wind_speed):
        """

        Args:
            wind_speed:

        Returns:
            damaged, failure_v_raw

        """
        self.damaged = True
        # self.damaged_by_dist = False
        # self.failure_v_i += 1
        self.failure_v_raw = wind_speed
        # num_ = wind_speed + (
        #                     self.failure_v_i - 1) * self.failure_v
        # denom_ = self.failure_v_i
        # self.failure_v = num_ / denom_

    def update_influence(self, source_conn, infl_coeff):
        """

        Args:
            source_conn: source conn (sheeting)
            infl_coeff:

        Returns: load

        """

        # looking at influences
        for _id, _infl in source_conn.influences.iteritems():

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

        default_attr = {'costing_area': None,
                        'lognormal_dead_load': None,
                        'lognormal_strength': None,
                        'group_name': None}

        default_attr.update(kwargs)
        for key, value in default_attr.iteritems():
            setattr(self, key, value)

        self._connections = None
        self.no_connections = None

        self.damage_capacity = None  # min wind speed at damage
        self.prop_damaged_type = None

    @property
    def connections(self):
        return self._connections

    @connections.setter
    def connections(self, _dic):

        assert isinstance(_dic, dict)

        self._connections = dict()
        for key, value in _dic.iteritems():

            _conn = Connection(conn_name=key, **value)
            _conn.lognormal_strength = self.lognormal_strength
            _conn.lognormal_dead_load = self.lognormal_dead_load

            self._connections[key] = _conn

        self.no_connections = len(self._connections)

    def damage_summary(self):
        """
        compute proportion of damaged connections and damage capacity
        Returns: prop_damaged_type, damage_capacity

        """
        num_damaged = 0
        _value = 99999
        for _conn in self.connections.itervalues():
            num_damaged += _conn.damaged
            _value = min(_conn.failure_v_raw, _value)

        try:
            self.prop_damaged_type = num_damaged / float(self.no_connections)
        except ZeroDivisionError:
            self.prop_damaged_type = 0.0

        self.damage_capacity = _value


class ConnectionTypeGroup(object):

    use_struct_pz_for = ['rafter', 'piersgroup', 'wallracking']

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
        # for item in inst.conn_types:
        #     _type = ConnectionType(item)
        #     self.types.setdefault(item.id, _type)
        #     self.no_connections += _type.no_connections

        # if self.dist_ord >= 0:
        #     self.enabled = True
        # else:
        #     self.enabled = False

        # self.primary_dir = None
        # self.secondary_dir = None
        # self.dist_by_col = None

        self._damage_grid = None  # column (chr), row (num)
        # negative: no connection, 0: Intact,  1: Failed

        self._conn_by_grid = dict()  # dict of connections with zone loc grid

        # self._dist_tuple = None

        self.damaged = None
        self.prop_damaged_group = None
        self.prop_damaged_area = 0.0
        self.repair_cost = None

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

        self._types = dict()
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
    def costing(self):
        return self._costing

    @costing.setter
    def costing(self, value):
        assert isinstance(value, dict)
        self._costing = Costing(costing_name=self.damage_scenario,
                                **value)

    @property
    def damage_grid(self):
        return self._damage_grid

    @damage_grid.setter
    def damage_grid(self, _tuple):

        assert isinstance(_tuple, tuple)
        no_rows, no_cols = _tuple

        if self.dist_dir:
            self._damage_grid = -1 * np.ones(dtype=int, shape=(no_rows, no_cols))
        else:
            self._damage_grid = None

    @property
    def conn_by_grid(self):
        return self._conn_by_grid

    @conn_by_grid.setter
    def conn_by_grid(self, _tuple):

        assert isinstance(_tuple, tuple)
        _conn_grid, _conn = _tuple

        self._conn_by_grid[_conn_grid] = _conn

    def cal_repair_cost(self, value):
        """

        Args:
            value: proportion of damaged area

        Returns:

        """
        try:
            self.repair_cost = self.costing.calculate_cost(value)
        except AssertionError:
            self.repair_cost = 0.0

        logging.debug('group {}: repair_cost: {:.3f} for value {:.3f}'.format(
            self.name, self.repair_cost, value))

    def cal_prop_damaged(self):
        """

        Returns: prop_damaged_group, prop_damaged_area

        """

        num_damaged = 0
        area_damaged = 0.0

        for _type in self.types.itervalues():
            for _conn in _type.connections.itervalues():
                num_damaged += _conn.damaged
                area_damaged += _type.costing_area * _conn.damaged

        try:
            self.prop_damaged_group = num_damaged / float(self.no_connections)
        except ZeroDivisionError:
            self.prop_damaged_group = 0.0

        try:
            self.prop_damaged_area = area_damaged / self.costing_area
        except ZeroDivisionError:
            self.prop_damaged_area = 0.0

        logging.debug('group {}: prop damaged: {:.3f}, prop. damaged area: {:.3f}'.format(
            self.name, self.prop_damaged_group, self.prop_damaged_area))

    def check_damage(self, wind_speed):
        """
        Args:
            wind_speed: wind speed

        Returns:

        """

        # hard coded for optimization
        # use_struct_pz = self.name in self.use_struct_pz_for

        self.damaged = list()

        for _type in self.types.itervalues():

            for _conn in _type.connections.itervalues():

                _conn.cal_load()

                # if load is negative, check failure
                if _conn.load < -1.0 * _conn.strength:

                    _conn.set_damage(wind_speed)

                    logging.debug(
                        'conn {} of {} at {} damaged at {:.3f} b/c load {:.3f} > strength {:.3f}'.format(
                            _conn.name, self.name, _conn.grid, wind_speed, _conn.load, _conn.strength))

                    if self.dist_dir == 'col':
                        self.damaged.append(_conn.grid[0])

                    elif self.dist_dir == 'row':
                        self.damaged.append(_conn.grid[1])

                    self.damage_grid[_conn.grid] = 1

            # summary by connection type
            _type.damage_summary()

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

                logging.debug('cols of intact conns:{}'.format(intact))

                if intact_left.size * intact_right.size > 0:
                    infl_coeff = 0.5  # can be prop. to distance later
                else:
                    infl_coeff = 1.0

                # logging.debug('conn {} at {} is distributed'.format(
                #     source_conn.name, source_conn.grid))
                # source_conn.distributed = True

                source_conn = self.conn_by_grid[row, col]

                try:
                    target_conn = self.conn_by_grid[intact_right[0], col]
                except IndexError:
                    pass
                else:
                    target_conn.update_influence(source_conn, infl_coeff)
                    logging.debug('Influence of conn {} is updated: '
                                  'conn {} with {:.2f}'.format(target_conn.name,
                                                               source_conn.name,
                                                               infl_coeff))

                try:
                    target_conn = self.conn_by_grid[intact_left[-1], col]
                except IndexError:
                    pass
                else:
                    target_conn.update_influence(source_conn, infl_coeff)
                    logging.debug('Influence of conn {} is updated: '
                                  'conn {} with {:.2f}'.format(target_conn.name,
                                                               source_conn.name,
                                                               infl_coeff))

                # empty the influence of source connection
                source_conn.influences.clear()
                self.conn_by_grid[row, col].load = 0.0

        elif self.dist_dir == 'col':  # sheeting
            # distributed over the same char.

            # row: index of chr, col: index of number e.g, 0,0 : A1
            # for row, col in zip(*np.where(group.damage_grid)):

            for row0, col in zip(
                    *np.where(self.damage_grid[self.damaged, :] > 0)):

                row = self.damaged[row0]

                # # looking at influences
                # linked_conn = None
                # for val in source_conn.influences.itervalues():
                #     if val.coeff == 1.0:
                #         linked_conn = val.source

                intact = np.where(-self.damage_grid[row, :] == 0)[0]

                intact_left = intact[np.where(col > intact)[0]]
                intact_right = intact[np.where(col < intact)[0]]

                logging.debug('rows of intact zones:{}'.format(intact))

                if intact_left.size * intact_right.size > 0:
                    infl_coeff = 0.5
                else:
                    infl_coeff = 1.0

                # if group.set_zone_to_zero:
                #     logging.debug('zone {} at {} distributed'.format(
                #         source_zone.name, source_zone.grid))
                #     source_zone.distributed = True

                # source_zone = self.house.zone_by_grid[row, col]
                source_conn = self.conn_by_grid[row, col]

                try:
                    target_conn = self.conn_by_grid[row, intact_right[0]]
                except IndexError:
                    pass
                else:
                    target_conn.update_influence(source_conn, infl_coeff)
                    logging.debug('Influence of conn {} is updated: '
                                  'conn {} with {:.2f}'.format(target_conn.name,
                                                               source_conn.name,
                                                               infl_coeff))

                try:
                    target_conn = self.conn_by_grid[row, intact_left[-1]]
                except IndexError:
                    pass
                else:
                    target_conn.update_influence(source_conn, infl_coeff)
                    logging.debug('Influence of conn {} is updated: '
                                  'conn {} with {:.2f}'.format(target_conn.name,
                                                               source_conn.name,
                                                               infl_coeff))

                # empty the influence of source connection
                source_conn.influences.clear()
                self.conn_by_grid[row, col].load = 0.0


class Influence(object):
    def __init__(self,
                 infl_name=None,
                 infl_coeff=None):
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

# unit tests
if __name__ == '__main__':
    import unittest
    import os
    from scenario import Scenario
    from house import House

    class MyTestCase(unittest.TestCase):

        @classmethod
        def setUpClass(cls):

            path = '/'.join(__file__.split('/')[:-1])
            cfg_file = os.path.join(path, '../../scenarios/test_sheeting_batten.cfg')
            cls.cfg = Scenario(cfg_file=cfg_file)

        def test_cal_prop_damaged(self):

            rnd_state = np.random.RandomState(1)
            house = House(self.cfg, rnd_state=rnd_state)
            group = house.groups['sheeting']  # sheeting

            for _conn in house.connections.itervalues():
                self.assertEqual(_conn.damaged, False)

            # 1: sheeting gable(1): 2 - 5
            # 2: sheeting eave(2): 7, 12, 13, 18, 19, 24, 25, 30
            # 3: sheeting corner(3): 1, 6
            # 4: sheeting(4): 8 - 11, 14 - 17, 20 - 23, 26 - 29
            for _id in [1, 4, 5, 7, 8, 11, 12]:
                house.connections[_id].set_damage(20.0)

            ref_dic = {'sheetinggable': 4, 'sheetingeave': 8,
                       'sheetingcorner': 2, 'sheeting': 16}
            ref_area = {'sheetinggable': 0.405, 'sheetingeave': 0.405,
                        'sheetingcorner': 0.225, 'sheeting': 0.81}
            for id_type, _type in group.types.iteritems():
                self.assertEqual(_type.no_connections, ref_dic[id_type])
                self.assertEqual(_type.costing_area, ref_area[id_type])

            # costing area by group
            self.assertAlmostEqual(group.costing_area, 18.27, places=2)

            ref_dic = {'sheetinggable': 0.5, 'sheetingeave': 0.25,
                       'sheetingcorner': 0.5, 'sheeting': 0.125}
            for id_type, _type in group.types.iteritems():
                _type.damage_summary()
                self.assertEqual(_type.prop_damaged_type, ref_dic[id_type])

            group.cal_prop_damaged()
            self.assertAlmostEqual(group.prop_damaged_group, 0.2333, places=4)
            self.assertAlmostEqual(group.prop_damaged_area, 0.1897, places=4)

        def test_cal_damage_capacity(self):

            rnd_state = np.random.RandomState(1)
            house = House(self.cfg, rnd_state)

            # type 1
            house.connections[2].set_damage(20.0)
            house.connections[5].set_damage(30.0)

            # type 2
            house.connections[7].set_damage(30.0)

            # type 3
            house.connections[1].set_damage(45.0)

            ref_dic = {'sheetinggable': 20.0, 'sheetingeave': 30.0,
                       'sheetingcorner': 45.0, 'sheeting': 9999.0}
            for id_type, _type in house.groups['sheeting'].types.iteritems():
                _type.damage_summary()
                self.assertAlmostEqual(_type.damage_capacity, ref_dic[id_type])

        def test_cal_load(self):

            rnd_state = np.random.RandomState(1)
            house = House(self.cfg, rnd_state)

            # compute zone pressures
            wind_dir_index = 3
            cpi = 0.0
            qz = 0.8187
            Ms = 1.0
            building_spacing = 0

            _zone = house.zones['A1']
            _zone.cpe = _zone.cpe_mean[0]  # originally randomly generated

            _zone.calc_zone_pressures(wind_dir_index,
                                      cpi,
                                      qz,
                                      Ms,
                                      building_spacing)

            self.assertAlmostEqual(_zone.cpi_alpha, 0.0, places=2)
            self.assertAlmostEqual(_zone.cpe_mean[0], -1.25, places=2)
            self.assertAlmostEqual(house.zones['A1'].area, 0.2025, places=4)

            # pz = qz * (cpe - cpi_alpha * cpi) * diff_shielding
            self.assertAlmostEqual(house.zones['A1'].pz, -1.0234, places=4)

            _conn = house.connections[1]

            # init
            self.assertEqual(_conn.damaged, False)
            self.assertEqual(_conn.load, None)
            self.assertAlmostEqual(_conn.dead_load, 0.0130, places=4)
            _conn.cal_load()

            # load = influence.pz * influence.coeff * influence.area + dead_load
            self.assertAlmostEqual(_conn.influences['A1'].source.area, 0.2025,
                                   places=4)
            self.assertAlmostEqual(_conn.influences['A1'].source.pz, -1.0234,
                                   places=4)
            self.assertAlmostEqual(_conn.load, -0.1942, places=4)

        def test_check_damage(self):

            rnd_state = np.random.RandomState(1)
            house = House(self.cfg, rnd_state)

            # compute zone pressures
            mzcat = 0.9235
            wind_speed = 75.0
            wind_dir_index = 3
            cpi = 0.0
            qz = 0.6 * 1.0e-3 * (wind_speed * mzcat) ** 2
            Ms = 1.0
            building_spacing = 0

            # compute pz using constant cpe
            for _zone in house.zones.itervalues():
                _zone.cpe = _zone.cpe_mean[0]
                _zone.calc_zone_pressures(wind_dir_index,
                                          cpi,
                                          qz,
                                          Ms,
                                          building_spacing)
                self.assertAlmostEqual(_zone.pz, qz*_zone.cpe_mean[0], places=4)

            # compute dead_load and strength using constant values
            for _conn in house.connections.itervalues():
                _conn.lognormal_dead_load = _conn.lognormal_dead_load[0], 0.0
                _conn.lognormal_strength = _conn.lognormal_strength[0], 0.0

                _conn.sample_dead_load(rnd_state)
                _conn.sample_strength(mean_factor=1.0, cov_factor=0.0,
                                      rnd_state=rnd_state)
                _conn.cal_load()

                self.assertAlmostEqual(_conn.dead_load,
                                       np.exp(_conn.lognormal_dead_load[0]),
                                       places=4)

                self.assertAlmostEqual(_conn.strength,
                                       np.exp(_conn.lognormal_strength[0]),
                                       places=4)

            # check load
            group = house.groups['sheeting']
            group.check_damage(wind_speed=wind_speed)

            ref_dic = {x: False for x in range(1, 61)}
            for i in [8, 14, 20, 26]:
                ref_dic[i] = True

            for id_conn, _conn in house.connections.iteritems():
                try:
                    self.assertEqual(_conn.damaged, ref_dic[id_conn])
                except AssertionError:
                    print '{}: {} vs {}'.format(_conn.name, _conn.damaged,
                                                ref_dic[id_conn])

            ref_prop = {'sheetinggable': 0.0, 'sheetingeave': 0.0,
                        'sheetingcorner': 0.0, 'sheeting': 0.25}
            ref_capacity = {'sheetinggable': 9999, 'sheetingeave': 9999,
                        'sheetingcorner': 9999, 'sheeting': 75.0}
            for id_type, _type in group.types.iteritems():
                self.assertAlmostEqual(_type.prop_damaged_type,
                                       ref_prop[id_type], places=3)
                self.assertAlmostEqual(_type.damage_capacity,
                                       ref_capacity[id_type], places=1)

    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
