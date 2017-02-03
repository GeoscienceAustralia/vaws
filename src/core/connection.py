"""
    Connection Module - reference storage Connections
        - loaded from database
        - imported from '../data/houses/subfolder'
"""

import copy
import numpy as np
import logging

from zone import Zone
from damage_costing import Costing
from stats import compute_logarithmic_mean_stddev, \
    compute_arithmetic_mean_stddev, sample_lognormal


class Connection(object):
    def __init__(self, inst_conn, dic_ctype):
        """

        Args:
            inst_conn: instance of database.Connection
            dic_ctype: dictionary of database.ConnectionType

        """

        dic_conn = copy.deepcopy(inst_conn.__dict__)

        self.id = dic_conn['id']
        self.name = dic_conn['connection_name']
        # self.edge = dic_conn['edge']

        self.zone_id = dic_conn['zone_id']  # zone location

        self.type_id = dic_ctype['id']
        self.type_name = dic_ctype['connection_type']

        self.group_id = dic_ctype['grouping_id']
        self.group_name = dic_ctype['group_name']

        self.strength_mean = dic_ctype['strength_mean']
        self.strength_std = dic_ctype['strength_std_dev']
        self.dead_load_mean = dic_ctype['deadload_mean']
        self.dead_load_std = dic_ctype['deadload_std_dev']

        self.influences = dict()
        for item in inst_conn.influences:
            dic_ = copy.deepcopy(item.__dict__)
            self.influences.setdefault(dic_['id'], Influence(dic_))

        self.strength = None
        self.dead_load = None

        # self.failure_v = 0.0  # FIXME!!
        # self.failure_v_i = 0  # FIXME!!
        self.failure_v_raw = 9999  # dummy value

        self.damaged = False
        self.distributed = None

        self.load = None

        self._grid = None  # zero-based col, row index

        # self._dist_by_col = None

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, _tuple):
        assert isinstance(_tuple, tuple)
        self._grid = _tuple

    # def reset_connection_failure(self):
    #     self.result_failure_v = 0.0
    #     self.result_failure_v_i = 0

    # @property
    # def dist_by_col(self):
    #     return self._dist_by_col
    #
    # @dist_by_col.setter
    # def dist_by_col(self, value):
    #     assert isinstance(value, bool)
    #     self._dist_by_col = value

    def sample_strength(self, mean_factor, cov_factor, rnd_state):
        """

        Args:
            mean_factor: factor adjusting arithmetic mean strength
            cov_factor: factor adjusting arithmetic cov
            rnd_state:

        Returns: sample of strength following log normal dist.

        """
        mu, std = compute_arithmetic_mean_stddev(self.strength_mean,
                                                 self.strength_std)
        mu *= mean_factor
        std *= cov_factor

        mu_lnx, std_lnx = compute_logarithmic_mean_stddev(mu, std)
        self.strength = sample_lognormal(mu_lnx, std_lnx, rnd_state)

    def sample_dead_load(self, rnd_state):
        """

        Args:
            rnd_state:

        Returns: sample of dead load following log normal dist.

        """
        self.dead_load = sample_lognormal(self.dead_load_mean,
                                          self.dead_load_std,
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
                self.influences.update({_id: Influence({'coeff': updated_coeff,
                                                        'id': _id})})
                self.influences[_id].source = _infl.source

        # logging.debug('influences of {}:{:.2f}'.format(self.name,
        #                                                _inf.coeff))


class ConnectionType(object):
    def __init__(self, inst):
        """

        Args:
            inst: instance of database.ConnectionType
        """
        dic_ = copy.deepcopy(inst.__dict__)

        self.id = dic_['id']
        self.name = dic_['connection_type']
        self.costing_area = dic_['costing_area']

        self.strength_mean = dic_['strength_mean']
        self.strength_std = dic_['strength_std_dev']
        self.dead_load_mean = dic_['deadload_mean']
        self.dead_load_std = dic_['deadload_std_dev']
        self.group_id = dic_['grouping_id']
        self.group_name = inst.group.group_name
        dic_.setdefault('group_name', self.group_name)

        self.connections = dict()
        for item in inst.connections_of_type:
            self.connections.setdefault(item.id, Connection(item, dic_))

        self.no_connections = len(self.connections)
        self.damage_capacity = None  # min wind speed at damage
        self.prop_damaged_type = None

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

    def __init__(self, inst):
        """

        Args:
            inst: instance of database.ConnectionTypeGroup
        """
        dic_ = copy.deepcopy(inst.__dict__)

        self.id = dic_['id']
        self.name = dic_['group_name']

        self.dist_dir = dic_['distribution_direction']
        # self.dist_ord = dic_['distribution_order']

        self.trigger_collapse_at = dic_['trigger_collapse_at']
        self.patch_dist = dic_['patch_distribution']
        self.set_zone_to_zero = dic_['set_zone_to_zero']
        self.water_ingress_ord = dic_['water_ingress_order']
        self.costing_id = dic_['costing_id']
        self.costing = Costing(inst.costing)

        self._costing_area = None
        self.no_connections = 0

        self.types = dict()
        for item in inst.conn_types:
            _type = ConnectionType(item)
            self.types.setdefault(item.id, _type)
            self.no_connections += _type.no_connections

        # if self.dist_ord >= 0:
        #     self.enabled = True
        # else:
        #     self.enabled = False

        # self.primary_dir = None
        # self.secondary_dir = None
        # self.dist_by_col = None

        self.damage_grid = None  # column (chr), row (num)
        self.damaged = None

        self.conn_by_grid = dict()  # dict of connections with zone loc grid

        # self._dist_tuple = None

        self.prop_damaged_group = None
        self.prop_damaged_area = 0.0
        self.repair_cost = None

    @property
    def costing_area(self):
        return self._costing_area

    @costing_area.setter
    def costing_area(self, value):
        self._costing_area = value

    def set_damage_grid(self, no_rows, no_cols):
        """

        Args:
            no_rows: no. of rows
            no_cols: no. of cols

        Returns:

        """
        if self.dist_dir:
            self.damage_grid = np.zeros(dtype=bool, shape=(no_rows, no_cols))
        else:
            self.damage_grid = None

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

        self.damaged = []

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

                    self.damage_grid[_conn.grid] = True

            # summary by connection type
            _type.damage_summary()

    def distribute_damage(self):

        if self.dist_dir == 'row':  # batten
            # distributed over the same number

            # row: index of chr, col: index of number e.g, 0,0 : A1
            # for row, col in zip(*np.where(self.damage_grid)):
            for row, col0 in zip(*np.where(self.damage_grid[:, self.damaged])):

                col = self.damaged[col0]

                intact = np.where(~self.damage_grid[:, col])[0]

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

            for row0, col in zip(*np.where(self.damage_grid[self.damaged, :])):

                row = self.damaged[row0]

                # # looking at influences
                # linked_conn = None
                # for val in source_conn.influences.itervalues():
                #     if val.coeff == 1.0:
                #         linked_conn = val.source

                intact = np.where(~self.damage_grid[row, :])[0]

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
    def __init__(self, dic_):
        """

        Args:
            dic_: dictionary coeff, id, source
        """

        # dic_ = copy.deepcopy(inst.__dict__)
        self.coeff = dic_['coeff']
        self.id = dic_['id']  # source connection or zone id
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
    from scenario import Scenario
    from house import House

    class MyTestCase(unittest.TestCase):

        @classmethod
        def setUpClass(cls):

            cls.cfg = Scenario(cfg_file='../scenarios/test_sheeting_batten.cfg')

        def test_cal_prop_damaged(self):

            rnd_state = np.random.RandomState(1)
            house = House(self.cfg, rnd_state=rnd_state)
            group = house.groups[1]  # sheeting

            for _conn in house.connections.itervalues():
                self.assertEqual(_conn.damaged, False)

            # 1: sheeting gable(1): 2 - 5
            # 2: sheeting eave(2): 7, 12, 13, 18, 19, 24, 25, 30
            # 3: sheeting corner(3): 1, 6
            # 4: sheeting(4): 8 - 11, 14 - 17, 20 - 23, 26 - 29
            for _id in [1, 4, 5, 7, 8, 11, 12]:
                house.connections[_id].set_damage(20.0)

            ref_dic = {1: 4, 2: 8, 3: 2, 4: 16}
            ref_area = {1: 0.405, 2: 0.405, 3: 0.225, 4: 0.81}
            for id_type, _type in group.types.iteritems():
                self.assertEqual(_type.no_connections, ref_dic[id_type])
                self.assertEqual(_type.costing_area, ref_area[id_type])

            # costing area by group
            self.assertAlmostEqual(group.costing_area, 18.27, places=2)

            ref_dic = {1: 0.5, 2: 0.25, 3: 0.5, 4: 0.125}
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

            ref_dic = {1: 20.0, 2: 30.0, 3: 45.0, 4: 9999.0}
            for id_type, _type in house.groups[1].types.iteritems():
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

            for _zone in house.zones.itervalues():
                _zone.calc_zone_pressures(wind_dir_index,
                                          cpi,
                                          qz,
                                          Ms,
                                          building_spacing)

            _conn = house.connections[1]

            # init
            self.assertEqual(_conn.damaged, False)
            self.assertEqual(_conn.load, None)

            _conn.cal_load()

            self.assertAlmostEqual(_conn.dead_load, 0.0130, places=4)
            self.assertAlmostEqual(_conn.influences[1].source.area,
                                   0.2025, places=4)
            self.assertAlmostEqual(_conn.influences[1].source.pz, -0.05651,
                                   places=4)

            self.assertAlmostEqual(_conn.load, 0.00158, places=4)

        def test_check_damage(self):

            rnd_state = np.random.RandomState(1)
            house = House(self.cfg, rnd_state)

            # compute zone pressures
            wind_speed = 50.0
            mzcat = 0.9235
            wind_dir_index = 3
            cpi = 0.0
            qz = 0.6 * 1.0e-3 * (wind_speed * mzcat) ** 2
            Ms = 1.0
            building_spacing = 0

            for _zone in house.zones.itervalues():
                _zone.calc_zone_pressures(wind_dir_index,
                                          cpi,
                                          qz,
                                          Ms,
                                          building_spacing)

            group = house.groups[1]
            group.check_damage(wind_speed=wind_speed)

            ref_dic = {x: False for x in range(1, 61)}
            ref_dic[10] = True
            for id_conn, _conn in house.connections.iteritems():
                _conn.cal_load()
                self.assertEqual(_conn.damaged, ref_dic[id_conn])

            ref_prop = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0625}
            ref_capacity = {1: 9999, 2: 9999, 3: 9999, 4: 50.0}
            for id_type, _type in group.types.iteritems():
                self.assertAlmostEqual(_type.prop_damaged_type,
                                       ref_prop[id_type], places=3)
                self.assertAlmostEqual(_type.damage_capacity,
                                       ref_capacity[id_type], places=1)


    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
