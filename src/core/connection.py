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
        self.edge = dic_conn['edge']

        self.zone_id = dic_conn['zone_id']  # zone location

        self.type_id = dic_ctype['id']
        self.type_name = dic_ctype['connection_type']

        self.group_id = dic_ctype['grouping_id']
        self.group_name = dic_ctype['group_name']

        self.strength_mean = dic_ctype['strength_mean']
        self.strength_std = dic_ctype['strength_std_dev']
        self.dead_load_mean = dic_ctype['deadload_mean']
        self.dead_load_std = dic_ctype['deadload_std_dev']

        self.inf_zones = dict()
        for item in inst_conn.zones:
            self.inf_zones[item.zone_id] = Influence(item)

        self.strength = None
        self.dead_load = None

        # self.failure_v = 0.0  # FIXME!!
        # self.failure_v_i = 0  # FIXME!!
        self.failure_v_raw = 9999  # dummy value

        self.damaged = False
        # self.damaged_by_dist = None

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

    def cal_load(self, use_struct_pz):
        """

        Args:

            use_struct_pz: True or False

        Returns: load

        """

        self.load = 0.0

        if not self.damaged:

            for _inf in self.inf_zones.itervalues():
                # if _inf.zone.effective_area > 0.0:
                try:
                    temp = _inf.coeff * _inf.zone.effective_area
                except TypeError:
                    temp = 0.0

                    logging.debug(
                        'zone {} at {} has {}'.format(
                            _inf.zone.name, _inf.zone.grid,
                            _inf.zone.effective_area))

                if use_struct_pz:
                    self.load += temp * _inf.zone.pz_struct
                else:
                    self.load += temp * _inf.zone.pz

            self.load += self.dead_load

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

    def check_damage(self, wind_speed):
        """
        Args:
            wind_speed: wind speed

        Returns:

        """

        # hard coded for optimization
        use_struct_pz = self.name in self.use_struct_pz_for

        if self.dist_dir:

            for _type in self.types.itervalues():

                for _conn in _type.connections.itervalues():

                    _conn.cal_load(use_struct_pz)

                    # if load is negative, check failure
                    if _conn.load < -1.0 * _conn.strength:

                        _conn.set_damage(wind_speed)

                        logging.debug(
                            'conn #{} of {} at {} damaged at {:.3f}'.format(
                                _conn.name, self.name, _conn.grid, wind_speed))

                        self.damage_grid[_conn.grid] = True

                # summary by connection type
                _type.damage_summary()


class Influence(object):
    def __init__(self, inst):
        """

        Args:
            inst: instance of database.Influence
        """



        dic_ = copy.deepcopy(inst.__dict__)
        self.coeff = dic_['coeff']
        self.conn_id = dic_['connection_id']
        self.zone_id = dic_['zone_id']
        self._zone = None

    @property
    def zone(self):
        return self._zone

    @zone.setter
    def zone(self, inst_zone):
        """

        Args:
            inst_zone: instance of Zone class

        Returns:

        """
        assert isinstance(inst_zone, Zone)
        self._zone = inst_zone

# unit tests
if __name__ == '__main__':
    import unittest
    from scenario import Scenario
    from house import House

    class MyTestCase(unittest.TestCase):

        @classmethod
        def setUpClass(cls):

            cls.cfg = Scenario(cfg_file='../scenarios/test_roof_sheeting2.cfg')

        def test_cal_prop_damaged(self):

            rnd_state = np.random.RandomState(1)
            house = House(self.cfg, rnd_state=rnd_state)
            group = house.groups[1]

            for _conn in house.connections.itervalues():
                self.assertEqual(_conn.damaged, False)

            for _id in [1, 4, 5, 7, 8, 11, 12]:
                house.connections[_id].set_damage(20.0)

            ref_dic = {1: 4, 2: 4, 3: 2, 4: 8}
            ref_area = {1: 0.405, 2: 0.405, 3: 0.225, 4: 0.81}
            for id_type, _type in group.types.iteritems():
                self.assertEqual(_type.no_connections, ref_dic[id_type])
                self.assertEqual(_type.costing_area, ref_area[id_type])

            # costing area by group
            self.assertAlmostEqual(group.costing_area, 10.17, places=2)

            ref_dic = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.25}
            for id_type, _type in group.types.iteritems():
                _type.damage_summary()
                self.assertEqual(_type.prop_damaged_type, ref_dic[id_type])

            group.cal_prop_damaged()
            self.assertAlmostEqual(group.prop_damaged_group, 0.3889, places=4)
            self.assertAlmostEqual(group.prop_damaged_area, 0.3407, places=4)

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
            for id_type, _type in house.types.iteritems():
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

            _conn.cal_load(use_struct_pz=False)
            self.assertAlmostEqual(_conn.load, -0.01032, places=4)

        def test_check_damage1(self):

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

            ref_dic = {x: False for x in range(1, 19)}
            ref_dic[10] = True
            for id_conn, _conn in house.connections.iteritems():
                _conn.cal_load(use_struct_pz=False)
                self.assertEqual(_conn.damaged, ref_dic[id_conn])

            ref_prop = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.125}
            ref_capacity = {1: 9999, 2: 9999, 3: 9999, 4: 50.0}
            for id_type, _type in group.types.iteritems():
                self.assertAlmostEqual(_type.prop_damaged_type,
                                       ref_prop[id_type], places=3)
                self.assertAlmostEqual(_type.damage_capacity,
                                       ref_capacity[id_type], places=1)

    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
