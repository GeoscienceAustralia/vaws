"""
    Connection Module - reference storage Connections
        - loaded from database
        - imported from '../data/houses/subfolder'
"""

import copy
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

        self.zone_id = dic_conn['zone_id']

        self.type_id = dic_ctype['id']
        self.type_name = dic_ctype['connection_type']

        self.group_id = dic_ctype['grouping_id']
        self.group_name = dic_ctype['group_name']

        self.strength_mean = dic_ctype['strength_mean']
        self.strength_std = dic_ctype['strength_std_dev']
        self.deadload_mean = dic_ctype['deadload_mean']
        self.deadload_std = dic_ctype['deadload_std_dev']

        self.inf_zones = dict()
        for item in inst_conn.zones:
            self.inf_zones[item.zone_id] = Influence(item)

        self.strength = None
        self.deadload = None

        self.result_failure_v = 0.0  # FIXME!!
        self.result_failure_v_i = 0  # FIXME!!
        self.result_failure_v_raw = 9999  # dummy value

        self.result_damaged = False
        self.result_damage_distributed = None

        self.result_load = None

    # def reset_connection_failure(self):
    #     self.result_failure_v = 0.0
    #     self.result_failure_v_i = 0

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

    def sample_deadload(self, rnd_state):
        """

        Args:
            rnd_state:

        Returns: sample of dead load following log normal dist.

        """
        self.deadload = sample_lognormal(self.deadload_mean, self.deadload_std,
                                         rnd_state)

    def cal_load(self, use_struct_pz):
        """

        Args:

            use_struct_pz: True or False

        Returns:

        """

        self.result_load = 0.0
        if not self.result_damaged:

            for _inf in self.inf_zones.itervalues():
                # if _inf.zone.effective_area > 0.0:
                temp = _inf.coeff * _inf.zone.effective_area

                if use_struct_pz:
                    self.result_load += temp * _inf.zone.pz_struct
                else:
                    self.result_load += temp * _inf.zone.pz

            self.result_load += self.deadload

    def set_damage(self, wind_speed):
        self.result_damaged = True
        self.result_damage_distributed = False
        self.result_failure_v_i += 1
        self.result_failure_v_raw = wind_speed
        num_ = wind_speed + (
                            self.result_failure_v_i - 1) * self.result_failure_v
        denom_ = self.result_failure_v_i
        self.result_failure_v = num_ / denom_


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
        self.deadload_mean = dic_['deadload_mean']
        self.deadload_std = dic_['deadload_std_dev']
        self.group_id = dic_['grouping_id']
        self.group_name = inst.group.group_name
        dic_.setdefault('group_name', self.group_name)

        self.connections = dict()
        for item in inst.connections_of_type:
            self.connections.setdefault(item.id, Connection(item, dic_))

        self.no_connections = len(self.connections)
        self.damage_capacity = None  # min wind speed at damage
        self.prop_damaged = None

    def run_damage_summary(self):
        """
        compute proportion of damaged connections and damage capacity
        Returns: prop_damaged, damage_capacity

        """
        num_damaged = 0
        _value = 99999
        for _conn in self.connections.itervalues():
            num_damaged += _conn.result_damaged
            _value = min(_conn.result_failure_v_raw, _value)

        try:
            self.prop_damaged = num_damaged / float(self.no_connections)
        except ZeroDivisionError:
            self.prop_damaged = 0.0

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
        self.dist_ord = dic_['distribution_order']
        self.trigger_collapse_at = dic_['trigger_collapse_at']
        self.patch_dist = dic_['patch_distribution']
        self.set_zone_to_zero = dic_['set_zone_to_zero']
        self.water_ingress_ord = dic_['water_ingress_order']
        self.costing_id = dic_['costing_id']
        self.costing = Costing(inst.costing)
        self.costing_area = 0.0
        self.no_connections = 0

        self.types = dict()
        for item in inst.conn_types:
            _type = ConnectionType(item)
            self.types.setdefault(item.id, _type)
            self.no_connections += _type.no_connections

        if self.dist_ord >= 0:
            self.enabled = True
        else:
            self.enabled = False

        self.primary_dir = None
        self.secondary_dir = None
        self.dist_by_col = None

        self._dist_tuple = None

        self.damage_ratio = None

        self.prop_damaged = None

    def cal_prop_damaged(self):

        num_damaged = 0
        for _type in self.types.itervalues():
            for _conn in _type.connections.itervalues():
                num_damaged += _conn.result_damaged

        try:
            self.prop_damaged = num_damaged / float(self.no_connections)
        except ZeroDivisionError:
            self.prop_damaged = 0.0

    @property
    def dist_tuple(self):
        return self._dist_tuple

    @dist_tuple.setter
    def dist_tuple(self, _tuple):
        """

        Args:
            _tuple: rows, cols

        Returns:

        """
        try:
            _rows, _cols = _tuple
        except ValueError:
            raise ValueError("Pass an iterable with two items")
        else:
            if self.dist_dir == 'col':
                self.dist_by_col = True
                self.primary_dir = _cols
                self.secondary_dir = _rows
            else:
                self.dist_by_col = False
                self.primary_dir = _rows
                self.secondary_dir = _cols

    def check_damage(self, wind_speed):
        """
        Args:
            wind_speed: wind speed

        Returns:

        """

        # hard coded for optimization
        use_struct_pz = self.name in self.use_struct_pz_for

        if self.enabled:

            for _type in self.types.itervalues():

                for _conn in _type.connections.itervalues():

                    _conn.cal_load(use_struct_pz)

                    # if load is negative, check failure
                    if _conn.result_load < -1.0 * _conn.strength:

                        _conn.set_damage(wind_speed)

                # summary by connection type
                _type.run_damage_summary()


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

            house = House(self.cfg)
            group = house.groups[1]

            for _conn in house.connections.itervalues():
                self.assertEqual(_conn.result_damaged, False)

            for _id in [1, 4, 5, 7, 8, 11, 12]:
                house.connections[_id].set_damage(20.0)

            ref_dic = {1: 4, 2: 4, 3: 2, 4: 8}
            for id_type, _type in group.types.iteritems():
                self.assertEqual(_type.no_connections, ref_dic[id_type])

            ref_dic = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.25}
            for id_type, _type in group.types.iteritems():
                _type.run_damage_summary()
                self.assertEqual(_type.prop_damaged, ref_dic[id_type])

            group.cal_prop_damaged()
            self.assertAlmostEqual(group.prop_damaged, 0.3889, places=4)

        def test_cal_damage_capacity(self):

            house = House(self.cfg)

            # type 1
            house.connections[2].set_damage(20.0)
            house.connections[5].set_damage(30.0)

            # type 2
            house.connections[7].set_damage(30.0)

            # type 3
            house.connections[1].set_damage(45.0)

            ref_dic = {1: 20.0, 2: 30.0, 3: 45.0, 4: 9999.0}
            for id_type, _type in house.types.iteritems():
                _type.run_damage_summary()
                self.assertAlmostEqual(_type.damage_capacity, ref_dic[id_type])

        def test_cal_load(self):

            house = House(self.cfg)

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
            self.assertEqual(_conn.result_damaged, False)
            self.assertEqual(_conn.result_load, None)

            _conn.cal_load(use_struct_pz=False)
            self.assertAlmostEqual(_conn.result_load, 0.01736, places=4)

        def test_check_damage1(self):

            house = House(self.cfg)

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

            house = House(self.cfg)

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
                self.assertEqual(_conn.result_damaged, ref_dic[id_conn])

            ref_prop = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.125}
            ref_capacity = {1: 9999, 2: 9999, 3: 9999, 4: 50.0}
            for id_type, _type in group.types.iteritems():
                self.assertAlmostEqual(_type.prop_damaged,
                                       ref_prop[id_type], places=3)
                self.assertAlmostEqual(_type.damage_capacity,
                                       ref_capacity[id_type], places=1)

    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
