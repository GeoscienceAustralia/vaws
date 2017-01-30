"""
    House Module - reference storage for House type information.
        - loaded from database
        - imported from '../data/houses/subfolder' (so we need python constr)
"""

import numpy as np

from connection import ConnectionTypeGroup
from zone import Zone
from database import House as TableHouse
from database import DatabaseManager
from stats import calc_big_a_b_values
from scenario import Scenario


class House(object):

    def __init__(self, cfg, rnd_state):

        assert isinstance(cfg, Scenario)
        assert isinstance(rnd_state, np.random.RandomState)

        self.cfg = cfg
        self.rnd_state = rnd_state

        _db = DatabaseManager(cfg.db_file)

        db_house = _db.session.query(TableHouse).filter_by(
            house_name=cfg.house_name).one()

        self.width = db_house.__dict__['width']
        self.height = db_house.__dict__['height']
        self.length = db_house.__dict__['length']
        self.replace_cost = db_house.__dict__['replace_cost']

        self.cpe_cov = db_house.__dict__['cpe_V']
        self.cpe_str_cov = db_house.__dict__['cpe_struct_V']
        self.cpe_k = db_house.__dict__['cpe_k']

        self.roof_rows = db_house.__dict__['roof_rows']
        self.roof_cols = db_house.__dict__['roof_columns']

        self.wind_orientation = None  # 0 to 7
        self.construction_level = None
        self.profile = None
        self.str_mean_factor = None
        self.str_cov_factor = None
        self.mzcat = None
        self.big_a = None
        self.big_b = None

        self.cols = [chr(x) for x in range(ord('A'), ord('A') + self.roof_cols)]
        self.rows = range(1, self.roof_rows + 1)  # one base

        self.groups = dict()  # dict of conn type groups with id
        self.types = dict()  # dict of conn types with id
        self.connections = dict()  # dict of connections with id
        self.zones = dict()  # dict of zones with id
        self.factors_costing = dict()  # dict of damage factoring with id

        self.zone_by_name = dict()  # dict of zones with zone name
        self.zone_by_grid = dict()  # dict of zones with zone loc grid in tuple

        # init house
        self.set_house_wind_params()
        self.set_house_components(db_house)

        # db session close
        _db.close()

    def set_house_components(self, db_house):
        # house is consisting of connections and zones

        for item in db_house.zones:
            _zone = Zone(item)
            _zone.sample_zone_pressure(
                wind_dir_index=self.wind_orientation,
                cpe_cov=self.cpe_cov,
                cpe_k=self.cpe_k,
                cpe_str_cov=self.cpe_str_cov,
                big_a=self.big_a,
                big_b=self.big_b,
                rnd_state=self.rnd_state)

            self.zones.setdefault(item.id, _zone)
            self.zone_by_name.setdefault(_zone.name, _zone)
            self.zone_by_grid.setdefault(_zone.grid, _zone)

        for item in db_house.conn_type_groups:
            _group = ConnectionTypeGroup(item)
            costing_area_by_group = 0.0

            for id_type, _type in _group.types.iteritems():

                self.types.setdefault(id_type, _type)

                for id_conn, _conn in _type.connections.iteritems():

                    _conn.sample_strength(mean_factor=self.str_mean_factor,
                                          cov_factor=self.str_cov_factor,
                                          rnd_state=self.rnd_state)

                    _conn.sample_dead_load(rnd_state=self.rnd_state)

                    _conn.grid = self.zones[_conn.zone_id].grid

                    costing_area_by_group += _type.costing_area

                    # linking connections either zones or connections
                    if _conn.group_name == 'sheeting':
                        for _inf in _conn.influences:
                            _inf.source = self.zones[_inf.id]
                    else:
                        for _inf in _conn.influences:
                            _inf.source = self.connections[_inf.id]

                    self.connections.setdefault(id_conn, _conn)
                    _group.conn_by_grid.setdefault(_conn.grid, _conn)

            _group.costing_area = costing_area_by_group
            _group.set_damage_grid(self.roof_cols, self.roof_rows)

            self.groups.setdefault(item.id, _group)

        # factors to avoid double counting in computing repair cost
        for item in db_house.factorings:
            self.factors_costing.setdefault(
                item.parent_id, []).append(item.factor_id)

        # linking zone loc grid to connection by group
        # for _zone_id, _zone in self.zones.iteritems():
        #     for _conn in self.connections.itervalues():
        #         if _conn.zone_id == _zone_id:
        #             self.conn_by_grid.setdefault(_zone.grid, {})[_conn.group_id] = _conn

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
        _wind_profile = self.cfg.wind_profiles[self.cfg.terrain_category][self.profile]
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



# unit tests
if __name__ == '__main__':
    import unittest
    from collections import Counter, OrderedDict

    class MyTestCase(unittest.TestCase):

        @classmethod
        def setUpClass(cls):

            cfg = Scenario(cfg_file='../scenarios/test_sheeting_batten.cfg')
            rnd_state = np.random.RandomState(1)
            cls.house = House(cfg, rnd_state)

        def test_set_house_wind_params(self):

            assert self.house.cpe_k == 0.1
            self.assertAlmostEqual(self.house.big_a, 0.48649, places=4)
            self.assertAlmostEqual(self.house.big_b, 1.14457, places=4)

            assert self.house.wind_orientation == 3

            if self.house.construction_level == 'low':
                self.assertAlmostEqual(self.house.str_mean_factor, 0.9)
            elif self.house.construction_level == 'medium':
                self.assertAlmostEqual(self.house.str_mean_factor, 1.0)
            elif self.house.construction_level == 'high':
                self.assertAlmostEqual(self.house.str_mean_factor, 1.1)

            self.assertAlmostEqual(self.house.str_cov_factor, 0.58)

        def test_set_construction_levels(self):
            self.house.cfg.construction_levels = OrderedDict(
                [('low', {'cov_factor': 0.58,
                          'mean_factor': 0.9,
                          'probability': 0.3}),
                 ('medium', {'cov_factor': 0.58,
                             'mean_factor': 1.0,
                             'probability': 0.6}),
                 ('high', {'cov_factor': 0.58,
                           'mean_factor': 1.1,
                           'probability': 0.1})])
            tmp = []
            for i in range(1000):
                self.house.set_construction_level()
                tmp.append(self.house.construction_level)

            counts = Counter(tmp)
            self.assertAlmostEqual(counts['low']*0.001, 0.30, places=1)
            self.assertAlmostEqual(counts['medium']*0.001, 0.60, places=1)
            self.assertAlmostEqual(counts['high']*0.001, 0.10, places=1)

        def test_set_wind_profile(self):
            self.assertAlmostEqual(self.house.height, 4.5, places=1)
            self.assertEquals(self.house.cfg.terrain_category, '2')

            # copied from mzcat_terrain_2.csv
            data = np.array([[3, 0.908, 0.896, 0.894, 0.933, 0.884, 0.903, 0.886, 0.902, 0.859, 0.927],
            [5, 0.995, 0.980, 0.946, 0.986, 0.962, 1.010, 0.978, 0.970, 0.945, 0.990],
            [7, 0.994, 1.031, 1.010, 0.986, 0.982, 0.987, 0.959, 0.984, 0.967, 0.998],
            [10, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
            [12, 1.056, 1.025, 1.032, 1.033, 0.998, 1.043, 0.997, 1.008, 1.005, 1.027],
            [15, 1.058, 1.059, 1.028, 1.069, 1.048, 1.076, 1.016, 1.027, 1.021, 1.039],
            [17, 1.092, 1.059, 1.079, 1.060, 1.042, 1.053, 1.046, 1.045, 1.047, 1.102],
            [20, 1.110, 1.103, 1.037, 1.068, 1.088, 1.107, 1.068, 1.106, 1.098, 1.103],
            [25, 1.145, 1.151, 1.069, 1.091, 1.089, 1.196, 1.126, 1.113, 1.099, 1.142],
            [30, 1.245, 1.188, 1.177, 1.178, 1.192, 1.199, 1.179, 1.165, 1.127, 1.203]])

            heights = data[:, 0]
            value = data[:, self.house.profile]
            _mzcat = np.interp([4.5], heights, value)[0]
            self.assertAlmostEqual(_mzcat, self.house.mzcat, places=4)

        def test_set_house_components(self):

            self.assertEquals(len(self.house.groups), 2)
            self.assertEquals(len(self.house.types), 8)
            self.assertEquals(len(self.house.connections), 60)

            # sheeting
            self.assertEquals(len(self.house.groups[1].types[1].connections), 4)
            self.assertEqual(self.house.groups[1].types[1].no_connections, 4)

            self.assertEquals(len(self.house.groups[1].types[2].connections), 8)
            self.assertEqual(self.house.groups[1].types[2].no_connections, 8)

            self.assertEquals(len(self.house.groups[1].types[3].connections), 2)
            self.assertEqual(self.house.groups[1].types[3].no_connections, 2)

            self.assertEquals(len(self.house.groups[1].types[4].connections), 16)
            self.assertEqual(self.house.groups[1].types[4].no_connections, 16)

            self.assertEqual(self.house.groups[1].no_connections, 30)

            # batten
            self.assertEquals(len(self.house.groups[2].types[5].connections), 12)
            self.assertEqual(self.house.groups[2].types[5].no_connections, 12)

            self.assertEquals(len(self.house.groups[2].types[6].connections), 8)
            self.assertEqual(self.house.groups[2].types[6].no_connections, 8)

            self.assertEquals(len(self.house.groups[2].types[7].connections), 6)
            self.assertEqual(self.house.groups[2].types[7].no_connections, 6)

            self.assertEquals(len(self.house.groups[2].types[8].connections), 4)
            self.assertEqual(self.house.groups[2].types[8].no_connections, 4)

            self.assertEqual(self.house.groups[1].no_connections, 30)

            # costing area by group
            self.assertAlmostEqual(self.house.groups[1].costing_area,
                                   18.27, places=2)

            self.assertAlmostEqual(self.house.groups[2].costing_area,
                                   16.29, places=2)
            # zone
            conn_zone_loc_map = {
                1: 'A1', 2: 'A2', 3: 'A3', 4: 'A4', 5: 'A5', 6: 'A6',
                7: 'B1', 8: 'B2', 9: 'B3', 10: 'B4', 11: 'B5', 12: 'B6',
                13: 'C1', 14: 'C2', 15: 'C3', 16: 'C4', 17: 'C5', 18: 'C6',
                19: 'D1', 20: 'D2', 21: 'D3', 22: 'D4', 23: 'D5', 24: 'D6',
                25: 'E1', 26: 'E2', 27: 'E3', 28: 'E4', 29: 'E5', 30: 'E6',
                31: 'A1', 32: 'A2', 33: 'A3', 34: 'A4', 35: 'A5', 36: 'A6',
                37: 'B1', 38: 'B2', 39: 'B3', 40: 'B4', 41: 'B5', 42: 'B6',
                43: 'C1', 44: 'C2', 45: 'C3', 46: 'C4', 47: 'C5', 48: 'C6',
                49: 'D1', 50: 'D2', 51: 'D3', 52: 'D4', 53: 'D5', 54: 'D6',
                55: 'E1', 56: 'E2', 57: 'E3', 58: 'E4', 59: 'E5', 60: 'E6',
            }

            for id_conn, _conn in self.house.connections.iteritems():

                _zone_name = conn_zone_loc_map[id_conn]

                self.assertEquals(_zone_name,
                                  self.house.zones[_conn.zone_id].name)

            zone_loc_to_grid_map = {
                'A1': (0, 0), 'A2': (0, 1), 'A3': (0, 2), 'A4': (0, 3), 'A5': (0, 4), 'A6': (0, 5),
                'B1': (1, 0), 'B2': (1, 1), 'B3': (1, 2), 'B4': (1, 3), 'B5': (1, 4), 'B6': (1, 5),
                'C1': (2, 0), 'C2': (2, 1), 'C3': (2, 2), 'C4': (2, 3), 'C5': (2, 4), 'C6': (2, 5),
                'D1': (3, 0), 'D2': (3, 1), 'D3': (3, 2), 'D4': (3, 3), 'D5': (3, 4), 'D6': (3, 5),
                'E1': (4, 0), 'E2': (4, 1), 'E3': (4, 2), 'E4': (4, 3), 'E5': (4, 4), 'E6': (4, 5)}

            for _grid, _zone in self.house.zone_by_grid.iteritems():
                self.assertEqual(_grid, zone_loc_to_grid_map[_zone.name])
                self.assertEqual(_grid, _zone.grid)

            for _name, _zone in self.house.zone_by_name.iteritems():
                self.assertEqual(_name, _zone.name)

            # influence zone
            for id_conn, _conn in self.house.connections.iteritems():

                if _conn.group_name == 'sheeting':
                    for _inf in _conn.influences:
                        self.assertEqual(_inf.id, _inf.source.id)
                        self.assertEqual(self.house.zones[_inf.id], _inf.source)
                else:
                    for _inf in _conn.influences:
                        self.assertEqual(_inf.id, _inf.source.id)
                        self.assertEqual(self.house.connections[_inf.id], _inf.source)

            # group.primary_dir and secondary_dir
            self.assertEqual(self.house.groups[1].dist_dir, 'col')
            self.assertEqual(self.house.groups[2].dist_dir, 'row')

            # self.assertEqual(self.house.groups[1].dist_ord, 1)
            # self.assertEqual(self.house.groups[1].dist_by_col, True)
            # self.assertEqual(self.house.groups[1].primary_dir, self.house.cols)
            # self.assertEqual(self.house.groups[1].secondary_dir, self.house.rows)

            # check identity
            for id_type, _type in self.house.groups[1].types.iteritems():
                self.assertEqual(self.house.types[id_type], _type)

            for id_conn, _conn in self.house.types[1].connections.iteritems():
                self.assertEqual(self.house.connections[id_conn], _conn)

        # def test_factors_costing(self):
        #
        #     cfg = Scenario(
        #         cfg_file='../scenarios/carl1_dmg_dist_off_no_wall_no_water.cfg')
        #     rnd_state = np.random.RandomState(1)
        #     house = House(cfg, rnd_state=rnd_state)
        #
        #     ref_dic = {'wallcladding': ['debris', 'wallracking', 'wallcollapse'],
        #                'wallracking': 'wallcollapse',
        #                'debris': ['wallracking', 'wallcollapse'],
        #                'sheeting': ['rafter', 'batten'],
        #                'batten': ['rafter']}
        #
        #     for _id, values in house.factors_costing.iteritems():
        #         for _id_upper in values:
        #
        #             _group_name = house.groups[_id].name
        #             _upper_name = house.groups[_id_upper].name
        #             msg = 'The repair cost of {} is not included in {}'.format(
        #                 _group_name, _upper_name)
        #             self.assertIn(_upper_name, ref_dic[_group_name], msg=msg)

        def test_list_groups_types_conns(self):

            list_groups = [x.name for x in self.house.groups.itervalues()]
            list_types = [x.name for x in self.house.types.itervalues()]
            list_conns = [x.name for x in self.house.connections.itervalues()]

            self.assertEqual(['sheeting', 'batten'], list_groups)
            self.assertEqual(['sheetinggable', 'sheetingeave', 'sheetingcorner',
                              'sheeting', 'batten', 'battenend', 'batteneave',
                              'battencorner'], list_types)
            self.assertEqual([str(x) for x in range(1, 61)], list_conns)

        '''
        # def test_conn_by_grid(self):
        #
        #     cfg = Scenario(cfg_file='../scenarios/carl1_dmg_dist_on_no_wall_no_debris_no_water.cfg')
        #     rnd_state = np.random.RandomState(1)
        #     house = House(cfg, rnd_state)
        #
        #     # zone loc: A1
        #
        #     # sheeting: 4
        #     self.assertEqual(house.conn_by_grid[(0, 0)][4].id, 783)
        #     self.assertEqual(house.conn_by_grid[(0, 0)][4].name, '1')
        #
        #     # batten: 5
        #     self.assertEqual(house.conn_by_grid[(0, 0)][5].id, 963)
        #     self.assertEqual(house.conn_by_grid[(0, 0)][5].name, '181')
        '''
    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
