# unit tests
import unittest
import os
import numpy as np
from collections import Counter, OrderedDict

from vaws.config import Config
from vaws.house import House


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = '/'.join(__file__.split('/')[:-1])
        cfg_file = os.path.join(path, '../../scenarios/test_sheeting_batten/test_sheeting_batten.cfg')
        cfg = Config(cfg_file=cfg_file)
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
        for i in xrange(1000):
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
        # self.assertEquals(len(self.house.types), 8)
        self.assertEquals(len(self.house.connections), 60)

        # sheeting
        self.assertEquals(len(self.house.groups['sheeting0'].types['sheetinggable'].connections), 4)
        self.assertEqual(self.house.groups['sheeting0'].types['sheetinggable'].no_connections, 4)

        self.assertEquals(len(self.house.groups['sheeting0'].types['sheetingeave'].connections), 8)
        self.assertEqual(self.house.groups['sheeting0'].types['sheetingeave'].no_connections, 8)

        self.assertEquals(len(self.house.groups['sheeting0'].types['sheetingcorner'].connections), 2)
        self.assertEqual(self.house.groups['sheeting0'].types['sheetingcorner'].no_connections, 2)

        self.assertEquals(len(self.house.groups['sheeting0'].types['sheeting'].connections), 16)
        self.assertEqual(self.house.groups['sheeting0'].types['sheeting'].no_connections, 16)

        self.assertEqual(self.house.groups['sheeting0'].no_connections, 30)

        # batten
        self.assertEquals(len(self.house.groups['batten0'].types['batten'].connections), 12)
        self.assertEqual(self.house.groups['batten0'].types['batten'].no_connections, 12)

        self.assertEquals(len(self.house.groups['batten0'].types['battenend'].connections), 8)
        self.assertEqual(self.house.groups['batten0'].types['battenend'].no_connections, 8)

        self.assertEquals(len(self.house.groups['batten0'].types['batteneave'].connections), 6)
        self.assertEqual(self.house.groups['batten0'].types['batteneave'].no_connections, 6)

        self.assertEquals(len(self.house.groups['batten0'].types['battencorner'].connections), 4)
        self.assertEqual(self.house.groups['batten0'].types['battencorner'].no_connections, 4)

        self.assertEqual(self.house.groups['batten0'].no_connections, 30)

        # costing area by group
        self.assertAlmostEqual(self.house.groups['sheeting0'].costing_area,
                               18.27, places=2)

        self.assertAlmostEqual(self.house.groups['batten0'].costing_area,
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
                              self.house.zones[_conn.zone_loc].name)

        zone_loc_to_grid_map = {
            'A1': (0, 0), 'A2': (0, 1), 'A3': (0, 2), 'A4': (0, 3), 'A5': (0, 4), 'A6': (0, 5),
            'B1': (1, 0), 'B2': (1, 1), 'B3': (1, 2), 'B4': (1, 3), 'B5': (1, 4), 'B6': (1, 5),
            'C1': (2, 0), 'C2': (2, 1), 'C3': (2, 2), 'C4': (2, 3), 'C5': (2, 4), 'C6': (2, 5),
            'D1': (3, 0), 'D2': (3, 1), 'D3': (3, 2), 'D4': (3, 3), 'D5': (3, 4), 'D6': (3, 5),
            'E1': (4, 0), 'E2': (4, 1), 'E3': (4, 2), 'E4': (4, 3), 'E5': (4, 4), 'E6': (4, 5)}

        for _grid, _zone in self.house.zone_by_grid.iteritems():
            self.assertEqual(_grid, zone_loc_to_grid_map[_zone.name])
            self.assertEqual(_grid, _zone.grid)

        for _name, _zone in self.house.zones.iteritems():
            self.assertEqual(_name, _zone.name)

        # influence zone
        for id_conn, _conn in self.house.connections.iteritems():

            if _conn.group_name == 'sheeting':
                for _inf in _conn.influences.itervalues():
                    self.assertEqual(_inf.name, _inf.source.name)
                    self.assertEqual(self.house.zones[_inf.name], _inf.source)
            else:
                for _inf in _conn.influences.itervalues():
                    self.assertEqual(_inf.name, _inf.source.name)
                    self.assertEqual(self.house.connections[_inf.name], _inf.source)

        # group.primary_dir and secondary_dir
        self.assertEqual(self.house.groups['sheeting0'].dist_dir, 'col')
        self.assertEqual(self.house.groups['batten0'].dist_dir, 'row')

        self.assertEqual(self.house.groups['sheeting0'].dist_order, 1)

        # check identity
        # for id_type, _type in self.house.groups['sheeting'].types.iteritems():
        #     self.assertEqual(self.house.types[id_type], _type)

        # for id_conn, _conn in self.house.types['sheeting0'].connections.iteritems():
        #     self.assertEqual(self.house.connections[id_conn], _conn)

    # def test_factors_costing(self):
    #
    #     cfg = Config(
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

        _groups = {x.name for x in self.house.groups.itervalues()}
        # _types = {x.name for x in self.house.types.itervalues()}
        _conns = {x.name for x in self.house.connections.itervalues()}

        self.assertEqual({'sheeting', 'batten'}, _groups)
        #self.assertEqual({'sheetinggable', 'sheetingeave', 'sheetingcorner',
        #                  'sheeting', 'batten', 'battenend', 'batteneave',
        #                  'battencorner'}, _types)
        self.assertEqual(set(range(1, 61)), _conns)

if __name__ == '__main__':
    unittest.main()

# suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
# unittest.TextTestRunner(verbosity=2).run(suite)
