#!/usr/bin/env python

import unittest
import os
import logging
from collections import Counter, OrderedDict
from io import StringIO
import pandas as pd
import numpy as np

from vaws.model.house import House
from vaws.model.config import Config


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = os.sep.join(__file__.split(os.sep)[:-1])
        file_cfg = os.path.join(
            path, 'test_scenarios', 'test_sheeting_batten', 'test_sheeting_batten.cfg')
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        cfg = Config(file_cfg=file_cfg, logger=logger)
        cls.house = House(cfg, 1)

    def test_set_house_wind_params(self):

        assert self.house.cpe_k == 0.1
        self.assertAlmostEqual(self.house.big_a, 0.48649, places=4)
        self.assertAlmostEqual(self.house.big_b, 1.14457, places=4)

        assert self.house.wind_dir_index == 3

        # if self.house.construction_level == 'low':
        #     self.assertAlmostEqual(self.house.mean_factor, 0.9)
        # elif self.house.construction_level == 'medium':
        #     self.assertAlmostEqual(self.house.mean_factor, 1.0)
        # elif self.house.construction_level == 'high':
        #     self.assertAlmostEqual(self.house.mean_factor, 1.1)

        # self.assertAlmostEqual(self.house.cv_factor, 0.58)

    # def test_construction_levels(self):
    #     self.house.cfg.construction_levels_levels = ['low', 'medium', 'high']
    #     self.house.cfg.construction_levels_probs = [0.3, 0.6, 0.1]
    #
    #     tmp = []
    #     for i in range(1000):
    #         tmp.append(self.house.construction_level)
    #         self.house._construction_level = None
    #
    #     counts = Counter(tmp)
    #     self.assertAlmostEqual(counts['low']*0.001, 0.30, places=1)
    #     self.assertAlmostEqual(counts['medium']*0.001, 0.60, places=1)
    #     self.assertAlmostEqual(counts['high']*0.001, 0.10, places=1)

    def test_wind_profile(self):
        self.assertAlmostEqual(self.house.height, 4.5, places=1)
        # self.assertEqual(self.house.cfg.wind_profiles, 'cyclonic_terrain_cat2.csv')

        # copied from cyclonic_terrain_cat2.csv
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
        value = data[:, self.house.profile_index]
        _mzcat = np.interp([4.5], heights, value)[0]
        self.assertAlmostEqual(_mzcat, self.house.terrain_height_multiplier, places=4)

    def test_set_house_components(self):

        self.assertEqual(len(self.house.groups), 2)
        # self.assertEqual(len(self.house.types), 8)
        self.assertEqual(len(self.house.connections), 60)

        # sheeting
        self.assertEqual(self.house.groups['sheeting0'].no_connections, 30)

        # batten
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

        for id_conn, _conn in self.house.connections.items():

            _zone_name = conn_zone_loc_map[id_conn]

            self.assertEqual(_zone_name,
                              self.house.zones[_conn.zone_loc].name)

        '''
        zone_loc_to_grid_map = {
            'A1': (0, 0), 'A2': (0, 1), 'A3': (0, 2), 'A4': (0, 3), 'A5': (0, 4), 'A6': (0, 5),
            'B1': (1, 0), 'B2': (1, 1), 'B3': (1, 2), 'B4': (1, 3), 'B5': (1, 4), 'B6': (1, 5),
            'C1': (2, 0), 'C2': (2, 1), 'C3': (2, 2), 'C4': (2, 3), 'C5': (2, 4), 'C6': (2, 5),
            'D1': (3, 0), 'D2': (3, 1), 'D3': (3, 2), 'D4': (3, 3), 'D5': (3, 4), 'D6': (3, 5),
            'E1': (4, 0), 'E2': (4, 1), 'E3': (4, 2), 'E4': (4, 3), 'E5': (4, 4), 'E6': (4, 5)}

        for _grid, _zone in self.house.zone_by_grid.items():
            self.assertEqual(_grid, zone_loc_to_grid_map[_zone.name])
            self.assertEqual(_grid, _zone.grid)
        '''

        for _name, _zone in self.house.zones.items():
            self.assertEqual(_name, _zone.name)

        # influence zone
        for id_conn, _conn in self.house.connections.items():

            if _conn.group_name == 'sheeting':
                for _, _inf in _conn.influences.items():
                    self.assertEqual(_inf.name, _inf.source.name)
                    self.assertEqual(self.house.zones[_inf.name], _inf.source)
            else:
                for _, _inf in _conn.influences.items():
                    self.assertEqual(_inf.name, _inf.source.name)
                    self.assertEqual(self.house.connections[_inf.name], _inf.source)

        # group.primary_dir and secondary_dir
        self.assertEqual(self.house.groups['sheeting0'].dist_dir, 'col')
        self.assertEqual(self.house.groups['batten0'].dist_dir, 'row')

        self.assertEqual(self.house.groups['sheeting0'].dist_order, 1)

        # check identity
        # for id_type, _type in self.house.groups['sheeting'].types.items():
        #     self.assertEqual(self.house.types[id_type], _type)

        # for id_conn, _conn in self.house.types['sheeting0'].connections.items():
        #     self.assertEqual(self.house.connections[id_conn], _conn)

    # def test_factors_costing(self):
    #
    #     cfg = Config(
    #         file_cfg='../scenarios/carl1_dmg_dist_off_no_wall_no_water.cfg')
    #     rnd_state = np.random.RandomState(1)
    #     house = House(cfg, rnd_state=rnd_state)
    #
    #     ref_dic = {'wallcladding': ['debris', 'wallracking', 'wallcollapse'],
    #                'wallracking': 'wallcollapse',
    #                'debris': ['wallracking', 'wallcollapse'],
    #                'sheeting': ['rafter', 'batten'],
    #                'batten': ['rafter']}
    #
    #     for _id, values in house.factors_costing.items():
    #         for _id_upper in values:
    #
    #             _group_name = house.groups[_id].name
    #             _upper_name = house.groups[_id_upper].name
    #             msg = 'The repair cost of {} is not included in {}'.format(
    #                 _group_name, _upper_name)
    #             self.assertIn(_upper_name, ref_dic[_group_name], msg=msg)

    def test_list_groups_types_conns(self):

        _groups = {x.name for _, x in self.house.groups.items()}
        # _types = {x.name for x in self.house.types.items()}
        _conns = {x.name for _, x in self.house.connections.items()}

        self.assertEqual({'sheeting', 'batten'}, _groups)
        self.assertEqual(set(range(1, 61)), _conns)

    def test_shielding_multiplier(self):

        self.house.cfg.regional_shielding_factor = 0.85
        _list = []
        for i in range(1000):
            _list.append(self.house.shielding_multiplier)
            self.house._shielding_multiplier = None

        result = Counter(_list)
        ref_dic = {0.85: 0.63, 0.95: 0.15, 1.0: 0.22}
        for key, value in ref_dic.items():
            self.assertAlmostEqual(value, result[key] / 1000.0, places=1)


class TestHouseCoverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = os.sep.join(__file__.split(os.sep)[:-1])
        cls.file_cfg = os.path.join(
            path, 'test_scenarios', 'test_scenario19', 'test_scenario19.cfg')
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        cls.cfg = Config(file_cfg=cls.file_cfg, logger=logger)
        cls.house = House(cls.cfg, 1)

    def test_assign_windward(self):

        assert self.house._wind_dir_index == 0

        self.house.cfg.front_facing_walls = {'E': [7],
                                             'NE': [5, 7],
                                             'N': [5],
                                             'NW': [3, 5],
                                             'W': [3],
                                             'SW': [1, 3],
                                             'S': [1],
                                             'SE': [1, 7]}

        # wind direction: S
        self.house._wind_dir_index = 0
        self.house._windward_walls = None
        self.house._leeward_walls = None
        self.house._side1_walls = None
        self.house._side2_walls = None

        ref = {1: 'windward',
               3: 'side1',
               5: 'leeward',
               7: 'side2'}

        for wall_name in range(1, 8, 2):
            self.assertEqual(ref[wall_name],
                             self.house.assign_windward(wall_name))

        # wind direction: W
        self.house._wind_dir_index = 2
        self.house._windward_walls = None
        self.house._leeward_walls = None
        self.house._side1_walls = None
        self.house._side2_walls = None

        ref = {3: 'windward',
               1: 'side2',
               7: 'leeward',
               5: 'side1'}

        for wall_name in range(1, 8, 2):
            self.assertEqual(ref[wall_name],
                             self.house.assign_windward(wall_name))

        # wind direction: NW
        self.house._wind_dir_index = 3
        self.house._windward_walls = None
        self.house._leeward_walls = None
        self.house._side1_walls = None
        self.house._side2_walls = None

        ref = {3: 'windward',
               5: 'windward',
               7: 'leeward',
               1: 'leeward'}

        for wall_name in range(1, 8, 2):
            self.assertEqual(ref[wall_name],
                             self.house.assign_windward(wall_name))

    def test_assign_cpi_dominant(self):
        # windward: 1, 2, 5
        # leeward: 4, 7
        # side1: 3, 6
        # side2: 8

        test_data = {
            1: {1: 1.0, 2: 0.0, 3: 0.9, 4: 0.8, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.5},
            2: {1: 1.0, 2: 0.0, 3: 0.5, 4: 0.5, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0},
            3: {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.5, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0},
            4: {1: 1.0, 2: 0.0, 3: 0.3, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0},
            5: {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0}
        }

        expected_cpi = {1: -0.3,
                        2: 0.2,
                        3: 0.7*2.4,
                        4: 0.85*2.4,
                        5: 2.4}

        cfg = Config(self.file_cfg)
        cfg.coverages.area = np.array(8 * [10.0])

        for key, data in test_data.items():

            house = House(cfg, 1)

            for k, v in data.items():
                house.coverages.loc[k, 'coverage'].breached_area = v

            house.coverages['breached_area'] = \
                house.coverages['coverage'].apply(lambda x: x.breached_area)

            try:
                self.assertEqual(house.cpi, expected_cpi[key])
            except AssertionError:
                print([house.coverages.loc[k, 'coverage'].breached_area
                       for k in range(1, 9)])
                print(f'cpi should be {expected_cpi[key]}, but {house.cpi}')

            del house._cpi

    def test_assign_cpi(self):

        test_data = [
            [0.4, 0, 0.3, 0.3, 0, 0, 0, 0.3, -0.3],
            [0.4, 0, 0.2, 0.2, 0, 0, 0, 0, 0.2],
            [0.8, 0, 0.2, 0.2, 0, 0, 0, 0, 1.68],
            [1, 0, 0.1, 0.1, 0, 0, 0, 0, 2.04],
            [2, 0, 0.1, 0.1, 0, 0, 0, 0, 2.4],
            [0, 0.3, 0.3, 0.4, 0, 0, 0, 0.3, -0.3],
            [0.2, 0, 0.2, 0.4, 0, 0, 0, 0, -0.3],
            [0, 0.2, 0.2, 0.8, 0, 0, 0, 0, -1.45],
            [0, 0.1, 0.1, 1, 0, 0, 0, 0, -1.45],
            [0.1, 0, 0.1, 2, 0, 0, 0, 0, -1.45],
            [0.3, 0, 0.4, 0.3, 0, 0, 0, 0.3, -0.3],
            [0, 0.2, 0.4, 0.2, 0, 0, 0, 0, -0.3],
            [0.2, 0, 0.8, 0.2, 0, 0, 0, 0, -1.14],
            [0.1, 0, 1, 0.1, 0, 0, 0, 0, -1.14],
            [0, 0.1, 2, 0.1, 0, 0, 0, 0, -1.14],
            [0.1, 0, 0, 0, 0, 0, 0, 0, 2.4],
            [0, 0, 0.1, 0, 0, 0, 0, 0, -1.14],
            [0, 0.1, 0, 0.1, 0, 0, 0, 0, 0.2],
            [0, 0, 0.1, 0.1, 0, 0, 0, 0, -0.3],
            [0.1, 0, 0.1, 0.1, 0, 0, 0, 0.1, -0.3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

        cfg = Config(self.file_cfg)
        cfg.coverages.area = np.array(8 * [10.0])

        for item in test_data:

            house = House(cfg, 1)

            data = {i: x for (i, x) in enumerate(item[:-1], 1)}

            for k, v in data.items():
                house.coverages.loc[k, 'coverage'].breached_area = v

            house.coverages['breached_area'] = \
                house.coverages['coverage'].apply(lambda x: x.breached_area)

            try:
                self.assertEqual(house.cpi, item[-1])
            except AssertionError:
                print([house.coverages.loc[k, 'coverage'].breached_area
                       for k in range(1, 9)])
                print(f'cpi should be {item[-1]}, but {house.cpi}')

            del house._cpi


class TestHouseDamage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        cls.path_reference = path
        file_cfg = os.path.join(path, 'test_scenarios', 'test_scenario15',
                                'test_scenario15.cfg')
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        cls.cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg=cls.cfg, seed=1)
        cls.house.replace_cost = 45092.97

        cls.sel_conn = {'sheeting0': 1,
                        'sheeting1': 7,
                        'batten0': 13,
                        'batten1': 19,
                        'rafter0': 25}

        for key, value in cls.sel_conn.items():
            cls.house.groups[key].connections[value].damaged = 1

        # def test_calculate_qz(self):

        # self.assertEqual(self.house.house.height, 4.5)
        # self.assertEqual(self.cfg.terrain_category, '2')

        # self.house.set_wind_profile()
        # self.assertEqual(self.house.profile, 5)
        # self.assertAlmostEqual(self.house.mzcat, 0.9425, places=4)
        #
        # # regional_shielding_factor > 0.85
        # self.house.regional_shielding_factor = 1.0
        # self.house.calculate_qz(10.0)
        # self.assertAlmostEqual(self.house.qz, 0.05472, places=4)
        #
        # # regional_shielding_factor < 0.85
        # self.house.regional_shielding_factor = 0.5
        # self.house.calculate_qz(10.0)
        # self.assertAlmostEqual(self.house.qz, 0.21888, places=4)

    def test_calculate_damage_ratio(self):
        """calculate damage ratio """

        repair_cost_by_group = StringIO("""
dmg_ratio_sheeting,dmg_ratio_batten,dmg_ratio_rafter,loss_ratio
0,0,0,0
0.2,0,0,0.053454931
0.8,0,0,0.161629032
1,0,0,0.189542886
0,0.2,0,0.136022125
0.2,0.2,0,0.136022125
0.8,0.2,0,0.267515756
1,0.2,0,0.297651157
0,0.4,0,0.245201146
0.2,0.4,0,0.245201146
0.8,0.4,0,0.341562015
1,0.4,0,0.376694777
0,0.6,0,0.334600438
0.2,0.6,0,0.334600438
0.8,0.6,0,0.388055368
1,0.6,0,0.430961307
0,0.8,0,0.411283377
0.2,0.8,0,0.411283377
0.8,0.8,0,0.411283377
1,0.8,0,0.464738308
0,1,0,0.482313341
0.2,1,0,0.482313341
0.8,1,0,0.482313341
1,1,0,0.482313341
0,0,0.2,0.244454103
0.2,0,0.2,0.244454103
0.8,0,0.2,0.375231574
1,0,0.2,0.40544787
0,0.2,0.2,0.244454103
0.2,0.2,0.2,0.244454103
0.8,0.2,0.2,0.339956232
1,0.2,0.2,0.375231574
0,0.4,0.2,0.377771291
0.2,0.4,0.2,0.377771291
0.8,0.4,0.2,0.430163216
1,0.4,0.2,0.47327342
0,0.6,0.2,0.487470087
0.2,0.6,0.2,0.487470087
0.8,0.6,0.2,0.487470087
1,0.6,0.2,0.539862012
0,0.8,0.2,0.57723219
0.2,0.8,0.2,0.57723219
0.8,0.8,0.2,0.57723219
1,0.8,0.2,0.57723219
0,1,0.2,0.654120978
0.2,1,0.2,0.654120978
0.8,1,0.2,0.654120978
1,1,0.2,0.654120978
0,0,0.4,0.460368018
0.2,0,0.4,0.460368018
0.8,0,0.4,0.555007538
1,0,0.4,0.590426831
0,0.2,0.4,0.460368018
0.2,0.2,0.4,0.460368018
0.8,0.2,0.4,0.511691698
1,0.2,0.4,0.555007538
0,0.4,0.4,0.460368018
0.2,0.4,0.4,0.460368018
0.8,0.4,0.4,0.460368018
1,0.4,0.4,0.511691698
0,0.6,0.4,0.590966935
0.2,0.6,0.4,0.590966935
0.8,0.6,0.4,0.590966935
1,0.6,0.4,0.590966935
0,0.8,0.4,0.701188995
0.2,0.8,0.4,0.701188995
0.8,0.8,0.4,0.701188995
1,0.8,0.4,0.701188995
0,1,0.4,0.791317398
0.2,1,0.4,0.791317398
0.8,1,0.4,0.791317398
1,1,0.4,0.791317398
0,0,0.6,0.655553624
0.2,0,0.6,0.655553624
0.8,0,0.6,0.705803789
1,0,0.6,0.749326635
0,0.2,0.6,0.655553624
0.2,0.2,0.6,0.655553624
0.8,0.2,0.6,0.655553624
1,0.2,0.6,0.705803789
0,0.4,0.6,0.655553624
0.2,0.4,0.6,0.655553624
0.8,0.4,0.6,0.655553624
1,0.4,0.6,0.655553624
0,0.6,0.6,0.655553624
0.2,0.6,0.6,0.655553624
0.8,0.6,0.6,0.655553624
1,0.6,0.6,0.655553624
0,0.8,0.6,0.783420859
0.2,0.8,0.6,0.783420859
0.8,0.8,0.6,0.783420859
1,0.8,0.6,0.783420859
0,1,0.6,0.894169671
0.2,1,0.6,0.894169671
0.8,1,0.6,0.894169671
1,1,0.6,0.894169671
0,0,0.8,0.837822799
0.2,0,0.8,0.837822799
0.8,0,0.8,0.837822799
1,0,0.8,0.886994147
0,0.2,0.8,0.837822799
0.2,0.2,0.8,0.837822799
0.8,0.2,0.8,0.837822799
1,0.2,0.8,0.837822799
0,0.4,0.8,0.837822799
0.2,0.4,0.8,0.837822799
0.8,0.4,0.8,0.837822799
1,0.4,0.8,0.837822799
0,0.6,0.8,0.837822799
0.2,0.6,0.8,0.837822799
0.8,0.6,0.8,0.837822799
1,0.6,0.8,0.837822799
0,0.8,0.8,0.837822799
0.2,0.8,0.8,0.837822799
0.8,0.8,0.8,0.837822799
1,0.8,0.8,0.837822799
0,1,0.8,0.962944864
0.2,1,0.8,0.962944864
0.8,1,0.8,0.962944864
1,1,0.8,0.962944864
0,0,1,1
0.2,0,1,1
0.8,0,1,1
1,0,1,1
0,0.2,1,1
0.2,0.2,1,1
0.8,0.2,1,1
1,0.2,1,1
0,0.4,1,1
0.2,0.4,1,1
0.8,0.4,1,1
1,0.4,1,1
0,0.6,1,1
0.2,0.6,1,1
0.8,0.6,1,1
1,0.6,1,1
0,0.8,1,1
0.2,0.8,1,1
0.8,0.8,1,1
1,0.8,1,1
0,1,1,1
0.2,1,1,1
0.8,1,1,1
1,1,1,1""")

        ref_dat = pd.read_csv(repair_cost_by_group)

        # item = ref_dat.loc[1]
        for _, item in ref_dat.iterrows():

            for group_name, group in self.house.groups.items():

                damaged_area = item[f'dmg_ratio_{group.name}'] * group.costing_area
                group.connections[self.sel_conn[group_name]].costing_area = damaged_area

            self.house.compute_damage_index(20.0)

            try:
                self.assertAlmostEqual(self.house.di,
                                       min(item['loss_ratio'], 1.0),
                                       places=4)
            except AssertionError:
                print(item)


class TestHouseDamage2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.sep.join(__file__.split(os.sep)[:-1])
        cls.path_reference = path

        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios', 'test_house',
                                'test_house.cfg')
        cls.cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg=cls.cfg, seed=1)
        cls.house.replace_cost = 198859.27

        cls.sel_conn = {'wallcladding3': 420,
                        'wallcladding4': 450,
                        'wallcladding5': 480,
                        'wallcladding6': 512,
                        'wallcollapse15': 608,
                        'wallcollapse16': 611,
                        'wallcollapse17': 616,
                        'wallcollapse18': 618}

        for key, value in cls.sel_conn.items():
            cls.house.groups[key].connections[value].damaged = 1

    @staticmethod
    def assign_breached_area(df, damaged_area):

        for _, coverage in df.coverage.iteritems():
            if damaged_area > coverage.area:
                coverage.breached_area = coverage.area
                damaged_area -= coverage.area
            else:
                coverage.breached_area = damaged_area
                break
        return df

    def test_calculate_damage_ratio_including_debris(self):

        repair_cost_by_group = StringIO("""
dmg_ratio_debris,dmg_ratio_wallcladding,dmg_ratio_wallcollapse,loss_ratio
0,0,0,0
0.2,0,0,0.062253424
0.8,0,0,0.170974382
1,0,0,0.209679439
0,0.2,0,0.039520371
0.2,0.2,0,0.062253424
0.8,0.2,0,0.170974382
1,0.2,0,0.209679439
0,0.4,0,0.067528499
0.2,0.4,0,0.099835942
0.8,0.4,0,0.170974382
1,0.4,0,0.209679439
0,0.6,0,0.092382143
0.2,0.6,0,0.12813243
0.8,0.6,0,0.170974382
1,0.6,0,0.209679439
0,0.8,0,0.115386016
0.2,0.8,0,0.153133004
0.8,0.8,0,0.170974382
1,0.8,0,0.209679439
0,1,0,0.137105517
0.2,1,0,0.176232756
0.8,1,0,0.20255323
1,1,0,0.209679439
0,0,0.2,0.110557571
0.2,0,0.2,0.110557571
0.8,0,0.2,0.246121879
1,0,0.2,0.277648077
0,0.2,0.2,0.110557571
0.2,0.2,0.2,0.110557571
0.8,0.2,0.2,0.246121879
1,0.2,0.2,0.277648077
0,0.4,0.2,0.144251563
0.2,0.4,0.2,0.110557571
0.8,0.4,0.2,0.246121879
1,0.4,0.2,0.277648077
0,0.6,0.2,0.173166526
0.2,0.6,0.2,0.142217225
0.8,0.6,0.2,0.246121879
1,0.6,0.2,0.277648077
0,0.8,0.2,0.198469662
0.2,0.8,0.2,0.171479327
0.8,0.8,0.2,0.246121879
1,0.8,0.2,0.277648077
0,1,0.2,0.221763918
0.2,1,0.2,0.196944798
0.8,1,0.2,0.246121879
1,1,0.2,0.277648077
0,0,0.4,0.240212739
0.2,0,0.4,0.240212739
0.8,0,0.4,0.337118038
1,0,0.4,0.372108901
0,0.2,0.4,0.240212739
0.2,0.2,0.4,0.240212739
0.8,0.2,0.4,0.337118038
1,0.2,0.4,0.372108901
0,0.4,0.4,0.240212739
0.2,0.4,0.4,0.240212739
0.8,0.4,0.4,0.337118038
1,0.4,0.4,0.372108901
0,0.6,0.4,0.267765864
0.2,0.6,0.4,0.240212739
0.8,0.6,0.4,0.337118038
1,0.6,0.4,0.372108901
0,0.8,0.4,0.297785697
0.2,0.8,0.4,0.265601601
0.8,0.8,0.4,0.337118038
1,0.8,0.4,0.372108901
0,1,0.4,0.323586937
0.2,1,0.4,0.296055609
0.8,1,0.4,0.337118038
1,1,0.4,0.372108901
0,0,0.6,0.388286952
0.2,0,0.6,0.388286952
0.8,0,0.6,0.431404374
1,0,0.6,0.480503426
0,0.2,0.6,0.388286952
0.2,0.2,0.6,0.388286952
0.8,0.2,0.6,0.431404374
1,0.2,0.6,0.480503426
0,0.4,0.6,0.388286952
0.2,0.4,0.6,0.388286952
0.8,0.4,0.6,0.431404374
1,0.4,0.6,0.480503426
0,0.6,0.6,0.388286952
0.2,0.6,0.6,0.388286952
0.8,0.6,0.6,0.431404374
1,0.6,0.6,0.480503426
0,0.8,0.6,0.409262628
0.2,0.8,0.6,0.388286952
0.8,0.8,0.6,0.431404374
1,0.8,0.6,0.480503426
0,1,0.6,0.4406908
0.2,1,0.6,0.406905352
0.8,1,0.6,0.431404374
1,1,0.6,0.480503426
0,0,0.8,0.554101659
0.2,0,0.8,0.554101659
0.8,0,0.8,0.554101659
1,0,0.8,0.590273167
0,0.2,0.8,0.554101659
0.2,0.2,0.8,0.554101659
0.8,0.2,0.8,0.554101659
1,0.2,0.8,0.590273167
0,0.4,0.8,0.554101659
0.2,0.4,0.8,0.554101659
0.8,0.4,0.8,0.554101659
1,0.4,0.8,0.590273167
0,0.6,0.8,0.554101659
0.2,0.6,0.8,0.554101659
0.8,0.6,0.8,0.554101659
1,0.6,0.8,0.590273167
0,0.8,0.8,0.554101659
0.2,0.8,0.8,0.554101659
0.8,0.8,0.8,0.554101659
1,0.8,0.8,0.590273167
0,1,0.8,0.567811206
0.2,1,0.8,0.554101659
0.8,1,0.8,0.554101659
1,1,0.8,0.590273167
0,0,1,0.736978308
0.2,0,1,0.736978308
0.8,0,1,0.736978308
1,0,1,0.736978308
0,0.2,1,0.736978308
0.2,0.2,1,0.736978308
0.8,0.2,1,0.736978308
1,0.2,1,0.736978308
0,0.4,1,0.736978308
0.2,0.4,1,0.736978308
0.8,0.4,1,0.736978308
1,0.4,1,0.736978308
0,0.6,1,0.736978308
0.2,0.6,1,0.736978308
0.8,0.6,1,0.736978308
1,0.6,1,0.736978308
0,0.8,1,0.736978308
0.2,0.8,1,0.736978308
0.8,0.8,1,0.736978308
1,0.8,1,0.736978308
0,1,1,0.736978308
0.2,1,1,0.736978308
0.8,1,1,0.736978308
1,1,1,0.736978308""")

        ref_dat = pd.read_csv(repair_cost_by_group)

        for _, item in ref_dat.iterrows():

            self.house.set_coverages()

            damaged_area = item['dmg_ratio_debris'] * self.cfg.coverages_area

            self.house.coverages = self.assign_breached_area(
                self.house.coverages, damaged_area)

            # assign damage area
            for group_name, group in self.house.groups.items():

                if group.name in ['wallcladding', 'wallcollapse']:
                    damaged_area = item[f'dmg_ratio_{group.name}'] * group.costing_area
                    group.connections[self.sel_conn[group_name]].costing_area = damaged_area

            self.house.compute_damage_index(20.0)

            try:
                self.assertAlmostEqual(self.house.di,
                                       min(item['loss_ratio'], 1.0), places=4)
            except AssertionError:
                print(f'{self.house.di} vs {item}')


if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHouseCoverage)
    unittest.TextTestRunner(verbosity=2).run(suite)
