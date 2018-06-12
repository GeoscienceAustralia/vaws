import unittest
import os
import numpy as np

from vaws.model.config import Config
from vaws.model.house import House


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        cfg_file = os.path.join(path, 'test_scenarios', 'test_sheeting_batten',
                                'test_sheeting_batten.cfg')
        cls.cfg = Config(cfg_file=cfg_file)

    def test_prop_damaged(self):

        house = House(self.cfg, 1)
        group = house.groups['sheeting0']  # sheeting

        for _, _conn in house.connections.items():
            self.assertEqual(_conn.damaged, False)

        # 1: sheeting gable(1): 2 - 5
        # 2: sheeting eave(2): 7, 12, 13, 18, 19, 24, 25, 30
        # 3: sheeting corner(3): 1, 6
        # 4: sheeting(4): 8 - 11, 14 - 17, 20 - 23, 26 - 29
        for _id in [1, 4, 5, 7, 8, 11, 12]:
            house.connections[_id].capacity = 20.0
            house.connections[_id].damaged = 1

        ref_dic = {'sheetinggable': 4, 'sheetingeave': 8,
                   'sheetingcorner': 2, 'sheeting': 16}
        ref_area = {'sheetinggable': 0.405, 'sheetingeave': 0.405,
                    'sheetingcorner': 0.225, 'sheeting': 0.81}
        # for id_type, _type in group.types.items():
        #     self.assertEqual(_type.no_connections, ref_dic[id_type])
        #     self.assertEqual(_type.costing_area, ref_area[id_type])

        # costing area by group
        self.assertAlmostEqual(group.costing_area, 18.27, places=2)

        #group.compute_damaged_area()
        # self.assertAlmostEqual(group.damaged_area, 3.465, places=4)
        self.assertAlmostEqual(group.damaged_area, 3.465, places=4)

    def test_compute_load(self):

        house = House(self.cfg, 1)

        # compute zone pressures
        assert house.wind_dir_index == 3
        cpi = 0.0
        qz = 0.8187
        combination_factor = house.combination_factor

        _zone = house.zones['A1']
        _zone._cpe = _zone.cpe_mean[0]  # originally randomly generated
        _zone._cpe_eave = _zone.cpe_eave_mean[0]
        _zone._cpe_str = _zone.cpe_str_mean[0]
        _zone.shielding_multiplier = 1.0

        _zone.calc_zone_pressure(cpi, qz, combination_factor)

        self.assertAlmostEqual(_zone.cpi_alpha, 0.0, places=2)
        self.assertAlmostEqual(_zone.cpe_mean[0], -1.25, places=2)
        self.assertAlmostEqual(_zone.cpe_eave_mean[0], 0.7, places=2)
        self.assertAlmostEqual(_zone.cpe_str_mean[0], 0.0, places=2)
        self.assertAlmostEqual(house.zones['A1'].area, 0.2025, places=4)

        # pz = qz * (cpe - cpi_alpha * cpi - cpe_eave) * differential_shielding
        #self.assertAlmostEqual(house.zones['A1'].pressure, -1.0234, places=4)

        _conn = house.connections[1]

        # init
        self.assertEqual(_conn.damaged, False)
        self.assertAlmostEqual(_conn.dead_load, 0.01013, places=4)

        # load = influence.pz * influence.coeff * influence.area + dead_load
        # ref_cpe = qz * (_zone.cpe - _zone.cpe_eave)
        self.assertAlmostEqual(_conn.influences['A1'].source.pressure_cpe,
                               -1.5965, places=4)
        # ref_cpe_str = qz * (_zone.cpe_str - _zone.cpe_eave)
        self.assertAlmostEqual(_conn.influences['A1'].source.pressure_cpe_str,
                               -0.5731, places=4)
        # ref_load = _zone.area * ref_cpe + _conn.dead_load
        self.assertAlmostEqual(_conn.load, -0.3132, places=4)

    def test_check_damage(self):

        house = House(self.cfg, 1)

        # compute zone pressures
        mzcat = 0.9235
        wind_speed = 75.0
        wind_dir_index = 3
        cpi = 0.0
        qz = 0.6 * 1.0e-3 * (wind_speed * mzcat) ** 2
        building_spacing = 0
        combination_factor = house.combination_factor

        # compute pz using constant cpe
        for _zone in house.zones.itervalues():
            _zone._cpe = _zone.cpe_mean[0]
            _zone._cpe_eave = _zone.cpe_eave_mean[0]
            _zone._cpe_str = _zone.cpe_str_mean[0]
            _zone.shielding_multiplier = 1.0
            _zone.calc_zone_pressure(cpi, qz, combination_factor)
            ref_cpe = qz * (_zone.cpe_mean[0] - _zone.cpe_eave_mean[0])
            ref_cpe_str = qz * (_zone.cpe_str_mean[0] - _zone.cpe_eave_mean[0])
            self.assertAlmostEqual(_zone.pressure_cpe, ref_cpe, places=4)
            self.assertAlmostEqual(_zone.pressure_cpe_str, ref_cpe_str, places=4)

        # compute dead_load and strength using constant values
        for _conn in house.connections.itervalues():
            _conn.lognormal_dead_load = _conn.lognormal_dead_load[0], 0.0
            _conn.lognormal_strength = _conn.lognormal_strength[0], 0.0
            _conn.mean_factor = 1.0
            _conn.cv_factor = 0.0
            _conn.rnd_state = house.rnd_state

            self.assertAlmostEqual(_conn.dead_load,
                                   np.exp(_conn.lognormal_dead_load[0]),
                                   places=4)

            self.assertAlmostEqual(_conn.strength,
                                   np.exp(_conn.lognormal_strength[0]),
                                   places=4)

        # check load
        group = house.groups['sheeting0']
        group.check_damage(wind_speed=wind_speed)

        ref_dic = {x: False for x in range(1, 61)}
        for i in [2, 8, 14, 20, 26]:
            ref_dic[i] = True

        for id_conn, _conn in house.connections.items():
            try:
                self.assertEqual(_conn.damaged, ref_dic[id_conn])
            except AssertionError:
                print('{}: {} vs {}'.format(_conn.name, _conn.damaged,
                                            ref_dic[id_conn]))

        # ref_prop = {'sheetinggable': 0.25, 'sheetingeave': 0.0,
        #            'sheetingcorner': 0.0, 'sheeting': 0.25}
        # ref_capacity = {'sheetinggable': 75.0, 'sheetingeave': 9999,
        #                'sheetingcorner': 9999, 'sheeting': 75.0}
        # for id_type, _type in group.types.items():
            #self.assertAlmostEqual(_type.prop_damaged_type,
            #                       ref_prop[id_type], places=3)
            #self.assertAlmostEqual(_type.capacity,
            #                      ref_capacity[id_type], places=1)


class MyTestCaseConnectionGroup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.sep.join(__file__.split(os.sep)[:-1])
        cfg_file = os.path.join(path, 'test_scenarios',
                                'test_scenario16',
                                'test_scenario16.cfg')
        cls.cfg = Config(cfg_file=cfg_file)

    def assert_influence_coeff(self, _dic, house_inst):

        for key0, value0 in _dic.items():
            for key1, value1 in value0.items():
                code_value = house_inst.connections[key0].influences[key1].coeff
                try:
                    self.assertEqual(code_value, value1)
                except AssertionError:
                    print('infl for {}:{} should be {} not {}'.format(
                        key0, key1, value1, code_value))

    def test_update_influence_by_patch1(self):

        house = House(self.cfg, 1)

        init_dic = {121: {'A13': 0.81, 'A14': 0.19},
                    122: {'A13': 0.75, 'A14': 0.75},
                    123: {'A13': 0.19, 'A14': 0.81}}

        self.assert_influence_coeff(init_dic, house)

        # connection 121 failed
        failed = house.groups['rafter0'].connections[121]
        failed.damaged = 1
        house.groups['rafter0'].update_influence_by_patch(failed,
                                                          house)

        _dic = {121: {'A13': 0.0, 'A14': 0.0},
                122: {'A13': 1.0, 'A14': 0.0},
                123: {'A13': 1.0, 'A14': 1.0}}

        self.assert_influence_coeff(_dic, house)

        # connection 122 failed after 121
        failed = house.groups['rafter0'].connections[122]
        failed.damaged = 1
        house.groups['rafter0'].update_influence_by_patch(failed,
                                                          house)

        assert house.connections[121].damaged == 1.0
        assert house.connections[122].damaged == 1.0

        _dic = {121: {'A13': 0.0, 'A14': 0.0},
                122: {'A13': 0.0, 'A14': 0.0},
                123: {'A13': 0.0, 'A14': 1.0}}

        self.assert_influence_coeff(_dic, house)

    def test_update_influence_by_patch2(self):

        house = House(self.cfg, 1)

        init_dic = {121: {'A13': 0.81, 'A14': 0.19},
                    122: {'A13': 0.75, 'A14': 0.75},
                    123: {'A13': 0.19, 'A14': 0.81}}

        self.assert_influence_coeff(init_dic, house)

        # connection 122 failed
        failed = house.groups['rafter0'].connections[122]
        failed.damaged = 1
        house.groups['rafter0'].update_influence_by_patch(failed,
                                                          house)

        _dic = {121: {'A13': 1.0, 'A14': 0.0},
                122: {'A13': 0.0, 'A14': 0.0},
                123: {'A13': 0.0, 'A14': 1.0}}

        self.assert_influence_coeff(_dic, house)

        # connection 121 failed after 122
        failed = house.groups['rafter0'].connections[121]
        failed.damaged = 1
        house.groups['rafter0'].update_influence_by_patch(failed,
                                                          house)

        assert house.connections[121].damaged == 1.0
        assert house.connections[122].damaged == 1.0

        _dic = {121: {'A13': 0.0, 'A14': 0.0},
                122: {'A13': 0.0, 'A14': 0.0},
                123: {'A13': 1.0, 'A14': 1.0}}

        self.assert_influence_coeff(_dic, house)

    def test_update_influence_for_connection1(self):

        rnd_state = np.random.RandomState(1)
        house = House(self.cfg, 1)

        init_dic = {1: {'A1': 1.0},
                    2: {'A2': 1.0},
                    3: {'A3': 1.0},
                    4: {'A4': 1.0},
                    5: {'A5': 1.0},
                    6: {'A6': 1.0}}

        self.assert_influence_coeff(init_dic, house)

        # connection 1 failed
        failed = house.connections[1]
        failed.damaged = 1
        target = house.connections[2]
        target.update_influence(failed, influence_coeff=1.0)

        _dic = {2: {'A2': 1.0, 'A1': 1.0},
                3: {'A3': 1.0},
                4: {'A4': 1.0},
                5: {'A5': 1.0},
                6: {'A6': 1.0}}

        self.assert_influence_coeff(_dic, house)

        # connection 2 failed after connection 1
        failed = house.connections[2]
        failed.damaged = 1
        target = house.connections[3]
        target.update_influence(failed, influence_coeff=1.0)

        _dic = {3: {'A2': 1.0, 'A1': 1.0, 'A3': 1.0},
                4: {'A4': 1.0},
                5: {'A5': 1.0},
                6: {'A6': 1.0}}

        self.assert_influence_coeff(_dic, house)

    def test_update_influence_for_connection2(self):

        house = House(self.cfg, 1)

        init_dic = {1: {'A1': 1.0},
                    2: {'A2': 1.0},
                    3: {'A3': 1.0},
                    4: {'A4': 1.0},
                    5: {'A5': 1.0},
                    6: {'A6': 1.0}}

        self.assert_influence_coeff(init_dic, house)

        # connection 2 failed
        failed = house.connections[2]
        failed.damaged = 1
        target = house.connections[1]
        target.update_influence(failed, influence_coeff=0.5)

        target = house.connections[3]
        target.update_influence(failed, influence_coeff=0.5)

        _dic = {1: {'A2': 0.5, 'A1': 1.0},
                3: {'A3': 1.0, 'A2': 0.5},
                4: {'A4': 1.0},
                5: {'A5': 1.0},
                6: {'A6': 1.0}}

        self.assert_influence_coeff(_dic, house)

        # connection 3 failed after connection 2
        failed = house.connections[3]
        failed.damaged = 1
        target = house.connections[1]
        target.update_influence(failed, influence_coeff=0.5)

        target = house.connections[4]
        target.update_influence(failed, influence_coeff=0.5)

        _dic = {1: {'A2': 0.75, 'A1': 1.0, 'A3': 0.5},
                4: {'A4': 1.0, 'A3': 0.5, 'A2': 0.25},
                5: {'A5': 1.0},
                6: {'A6': 1.0}}

        self.assert_influence_coeff(_dic, house)

    def test_update_influence_for_group1(self):
        """corresponding to test_update_influence_for_connection1, but uses
        group level update_influence
        """

        house = House(self.cfg, 1)

        # sheeting1 group connection 1 failed
        _group = house.groups['sheeting1']
        _group.damage_grid[0, 0] = 1
        _group.damage_grid_index = [(0, 0)]

        _group.update_influence(house)

        _dic = {2: {'A2': 1.0, 'A1': 1.0},
                3: {'A3': 1.0},
                4: {'A4': 1.0}}

        self.assert_influence_coeff(_dic, house)

        # connection 2 failed after connection 1
        _group.damage_grid[0, 1] = 1
        _group.damage_grid_index = [(0, 1)]

        _group.update_influence(house)

        _dic = {3: {'A2': 1.0, 'A1': 1.0, 'A3': 1.0}}

        self.assert_influence_coeff(_dic, house)

    def test_update_influence_for_group2(self):
        """corresponding to test_update_influence_by_patch1, but uses
        group level update_influence
        """

        house = House(self.cfg, 1)

        init_dic = {121: {'A13': 0.81, 'A14': 0.19},
                    122: {'A13': 0.75, 'A14': 0.75},
                    123: {'A13': 0.19, 'A14': 0.81}}

        self.assert_influence_coeff(init_dic, house)
        # connection 121 failed
        _group = house.groups['rafter0']
        _group.damage_grid[0, 0] = 1
        _group.damage_grid_index = [(0, 0)]
        _group.connections[121].damaged = 1.0
        _group.update_influence(house)

        _dic = {121: {'A13': 0.0, 'A14': 0.0},
                122: {'A13': 1.0, 'A14': 0.0},
                123: {'A13': 1.0, 'A14': 1.0}}

        self.assert_influence_coeff(_dic, house)

        # connection 122 failed after 121
        _group.damage_grid[0, 4] = 1
        _group.damage_grid_index = [(0, 4)]
        _group.connections[122].damaged = 1.0
        _group.update_influence(house)

        _dic = {121: {'A13': 0.0, 'A14': 0.0},
                122: {'A13': 0.0, 'A14': 0.0},
                123: {'A13': 0.0, 'A14': 1.0}}

        self.assert_influence_coeff(_dic, house)

if __name__ == '__main__':
    unittest.main()
    # suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCaseConnectionGroup)
    # unittest.TextTestRunner(verbosity=2).run(suite)
