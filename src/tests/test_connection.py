import unittest
import os
import numpy as np

from vaws.scenario import Scenario
from vaws.house import House


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cfg_file = os.path.join(path,
                                '../../scenarios/test_sheeting_batten.cfg')
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
            self.assertAlmostEqual(_zone.pz, qz * _zone.cpe_mean[0], places=4)

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

if __name__ == '__main__':
    unittest.main()

# suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
# unittest.TextTestRunner(verbosity=2).run(suite)
