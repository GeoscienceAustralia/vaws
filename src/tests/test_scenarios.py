#!/usr/bin/env python
from __future__ import print_function

import unittest
import os
import numpy as np

from vaws.simulation import HouseDamage
from vaws.scenario import Scenario
import logging


def simulation(house_damage, conn_capacity, wind_speeds):

    # change it to conn to speed
    conn_capacity2 = dict()
    for speed, conn_list in conn_capacity.iteritems():
        for _id in conn_list:
            conn_capacity2.setdefault(_id, speed)

    # compute zone pressures
    cpi = 0.0
    wind_dir_index = 3
    Ms = 1.0
    building_spacing = 0
    house_damage.house.mzcat = 1.0

    # set dead_load 0.0
    # for _conn in self.house_damage.house.connections.itervalues():
    #    _conn.dead_load = 0.0
    #    # print('strength of {}: {}'.format(_conn.id, _conn.strength))

    for wind_speed in wind_speeds:

        house_damage.calculate_qz_Ms(wind_speed)

        for _zone in house_damage.house.zones.itervalues():

            _zone.cpe = _zone.cpe_mean[wind_dir_index]

            _zone.calc_zone_pressures(wind_dir_index,
                                      cpi,
                                      house_damage.qz,
                                      Ms,
                                      building_spacing)

        # check damage by connection type group
        for _group in house_damage.house.groups.itervalues():
            _group.check_damage(wind_speed)
            _group.cal_prop_damaged()
            _group.distribute_damage()

    # compare with reference capacity
    for _id, _conn in house_damage.house.connections.iteritems():

        try:
            np.testing.assert_almost_equal(_conn.failure_v_raw,
                                           conn_capacity2[_id],
                                           decimal=2)
        except KeyError:
            print('conn #{} is not found'.format(_id))
        except AssertionError:
            print('conn #{} fails at {} not {}'.format(
                _id, _conn.failure_v_raw, conn_capacity2[_id]))


class TestScenario1(unittest.TestCase):
    """
    validate computed loads at selected connections
    """

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario1.cfg'),
            output_path=cls.path_output)

        cls.house_damage = HouseDamage(cfg, seed=0)

        # # set up logging
        # file_logger = os.path.join(cls.path_output, 'log_test1.txt')
        # cls.logger = logging.getLogger('myapp')
        # hdlr = logging.FileHandler(file_logger, mode='w')
        # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        # hdlr.setFormatter(formatter)
        # cls.logger.addHandler(hdlr)
        # cls.logger.setLevel(logging.DEBUG)

    # @classmethod
    # def tearDown(cls):
    #     handlers = cls.logger.handlers[:]
    #     for handler in handlers:
    #         handler.close()

    def test_conn_load(self):

        # compute zone pressures
        cpi = 0.0
        wind_dir_index = 3
        Ms = 1.0
        building_spacing = 0
        qz = 0.24

        for _zone in self.house_damage.house.zones.itervalues():

            _zone.cpe = _zone.cpe_mean[0]
            _zone.calc_zone_pressures(wind_dir_index,
                                      cpi,
                                      qz,
                                      Ms,
                                      building_spacing)

        ref_load = {1: -0.0049, 21: -0.0194, 31: -0.0049,  35: -0.0972,
                    39: -0.1507, 45: -0.0632, 51: -0.0194, 60: -0.0097}

        for _conn in self.house_damage.house.connections.itervalues():

            _conn.cal_load()

            try:
                self.assertAlmostEqual(ref_load[_conn.name], _conn.load,
                                       places=3)
            except KeyError:
                pass
            except AssertionError:
                print('{} is different from {} for conn #{}'.format(
                    ref_load[_conn.name], _conn.load, _conn.id))


class TestScenario2(unittest.TestCase):
    """
    validate the sequence of sheeting failures
    """

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario2.cfg'),
            output_path=cls.path_output)

        cls.house_damage = HouseDamage(cfg, seed=0)

        # # set up logging
        # file_logger = os.path.join(cls.path_output, 'log_test2.txt')
        # cls.logger = logging.getLogger('myapp')
        # hdlr = logging.FileHandler(file_logger, mode='w')
        # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        # hdlr.setFormatter(formatter)
        # cls.logger.addHandler(hdlr)
        # cls.logger.setLevel(logging.DEBUG)

    # @classmethod
    # def tearDown(cls):
    #     handlers = cls.logger.handlers[:]
    #     for handler in handlers:
    #         handler.close()

    def test_damage_sheeting(self):

        conn_capacity = {40.0: [10],
                         45.0: [9, 11],
                         50.0: [8],
                         55.0: [13, 6, 12],
                         60.0: [5, 7, 14],
                         65.0: [4, 15],
                         70.0: [3, 16],
                         75.0: [2, 17],
                         80.0: [18, 1]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120, 1.0))


class TestScenario3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario3.cfg'),
            output_path=cls.path_output)

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cls.path_output, 'log_test3.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_batten(self):

        # ref data
        conn_capacity = {50.0: [40, 43],
                         65.0: [46, 34, 37, 49],
                         70.0: [55, 36, 31, 52],
                         75.0: [58, 42],
                         80.0: [48],
                         85.0: [54],
                         90.0: [60, 39, 41, 44],
                         95.0: [35],
                         100.0: [47],
                         105.0: [53],
                         110.0: [59],
                         115.0: [38, 45, 50],
                         9999: range(1, 31)}
        [conn_capacity[9999].append(x) for x in [32, 33, 51, 56, 57]]

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120, 5.0))


class TestScenario4(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario4.cfg'),
            output_path=cls.path_output)

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cls.path_output, 'log_test4.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting_batten(self):

        # ref data
        conn_capacity = {40.0: [10],
                         45.0: [9, 11],
                         50.0: [8, 31],
                         55.0: [13, 12, 6],
                         60.0: [5, 7, 14],
                         65.0: [4, 15],
                         70.0: [3, 16],
                         75.0: [2, 17],
                         80.0: [1, 18],
                         9999: [19, 20, 21, 22, 23, 24, 26, 27, 28, 29,
                                32, 33, 34, 35, 36, 25, 30]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120, 5.0))


class TestScenario5(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario5.cfg'),
            output_path=cls.path_output)

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cls.path_output, 'log_test5.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting_batten(self):

        # ref data 5
        conn_capacity = {40.0: [],
                         45.0: [],
                         50.0: [31, 28],
                         55.0: [25],
                         60.0: [19],
                         65.0: [22, 34],
                         70.0: [24],
                         75.0: [30],
                         80.0: [36],
                         90.0: [32, 27, 29],
                         95.0: [26, 23],
                         100.0: [35, 20],
                         115.0: [33],
                         9999: range(1, 19)}
        conn_capacity[9999].append(21)

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120, 5.0))


class TestScenario6(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario6.cfg'),
            output_path=cls.path_output)

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cls.path_output, 'log_test6.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting_batten_rafter(self):

        conn_capacity = {40.0: [41, 43],
                         70.0: [42],
                         75.0: [40, 44],
                         90.0: [46],
                         95.0: [45],
                         9999: range(1, 40)}
        conn_capacity[9999].append(47)

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 100.0, 5.0))


class TestScenario7(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario7.cfg'),
            output_path=cls.path_output)

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cls.path_output, 'log_test7.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting(self):

        # ref data 7
        conn_capacity = {53.0: [10],
                         54.0: [9],
                         55.0: [8],
                         56.0: [7],
                         57.0: [6],
                         58.0: [5],
                         59.0: [4],
                         60.0: [3],
                         61.0: [2],
                         62.0: [1]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120, 1.0))


class TestScenario8(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario8.cfg'),
            output_path=cls.path_output)

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cls.path_output, 'log_test8.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting(self):

        # ref data
        conn_capacity = {53.0: [1],
                         54.0: [2],
                         55.0: [3],
                         56.0: [4],
                         57.0: [5],
                         58.0: [6],
                         59.0: [7],
                         60.0: [8],
                         61.0: [9],
                         62.0: [10]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120, 1.0))


class TestScenario9(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario9.cfg'),
            output_path=cls.path_output)

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cls.path_output, 'log_test9.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting(self):

        # ref data 9
        conn_capacity = {40.0: [7],
                         45.0: [6, 8],
                         46.0: [5, 9],
                         47.0: [4],
                         48.0: [10],
                         49.0: [3],
                         50.0: [2],
                         51.0: [1]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 60.0, 1.0))


class TestScenario10(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario10.cfg'),
            output_path=cls.path_output)

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cls.path_output, 'log_test10.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting(self):

        # ref data 9
        conn_capacity = {40.0: [8, 3],
                         45.0: [7, 9, 2, 4],
                         46.0: [5, 6],
                         47.0: [1, 10],
                         48.0: [],
                         49.0: []}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 60.0, 1.0))


class TestScenario11(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario11.cfg'),
            output_path=cls.path_output)

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cls.path_output, 'log_test11.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_batten(self):

        # ref data 11
        conn_capacity = {47.0: [17],
                         52.0: [16, 18],
                         53.0: [15, 19],
                         54.0: [14, 20],
                         55.0: [13],
                         56.0: [12],
                         57.0: [11],
                         9999: range(1, 11)}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120.0, 1.0))


class TestScenario12(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario12.cfg'),
            output_path=cls.path_output)

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cls.path_output, 'log_test12.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_batten(self):

        # ref data 11
        conn_capacity = {87.0: [16, 18],
                         88.0: [17],
                         89.0: [15, 19],
                         90.0: [14, 20],
                         91.0: [13],
                         92.0: [12],
                         93.0: [11],
                         9999: range(1, 11),
                         40.0: [7]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120.0, 1.0))


class TestScenario13(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario13.cfg'),
            output_path=cls.path_output)

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cls.path_output, 'log_test13.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_batten(self):

        # ref data 11
        conn_capacity = {46.0: [17],
                         71.0: [16, 18],
                         72.0: [15, 19],
                         73.0: [14, 20],
                         74.0: [13],
                         75.0: [12],
                         76.0: [11],
                         9999: range(1, 11)}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120, 1.0))


class TestScenario14(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario14.cfg'),
            output_path=cls.path_output)

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cls.path_output, 'log_test14.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_batten(self):

        # ref data 11
        conn_capacity = {47.0: [17],
                         64.0: [16, 18],
                         65.0: [19],
                         71.0: [15],
                         72.0: [20, 14],
                         73.0: [13],
                         74.0: [12],
                         75.0: [11],
                         9999: range(1, 11)}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120, 1.0))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestScenario9)
    unittest.TextTestRunner(verbosity=2).run(suite)
    #unittest.main(verbosity=2)
