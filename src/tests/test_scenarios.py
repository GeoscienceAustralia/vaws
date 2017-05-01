#!/usr/bin/env python
from __future__ import print_function

import unittest
import os
import numpy as np

from vaws.simulation import HouseDamage
from vaws.config import Config
import logging


def simulation(house_damage, conn_capacity, wind_speeds, list_connections):

    # change it to conn to speed
    conn_capacity2 = {x: -1.0 for x in list_connections}
    for speed, conn_list in conn_capacity.iteritems():
        for _id in conn_list:
            conn_capacity2.update({_id: speed})

    # compute zone pressures
    cpi = 0.0
    wind_dir_index = 0
    ms = 1.0
    building_spacing = 0
    house_damage.house.mzcat = 0.98325  # profile: 6, height: 4.5

    # set dead_load 0.0
    # for _conn in self.house_damage.house.connections.itervalues():
    #    _conn.dead_load = 0.0
    #    # print('strength of {}: {}'.format(_conn.id, _conn.strength))

    for wind_speed in wind_speeds:

        logging.info('wind speed {:.3f}'.format(wind_speed))

        house_damage.calculate_qz_ms(wind_speed)

        for _zone in house_damage.house.zones.itervalues():

            _zone.cpe = _zone.cpe_mean[wind_dir_index]
            _zone.cpe_str = _zone.cpe_str_mean[wind_dir_index]
            _zone.cpe_eave = _zone.cpe_eave_mean[wind_dir_index]

            _zone.calc_zone_pressures(wind_dir_index,
                                      cpi,
                                      house_damage.qz,
                                      ms,
                                      building_spacing)

        # check damage by connection type group
        for _group in house_damage.house.groups.itervalues():

            _group.check_damage(wind_speed)

            _group.cal_damaged_area()

            _group.distribute_damage()

        house_damage.cal_damage_index(wind_speed)

    # compare with reference capacity
    for _id, _conn in house_damage.house.connections.iteritems():

        try:
            np.testing.assert_almost_equal(_conn.capacity,
                                           conn_capacity2[_id],
                                           decimal=2)
        except KeyError:
            print('conn #{} is not found'.format(_id))
        except AssertionError:
            print('conn #{} fails at {} not {}'.format(
                _id, _conn.capacity, conn_capacity2[_id]))


class TestScenario1(unittest.TestCase):
    """
    validate computed loads at selected connections
    """

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario1/test_scenario1.cfg'))

        cls.house_damage = HouseDamage(cfg, seed=0)

        # # set up logging
        # file_logger = os.path.join(cfg.path_output, 'log_test1.txt')
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
            _zone.cpe_str = _zone.cpe_str_mean[0]
            _zone.cpe_eave = _zone.cpe_eave_mean[0]
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
                    ref_load[_conn.name], _conn.load, _conn.name))


class TestScenario2(unittest.TestCase):
    """
    validate the sequence of sheeting failures
    """

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario2/test_scenario2.cfg'))

        cls.house_damage = HouseDamage(cfg, seed=0)

        # # set up logging
        # file_logger = os.path.join(cfg.path_output, 'log_test2.txt')
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
                         41.0: [9, 11],
                         42.0: [8],
                         43.0: [12],
                         44.0: [7],
                         48.0: [13, 6],
                         49.0: [5, 14],
                         50.0: [4, 15],
                         51.0: [3, 16],
                         52.0: [2, 17],
                         53.0: [18, 1]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120, 1.0),
                   list_connections=range(1, 19))


class TestScenario3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario3/test_scenario3.cfg'))

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cfg.path_output, 'log_test3.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_batten(self):

        # ref data
        conn_capacity = {40.0: [40],
                         41.0: [43],
                         49.0: [49],
                         50.0: [37],
                         51.0: [55],
                         52.0: [31],
                         53.0: [46],
                         54.0: [34],
                         55.0: [52],
                         56.0: [58],
                         60.0: [36],
                         61.0: [42],
                         62.0: [48],
                         63.0: [54],
                         64.0: [60],
                         72.0: [39, 41, 44],
                         73.0: [35, 50],
                         74.0: [38, 47, 56],
                         75.0: [32, 53],
                         76.0: [59],
                         93.0: [45],
                         94.0: [33, 51],
                         95.0: [57]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120, 1.0),
                   list_connections=range(1, 61))


class TestScenario4(unittest.TestCase):


    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario4/test_scenario4.cfg'))

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cfg.path_output, 'log_test4.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting_batten(self):

        # ref data
        conn_capacity = {40.0: [10],
                         41.0: [9, 11, 31],
                         42.0: [8],
                         46.0: [13],
                         48.0: [6],
                         62.0: [14],
                         63.0: [15],
                         64.0: [16],
                         65.0: [17],
                         67.0: [5],
                         68.0: [4],
                         69.0: [3],
                         70.0: [2]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120, 1.0),
                   list_connections=range(1, 37))


class TestScenario5(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario5/test_scenario5.cfg'))

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cfg.path_output, 'log_test5.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting_batten(self):

        # ref data 5
        conn_capacity = {40.0: [28],
                         41.0: [31],
                         60.0: [24],
                         72.0: [27, 29, 32],
                         102.0: [23]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120, 1.0),
                   list_connections=range(1, 37))


class TestScenario6(unittest.TestCase):
    """
    FIXME!! NEED TO CHECK THE RESULTS
    """

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario6/test_scenario6.cfg'))
        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cfg.path_output, 'log_test6.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting_batten_rafter(self):

        conn_capacity = {40.0: [41, 43],
                         49.0: [46],
                         50.0: [14, 15, 16, 17],
                         64.0: [18],
                         70.0: [31],
                         71.0: [13],
                         72.0: [8, 9, 10, 11],
                         92.0: [30],
                         93.0: [12]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 100.0, 1.0),
                   list_connections=range(1, 48))


class TestScenario7(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario7/test_scenario7.cfg'))

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cfg.path_output, 'log_test7.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting(self):

        # ref data 7
        conn_capacity = {48.0: [10],
                         49.0: [9],
                         50.0: [8],
                         51.0: [7],
                         52.0: [6],
                         53.0: [5],
                         54.0: [4],
                         55.0: [3],
                         56.0: [2],
                         57.0: [1]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120, 1.0),
                   list_connections=range(1, 11))


class TestScenario8(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario8/test_scenario8.cfg'))

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cfg.path_output, 'log_test8.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting(self):

        # ref data
        conn_capacity = {53.0: [1],
                         54.0: [2],
                         55.0: [3],
                         56.0: [4, 6],
                         57.0: [5],
                         58.0: [7],
                         59.0: [8],
                         60.0: [9],
                         61.0: [10]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120, 1.0),
                   list_connections=range(1, 11))


class TestScenario9(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario9/test_scenario9.cfg'))

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cfg.path_output, 'log_test9.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting(self):

        # ref data 9
        conn_capacity = {41.0: [7],
                         42.0: [6],
                         43.0: [8],
                         44.0: [5, 9],
                         45.0: [4, 10],
                         46.0: [3],
                         47.0: [2],
                         48.0: [1]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 60.0, 1.0),
                   list_connections=range(1, 11))


class TestScenario10(unittest.TestCase):
    """
    FIXME!! NEED TO CHECK THE RESULTS
    """

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario10/test_scenario10.cfg'))

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cfg.path_output, 'log_test10.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting(self):

        # ref data 9
        conn_capacity = {41.0: [8, 3],
                         42.0: [9],
                         43.0: [7],
                         44.0: [6],
                         45.0: [10],
                         46.0: [2, 4, 5],
                         47.0: [1]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 60.0, 1.0),
                   list_connections=range(1, 11))


class TestScenario11(unittest.TestCase):
    """
    FIXME!! NEED TO CHECK THE RESULTS
    """

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario11/test_scenario11.cfg'))

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cfg.path_output, 'log_test11.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_batten(self):

        # ref data 11
        conn_capacity = {43.0: [17],
                         44.0: [16],
                         45.0: [18],
                         46.0: [15],
                         47.0: [19],
                         48.0: [14, 20],
                         49.0: [13],
                         50.0: [12],
                         51.0: [11]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120.0, 1.0),
                   list_connections=range(1, 21))


class TestScenario12(unittest.TestCase):
    """
    FIXME!! NEED TO CHECK THE RESULTS
    """

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario12/test_scenario12.cfg'))

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cfg.path_output, 'log_test12.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_batten(self):

        # ref data 11
        conn_capacity = {40.0: [7],
                         53.0: [16],
                         63.0: [15],
                         64.0: [17],
                         65.0: [14, 18],
                         66.0: [19, 13],
                         67.0: [12, 20],
                         68.0: [11]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120.0, 1.0),
                   list_connections=range(1, 21))


class TestScenario13(unittest.TestCase):
    """
    FIXME!! NEED TO CHECK THE RESULTS
    """

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario13/test_scenario13.cfg'))

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cfg.path_output, 'log_test13.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_batten(self):

        # ref data 11
        conn_capacity = {40.0: [17],
                         46.0: [16],
                         47.0: [18],
                         48.0: [15, 19],
                         49.0: [14, 20],
                         50.0: [13],
                         51.0: [12],
                         52.0: [11]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120, 1.0),
                   list_connections=range(1, 21))


class TestScenario14(unittest.TestCase):
    """
    FIXME!! NEED TO CHECK THE RESULTS
    """

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario14/test_scenario14.cfg'))

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cfg.path_output, 'log_test14.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_batten(self):

        # ref data 11
        conn_capacity = {43.0: [17],
                         44.0: [16],
                         45.0: [18],
                         53.0: [19],
                         54.0: [15],
                         55.0: [20, 14],
                         56.0: [13],
                         57.0: [12],
                         58.0: [11]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 120, 1.0),
                   list_connections=range(1, 21))


class TestScenario15(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario15/test_scenario15.cfg'))
        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cfg.path_output, 'log_test15.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting_batten_rafter(self):

        conn_capacity = {68.0: [25, 27],
                         81.0: [2, 3, 4, 5, 8, 9, 10, 11],
                         82.0: [1, 6, 7, 12],
                         88.0: [26]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 100.0, 1.0),
                   list_connections=range(1, 28))


class TestScenario16(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario16/test_scenario16.cfg'))
        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cfg.path_output, 'log_test16.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting_batten_rafter(self):

        conn_capacity = {73.0: [76],
                         74.0: [88],
                         75.0: [64, 100],
                         76.0: [112],
                         80.0: [14, 26, 38, 50],
                         81.0: [13, 15, 25, 27, 37, 39, 49, 51],
                         82.0: [16, 28, 40, 52],
                         83.0: [17, 29, 41, 53, 124, 127, 130, 133],
                         84.0: [18, 30, 42, 54, 126, 129, 132, 135]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(70.0, 101.0, 1.0),
                   list_connections=range(1, 136))


class TestScenario16p1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario16p1/test_scenario16p1.cfg'))
        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cfg.path_output, 'log_test16p1.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting_batten_rafter(self):

        conn_capacity = {73.0: [76],
                         74.0: [88],
                         75.0: [64, 100],
                         76.0: [112],
                         80.0: [14, 26, 38, 50],
                         81.0: [13, 15, 25, 27, 37, 39, 49, 51, 33],
                         82.0: [16, 28, 40, 52],
                         83.0: [17, 29, 41, 53, 124, 127, 130, 133],
                         84.0: [18, 30, 42, 54, 126, 129, 132, 135],
                         89.0: [32, 34],
                         90.0: [31, 35],
                         91.0: [36]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(70.0, 101.0, 1.0),
                   list_connections=range(1, 136))


class TestScenario17(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario17/test_scenario17.cfg'))
        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cfg.path_output, 'log_test17.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting_batten_rafter(self):

        conn_capacity = {61.0: [64, 87, 100],
                         62.0: [8, 31, 44],
                         63.0: [7],
                         64.0: [9],
                         65.0: [86, 88],
                         66.0: [32, 30],
                         67.0: [29, 75, 105],
                         68.0: [101],
                         69.0: [109],
                         70.0: [98],
                         74.0: [45],
                         75.0: [43],
                         76.0: [46, 59, 69, 77],
                         77.0: [13, 3, 21],
                         78.0: [14, 22, 4],
                         79.0: [15, 23],
                         80.0: [16, 24],
                         83.0: [19, 49, 133, 135],
                         84.0: [134, 136],
                         90.0: [18, 48],
                         91.0: [20, 50],
                         92.0: [17, 47]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(55.0, 101.0, 1.0),
                   list_connections=range(1, 137))


class TestScenario18(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario18/test_scenario18.cfg'))
        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cfg.path_output, 'log_test18.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.INFO,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting_batten_rafter(self):

        conn_capacity = {70.0: [78],
                         71.0: [22],
                         72.0: [21, 23],
                         73.0: [24],
                         76.0: [69, 59, 63, 85],
                         77.0: [3, 7, 13, 29],
                         78.0: [4, 8, 14, 30],
                         79.0: [9, 15, 31],
                         80.0: [16, 32],
                         83.0: [133, 135],
                         84.0: [134, 136]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(55.0, 101.0, 1.0),
                   list_connections=range(1, 137))

if __name__ == '__main__':
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestScenario3)
    #unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main(verbosity=2)
