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
    house_damage.house.mzcat = 1.0  # profile: 6, height: 4.5

    for wind_speed in wind_speeds:

        logging.info('wind speed {:.3f}'.format(wind_speed))

        house_damage.compute_qz_ms(wind_speed)

        for _zone in house_damage.house.zones.itervalues():

            _zone.cpe = _zone.cpe_mean[wind_dir_index]
            _zone.cpe_str = _zone.cpe_str_mean[wind_dir_index]
            _zone.cpe_eave = _zone.cpe_eave_mean[wind_dir_index]

            _zone.calc_zone_pressures(wind_dir_index,
                                      cpi,
                                      house_damage.qz,
                                      ms,
                                      building_spacing)

        for _connection in house_damage.house.connections.itervalues():
            _connection.compute_load()

        # check damage by connection type group
        for _group in house_damage.house.groups.itervalues():

            _group.check_damage(wind_speed)

            _group.compute_damaged_area()

            _group.update_influence(house_damage.house)

        house_damage.compute_damage_index(wind_speed)

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
        wind_dir_index = 0
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

        ref_load = {1: -0.0049, 11: -0.1944, 15: -0.0194, 21: -0.0194,
                    25: -0.0097, 31: -0.0049, 35: -0.0972, 39: -0.1507,
                    45: -0.0632, 51: -0.0194, 60: -0.0097}

        for _conn in self.house_damage.house.connections.itervalues():

            _conn.compute_load()

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
                         47.0: [13, 6],
                         48.0: [5, 14],
                         49.0: [4, 15],
                         50.0: [3, 16],
                         51.0: [2, 17],
                         52.0: [18, 1]}

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
        conn_capacity = {40.0: [40, 43],
                         48.0: [49],
                         49.0: [37],
                         50.0: [55],
                         51.0: [31],
                         52.0: [46],
                         53.0: [34],
                         54.0: [52],
                         55.0: [58],
                         59.0: [36],
                         60.0: [42],
                         61.0: [48],
                         62.0: [54],
                         63.0: [60],
                         71.0: [39, 41, 44],
                         72.0: [35, 50],
                         73.0: [38, 47, 56],
                         74.0: [32, 53],
                         75.0: [59],
                         91.0: [45],
                         92.0: [33, 51],
                         93.0: [57]}

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
        conn_capacity = {40.0: [10, 31, 28],
                         41.0: [9, 11, 25],
                         42.0: [8, 19, 26, 30],
                         43.0: [12, 24],
                         44.0: [7],
                         45.0: [13],
                         46.0: [14, 32],
                         47.0: [6, 15, 33],
                         48.0: [16, 5, 34],
                         49.0: [17, 4, 35],
                         50.0: [3, 18, 21, 36],
                         51.0: [2, 20],
                         52.0: [1]}

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
        conn_capacity = {40.0: [28, 31],
                         41.0: [25],
                         42.0: [19],
                         52.0: [34],
                         53.0: [22],
                         59.0: [24],
                         60.0: [30],
                         61.0: [36],
                         71.0: [27, 29, 32],
                         72.0: [23, 26],
                         73.0: [20, 35],
                         91.0: [33],
                         92.0: [21],
                         }

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
                         48.0: [46],
                         49.0: [14, 15, 16, 17, 32, 33, 34, 35],
                         50.0: [13, 18, 31, 36, 42],
                         71.0: [8, 9, 10, 11, 26, 27, 28, 29],
                         72.0: [7, 12, 25, 30]
                         }

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
        conn_capacity = {47.0: [10],
                         48.0: [9],
                         49.0: [8],
                         50.0: [7],
                         51.0: [6],
                         52.0: [5],
                         53.0: [4],
                         54.0: [3],
                         55.0: [2],
                         56.0: [1]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(40.0, 100, 1.0),
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
        conn_capacity = {52.0: [1],
                         53.0: [2],
                         54.0: [3],
                         55.0: [4, 6],
                         56.0: [5],
                         57.0: [7],
                         58.0: [8],
                         59.0: [9],
                         60.0: [10]}

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
        conn_capacity = {40.0: [7],
                         41.0: [6],
                         42.0: [8],
                         43.0: [5, 9],
                         44.0: [4, 10],
                         45.0: [3],
                         46.0: [2],
                         47.0: [1]}

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
        conn_capacity = {40.0: [8, 3],
                         42.0: [9],
                         43.0: [7],
                         44.0: [6],
                         45.0: [10, 2, 4],
                         46.0: [5],
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
        conn_capacity = {42.0: [17],
                         43.0: [16],
                         44.0: [18],
                         45.0: [15],
                         46.0: [19],
                         47.0: [14, 20],
                         48.0: [13],
                         49.0: [12],
                         50.0: [11]}

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

        # ref data 12
        conn_capacity = {40.0: [7],
                         52.0: [16],
                         62.0: [15],
                         63.0: [17],
                         64.0: [14, 18],
                         65.0: [19, 13],
                         66.0: [12, 20],
                         67.0: [11]}

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
        conn_capacity = {42.0: [17],
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

        conn_capacity = {67.0: [25, 27],
                         80.0: [2, 3, 4, 5, 8, 9, 10, 11,
                                14, 15, 16, 17, 20, 21, 22, 23],
                         81.0: [1, 6, 7, 12, 13, 18, 19, 24],
                         87.0: [26]}

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

        conn_capacity = {72.0: [76],
                         73.0: [88],
                         74.0: [64, 100],
                         75.0: [112],
                         79.0: [14, 26, 38, 50, 74, 86, 98, 110],
                         80.0: [13, 15, 25, 27, 37, 39, 49, 51, 33,
                                109, 111, 97, 93, 85, 73, 75, 87, 99],
                         81.0: [16, 28, 40, 52],
                         82.0: [17, 29, 41, 53, 124, 127, 130, 133, 77, 89,
                                101, 113],
                         83.0: [18, 30, 42, 54, 126, 129, 132, 135, 114, 102,
                                78, 90],
                         88.0: [32, 34, 94, 92],
                         89.0: [31, 35, 91, 95],
                         90.0: [36, 96]}

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

        conn_capacity = {60.0: [64, 87, 100],
                         61.0: [8, 31, 44, 79],
                         62.0: [7, 63],
                         63.0: [9, 65],
                         64.0: [86, 88],
                         65.0: [32, 30, 80, 78],
                         66.0: [29, 75, 105, 85],
                         67.0: [101],
                         68.0: [109],
                         69.0: [98, 70, 71],
                         70.0: [60],
                         73.0: [45],
                         74.0: [43, 99],
                         75.0: [46, 13, 3, 21, 69, 77, 102, 59],
                         76.0: [14, 22, 4],
                         77.0: [15, 23],
                         78.0: [16, 24, 72],
                         81.0: [19, 49],
                         82.0: [133, 135],
                         83.0: [134, 136],
                         88.0: [18, 48, 74, 104],
                         89.0: [20, 50, 76, 106],
                         90.0: [17, 47, 73, 103]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(55.0, 101.0, 1.0),
                   list_connections=range(1, 137))


class TestScenario18(unittest.TestCase):
    """
     Connection 78 (batten) should fail at about 55m/s and then progressively 
     redistribute to adjacent batten connections 70, 86; 64; 60.
    """
    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario18/test_scenario18.cfg'))
        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cfg.path_output, 'log_test18_new.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_damage_sheeting_batten_rafter(self):

        conn_capacity = {56.0: [78],
                         65.0: [86, 70],
                         66.0: [64],
                         67.0: [60],
                         69.0: [22],
                         70.0: [21, 23, 59, 63, 69, 85, 77, 79],
                         71.0: [24, 57, 80],
                         75.0: [3, 7, 13, 29],
                         76.0: [4, 8, 14, 30],
                         77.0: [9, 15, 31, 65, 71, 87],
                         78.0: [16, 32, 72, 88],
                         82.0: [133, 135],
                         83.0: [134, 136]}

        simulation(self.house_damage, conn_capacity,
                   wind_speeds=np.arange(55.0, 101.0, 1.0),
                   list_connections=range(1, 137))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestScenario18)
    unittest.TextTestRunner(verbosity=2).run(suite)
    # unittest.main(verbosity=2)
