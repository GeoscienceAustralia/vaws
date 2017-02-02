#!/usr/bin/env python
from __future__ import print_function

import unittest
import os
import numpy as np

from core.simulation import HouseDamage
from core.scenario import Scenario
import logging


class TestScenario1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, 'scenarios/test_scenario1.cfg'),
            output_path=cls.path_output)

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cls.path_output, 'log_test1.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_conn_load(self):

        # compute zone pressures
        cpi = 0.0
        wind_dir_index = 3
        Ms = 1.0
        building_spacing = 0
        qz = 0.24

        for _zone in self.house_damage.house.zones.itervalues():

            # _zone.cpe = -0.1
            _zone.cpi_alpha = 0.0
            _zone.cpe = _zone.cpe_mean[0]

            _zone.calc_zone_pressures(wind_dir_index,
                                      cpi,
                                      qz,
                                      Ms,
                                      building_spacing)

        ref_load = {1: -0.0049, 21: -0.0194, 31: -0.0049,  35: -0.0972,
                    39: -0.1507, 45: -0.0632, 51: -0.0194, 60: -0.0097}

        for _conn in self.house_damage.house.connections.itervalues():

            _conn.dead_load = 0.000
            _conn.cal_load()

            try:
                self.assertAlmostEqual(ref_load[_conn.id], _conn.load, places=3)
            except KeyError:
                pass
            except AssertionError:
                print('{} is different from {} for conn #{}'.format(
                    ref_load[_conn.id], _conn.load, _conn.id))


class TestScenario2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, 'scenarios/test_scenario2.cfg'),
            output_path=cls.path_output)

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cls.path_output, 'log_test2.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

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
        wind_speeds = np.arange(20.0, 115.0, 5.0)
        self.house_damage.house.mzcat = 1.0

        # set dead_load 0.0
        for _conn in self.house_damage.house.connections.itervalues():
            _conn.dead_load = 0.0

        for wind_speed in wind_speeds:

            self.house_damage.calculate_qz_Ms(wind_speed)

            for _zone in self.house_damage.house.zones.itervalues():

                # _zone.cpe = -0.1
                _zone.cpi_alpha = 0.0
                _zone.cpe = _zone.cpe_mean[0]

                _zone.calc_zone_pressures(wind_dir_index,
                                          cpi,
                                          self.house_damage.qz,
                                          Ms,
                                          building_spacing)

            # check damage by connection type group
            for _group in self.house_damage.house.groups.itervalues():

                _group.check_damage(wind_speed)
                _group.cal_prop_damaged()
                _group.distribute_damage()

        # compare with reference capacity
        for _id, _conn in self.house_damage.house.connections.iteritems():

            try:
                self.assertAlmostEqual(_conn.failure_v_raw,
                                       conn_capacity2[_id],
                                       places=4)
            except KeyError:
                print('conn #{} is not found'.format(_id))
            except AssertionError:
                print('conn #{}: {} is different from {}'.format(
                    _id, _conn.failure_v_raw, conn_capacity2[_id]))


class TestScenario3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, 'scenarios/test_scenario3.cfg'),
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
        wind_speeds = np.arange(40.0, 120.0, 5.0)
        self.house_damage.house.mzcat = 1.0

        # set dead_load 0.0
        for _conn in self.house_damage.house.connections.itervalues():
            _conn.dead_load = 0.0

        for wind_speed in wind_speeds:

            self.house_damage.calculate_qz_Ms(wind_speed)

            for _zone in self.house_damage.house.zones.itervalues():

                # _zone.cpe = -0.1
                _zone.cpi_alpha = 0.0
                _zone.cpe = _zone.cpe_mean[0]

                _zone.calc_zone_pressures(wind_dir_index,
                                          cpi,
                                          self.house_damage.qz,
                                          Ms,
                                          building_spacing)

            # check damage by connection type group
            for _group in self.house_damage.house.groups.itervalues():

                _group.check_damage(wind_speed)
                _group.cal_prop_damaged()
                _group.distribute_damage()

        # compare with reference capacity
        for _id, _conn in self.house_damage.house.connections.iteritems():

            try:
                self.assertAlmostEqual(_conn.failure_v_raw,
                                       conn_capacity2[_id],
                                       places=4)
            except KeyError:
                print('conn #{} is not found'.format(_id))
            except AssertionError:
                print('conn #{}: {} is different from {}'.format(
                    _id, _conn.failure_v_raw, conn_capacity2[_id]))


class TestScenario4(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, 'scenarios/test_scenario4.cfg'),
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
        wind_speeds = np.arange(40.0, 120.0, 5.0)
        self.house_damage.house.mzcat = 1.0

        # set dead_load 0.0
        for _conn in self.house_damage.house.connections.itervalues():
            _conn.dead_load = 0.0

        for wind_speed in wind_speeds:

            self.house_damage.calculate_qz_Ms(wind_speed)

            for _zone in self.house_damage.house.zones.itervalues():

                # _zone.cpe = -0.1
                _zone.cpi_alpha = 0.0
                _zone.cpe = _zone.cpe_mean[0]

                _zone.calc_zone_pressures(wind_dir_index,
                                          cpi,
                                          self.house_damage.qz,
                                          Ms,
                                          building_spacing)

            # check damage by connection type group
            for _group in self.house_damage.house.groups.itervalues():

                _group.check_damage(wind_speed)
                _group.cal_prop_damaged()

                _group.distribute_damage()

        # compare with reference capacity
        for _id, _conn in self.house_damage.house.connections.iteritems():

            try:
                self.assertAlmostEqual(_conn.failure_v_raw,
                                       conn_capacity2[_id],
                                       places=4)
            except KeyError:
                print('conn #{} is not found'.format(_id))
            except AssertionError:
                print('conn #{}: {} is different from {}'.format(
                    _id, _conn.failure_v_raw, conn_capacity2[_id]))


class TestScenario5(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, 'scenarios/test_scenario5.cfg'),
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
        wind_speeds = np.arange(40.0, 120.0, 5.0)
        self.house_damage.house.mzcat = 1.0

        # set dead_load 0.0
        for _conn in self.house_damage.house.connections.itervalues():
            _conn.dead_load = 0.0

        for wind_speed in wind_speeds:

            self.house_damage.calculate_qz_Ms(wind_speed)

            for _zone in self.house_damage.house.zones.itervalues():

                # _zone.cpe = -0.1
                _zone.cpi_alpha = 0.0
                _zone.cpe = _zone.cpe_mean[0]

                _zone.calc_zone_pressures(wind_dir_index,
                                          cpi,
                                          self.house_damage.qz,
                                          Ms,
                                          building_spacing)

            # check damage by connection type group
            for _group in self.house_damage.house.groups.itervalues():

                _group.check_damage(wind_speed)
                _group.cal_prop_damaged()
                _group.distribute_damage()

        # compare with reference capacity
        for _id, _conn in self.house_damage.house.connections.iteritems():

            try:
                self.assertAlmostEqual(_conn.failure_v_raw,
                                       conn_capacity2[_id],
                                       places=4)
            except KeyError:
                print('conn #{} is not found'.format(_id))
            except AssertionError:
                print('conn #{}: {} is different from {}'.format(
                    _id, _conn.failure_v_raw, conn_capacity2[_id]))


if __name__ == '__main__':
    unittest.main()
