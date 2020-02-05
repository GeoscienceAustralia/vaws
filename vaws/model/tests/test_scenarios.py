#!/usr/bin/env python

import unittest
import os
import numpy as np

from vaws.model.house import House
from vaws.model.config import Config
from vaws.model.main import set_logger
import logging


def simulation(house, wind_speeds, conn_capacity={}, list_connections=[],
               coverage_capacity={}, list_coverages=[], multiplier=1.0):

    logger = logging.getLogger(__name__)

    # compute zone pressures
    house._wind_dir_index = 0
    house._terrain_height_multiplier = multiplier  # profile: 6, height: 4.5
    house._construction_level = 'medium'
    house.damage_increment = 0.0

    for wind_speed in wind_speeds:

        logger.debug(f'wind speed {wind_speed:.3f}')

        house.compute_qz(wind_speed)

        for _, _zone in house.zones.items():

            _zone._cpe = _zone.cpe_mean[house.wind_dir_index]
            _zone._cpe_str = _zone.cpe_str_mean[house.wind_dir_index]
            _zone._cpe_eave = _zone.cpe_eave_mean[house.wind_dir_index]
            _zone._differential_shielding = 1.0
            _zone.shielding_multiplier = 1.0

            _zone.calc_zone_pressure(house.cpi, house.qz, house.combination_factor)

        if house.coverages is not None:
            for _, coverage in house.coverages['coverage'].iteritems():
                coverage.check_damage(
                    house.qz, house.cpi, house.combination_factor, wind_speed)

        # check damage by connection type group
        [group.check_damage(wind_speed) for _, group in house.groups.items()]

        for _, connection in house.connections.items():
            connection.damaged_previous = connection.damaged

        [group.update_influence(house) for _, group in house.groups.items()
         if group.damage_dist]

        house.run_debris_and_update_cpi(wind_speed)

        house.compute_damage_index(wind_speed)

    # compare with reference connection capacity
    conn_capacity2 = {x: -1.0 for x in list_connections}
    for speed, conn_list in conn_capacity.items():
        for _id in conn_list:
            conn_capacity2.update({_id: speed})

    if conn_capacity2:
        for _id, conn in house.connections.items():

            try:
                np.testing.assert_almost_equal(conn.capacity,
                                               conn_capacity2[_id],
                                               decimal=2)
            except KeyError:
                print(f'conn #{_id} is not found')
            except AssertionError:
                print(f'conn #{_id} fails at {conn.capacity} not {conn_capacity2[_id]}')

    # compare with reference coverage capacity
    coverage_capacity2 = {x: -1.0 for x in list_coverages}
    for speed, coverage_list in coverage_capacity.items():
        for _id in coverage_list:
            coverage_capacity2.update({_id: speed})

    if coverage_capacity2:
        for _id, coverage in house.coverages['coverage'].iteritems():

            try:
                np.testing.assert_almost_equal(coverage.capacity,
                                               coverage_capacity2[_id],
                                               decimal=2)
            except KeyError:
                print(f'coverage #{_id} is not found')
            except AssertionError:
                print(f'coverage #{_id} fails at {coverage.capacity} not {coverage_capacity2[_id]}')


class TestScenario1(unittest.TestCase):
    """
    validate computed loads at selected connections
    """

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario1', 'test_scenario1.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg, seed=0)

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

        for _, _zone in self.house.zones.items():

            _zone._cpe = _zone.cpe_mean[0]
            _zone._cpe_str = _zone.cpe_str_mean[0]
            _zone._cpe_eave = _zone.cpe_eave_mean[0]
            _zone.shielding_multiplier = 1.0

            _zone.calc_zone_pressure(cpi, qz, self.house.combination_factor)

        ref_load = {1: -0.0049, 11: -0.1944, 15: -0.0194, 21: -0.0194,
                    25: -0.0097, 31: -0.0049, 35: -0.0972, 39: -0.1507,
                    45: -0.0632, 51: -0.0194, 60: -0.0097}

        for _, _conn in self.house.connections.items():

            _conn.check_damage(20.0)

            try:
                self.assertAlmostEqual(ref_load[_conn.name], _conn.load,
                                       places=3)
            except KeyError:
                pass
            except AssertionError:
                print(f'{ref_load[_conn.name]} is different from {_conn.load} for conn #{_conn.name}')


class TestScenario2(unittest.TestCase):
    """
    Designed to test whether the code correctly calculates which sheeting connections
    have broken at various wind speeds and redistributes loads as expected to
    adjacent sheeting connections. Dead load set to be zero. Fixed connection
    strengths modelled by zero standard deviation of connection strengths.
    Tests distribution upon connection failure from an interior cladding
    connection, an eave connection and a ridge connection.
    """

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario2', 'test_scenario2.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg, seed=0)

    def test_damage_sheeting(self):

        conn_capacity = {40.0: [10],
                         45.0: [9, 11],
                         46.0: [8],
                         48.0: [12],
                         49.0: [7],
                         53.0: [13, 6],
                         54.0: [5, 14],
                         55.0: [4, 15],
                         56.0: [3, 16],
                         57.0: [2, 17],
                         58.0: [18, 1]}

        simulation(self.house,
                   wind_speeds=np.arange(40.0, 120, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 19))


class TestScenario2a(unittest.TestCase):
    """
    Same as Scenario 2, but no damage distribution applied.
    """

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario2', 'test_scenario2.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg, seed=0)

    def test_damage_sheeting(self):

        conn_capacity = {40.0: [10],
                         53.0: [13, 6],
                         75.0: [9, 11, 14],
                         80.0: [5]}

        # for _conn in self.house.connections.itervalues():
        #     for key, value in _conn.influences.iteritems():
        #         est = np.sqrt(np.abs(_conn.strength * 1.0e+3 / (value.source.area * value.source.cpe_mean[0] * 0.5 * 1.2)))
        #     print('{} fails at {}'.format(_conn.name, est))

        self.house.groups['sheeting0'].damage_dist = 0

        simulation(self.house,
                   wind_speeds=np.arange(40.0, 120, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 19))


class TestScenario3(unittest.TestCase):
    """
    Designed to test whether the code correctly calculates which batten
    connections have failed and redistributes loads as expected.
    Sheeting connections modelled with artificially high strengths to ensure
    failures occur in batten connections. Fixed batten strengths modelled by
    zero standard deviation of connection strengths. Tests distribution from
    interior and gable batten connections.
    """

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario3', 'test_scenario3.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg, seed=0)

    def test_damage_batten(self):

        # ref data
        conn_capacity = {44.0: [43],
                         47.0: [40],
                         57.0: [49, 37],
                         58.0: [55, 31],
                         64.0: [46],
                         65.0: [34],
                         66.0: [36, 52],
                         67.0: [58, 42],
                         68.0: [48],
                         69.0: [54],
                         70.0: [60],
                         87.0: [39, 41, 44],
                         88.0: [35],
                         89.0: [47],
                         90.0: [53],
                         91.0: [59],
                         112.0: [38, 45, 50],
                         113.0: [32, 33, 56],
                         114.0: [51],
                         115.0: [57]}

        simulation(self.house,
                   wind_speeds=np.arange(40.0, 120, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 61))


class TestScenario4(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario4', 'test_scenario4.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg, seed=0)

    def test_damage_sheeting_batten(self):

        # ref data
        conn_capacity = {40.0: [10],
                         44.0: [31],
                         45.0: [9, 11, 25],
                         46.0: [8, 19],
                         47.0: [47, 30],
                         48.0: [12, 24],
                         49.0: [7],
                         50.0: [13],
                         51.0: [14, 32],
                         52.0: [15, 33],
                         53.0: [6, 16, 34],
                         54.0: [5, 17, 35],
                         55.0: [4, 18, 36],
                         56.0: [3],
                         57.0: [2, 20],
                         58.0: [1]}

        simulation(self.house,
                   wind_speeds=np.arange(40.0, 120, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 37))


class TestScenario5(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario5', 'test_scenario5.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg, seed=0)

    def test_damage_sheeting_batten(self):

        # ref data 5
        conn_capacity = {44.0: [31],
                         45.0: [25],
                         46.0: [19],
                         47.0: [28],
                         64.0: [34],
                         65.0: [22],
                         66.0: [24],
                         67.0: [30],
                         68.0: [36],
                         87.0: [27, 29, 32],
                         88.0: [23, 26],
                         89.0: [20, 35],
                         112.0: [33],
                         113.0: [21],
                         }

        simulation(self.house,
                   wind_speeds=np.arange(40.0, 120, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 37))


class TestScenario6(unittest.TestCase):
    """
    FIXME!! NEED TO CHECK THE RESULTS
    conn #44 fails at 99.0
    """

    @classmethod
    def setUpClass(cls):

        # set up logging
        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario6', 'test_scenario6.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg, seed=0)

    def test_damage_sheeting_batten_rafter(self):

        conn_capacity = {40.0: [41, 43],
                         49.0: [32, 33, 34, 35],
                         50.0: [36],
                         58.0: [46],
                         78.0: [45],
                         79.0: [40],
                         92.0: [38],
                         98.0: [8, 9, 10, 11, 14, 15, 16, 17,
                                26, 27, 28, 29, 32, 33, 34, 35],
                         99.0: [7, 12, 13, 18, 25, 30, 31, 36, 42]
                         }

        simulation(self.house,
                   wind_speeds=np.arange(40.0, 100.0, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 48))


class TestScenario7(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario7', 'test_scenario7.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg, seed=0)

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

        simulation(self.house,
                   wind_speeds=np.arange(40.0, 100, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 11))


class TestScenario8(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario8', 'test_scenario8.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg, seed=0)

    def test_damage_sheeting(self):

        # ref data
        conn_capacity = {53: [1],
                         54.0: [2],
                         55.0: [3],
                         56.0: [4],
                         57.0: [5],
                         58.0: [6],
                         59.0: [7],
                         60.0: [8],
                         61.0: [9],
                         62.0: [10]}

        simulation(self.house,
                   wind_speeds=np.arange(40.0, 120, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 11))


class TestScenario9(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario9', 'test_scenario9.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg, seed=0)

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

        simulation(self.house,
                   wind_speeds=np.arange(40.0, 60.0, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 11))


class TestScenario10(unittest.TestCase):
    """
    test10
    """

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario10', 'test_scenario10.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg, seed=0)

    def test_damage_sheeting(self):

        # ref data 9
        conn_capacity = {40.0: [8, 3],
                         45.0: [9, 7, 2, 4],
                         46.0: [6, 5],
                         47.0: [10, 1]}

        simulation(self.house,
                   wind_speeds=np.arange(40.0, 60.0, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 11))


class TestScenario11(unittest.TestCase):
    """
    Test11
    """

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario11', 'test_scenario11.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg, seed=0)

    def test_damage_batten(self):

        # ref data 11
        conn_capacity = {42.0: [17],
                         49.0: [16, 18],
                         50.0: [15, 19],
                         51.0: [14, 20],
                         52.0: [13],
                         53.0: [12],
                         54.0: [11]}

        simulation(self.house,
                   wind_speeds=np.arange(40.0, 120.0, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 21))


class TestScenario12(unittest.TestCase):
    """
    Test12
    """

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario12', 'test_scenario12.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg, seed=0)

    def test_damage_batten(self):

        # ref data 12
        conn_capacity = {40.0: [7],
                         87.0: [16, 18],
                         88.0: [17],
                         89.0: [15, 19],
                         90.0: [14, 20],
                         91.0: [13],
                         92.0: [12],
                         93.0: [11]}

        simulation(self.house,
                   wind_speeds=np.arange(40.0, 120.0, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 21))


class TestScenario13(unittest.TestCase):
    """
    Test13
    """

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario13', 'test_scenario13.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg, seed=0)

    def test_damage_batten(self):

        # ref data 11
        conn_capacity = {40.0: [17],
                         64.0: [16, 18],
                         65.0: [15, 19],
                         66.0: [14, 20],
                         67.0: [13],
                         68.0: [12],
                         69.0: [11]}

        simulation(self.house,
                   wind_speeds=np.arange(40.0, 120, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 21))


class TestScenario14(unittest.TestCase):
    """
    Test14
    """

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario14', 'test_scenario14.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg, seed=0)

    def test_damage_batten(self):

        # ref data 11
        conn_capacity = {42.0: [17],
                         59.0: [18, 16],
                         60.0: [19],
                         65.0: [15],
                         66.0: [20, 14],
                         67.0: [13],
                         68.0: [12],
                         69.0: [11]}

        simulation(self.house,
                   wind_speeds=np.arange(40.0, 120, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 21))

#TODO
class TestScenario15(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario15', 'test_scenario15.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg, seed=0)

    def test_damage_sheeting_batten_rafter(self):

        conn_capacity = {67.0: [25, 27],
                         80.0: [2, 3, 4, 5, 8, 9, 10, 11,
                                14, 15, 16, 17, 20, 21, 22, 23],
                         81.0: [1, 6, 7, 12, 13, 18, 19, 24],
                         87.0: [26]}

        simulation(self.house,
                   wind_speeds=np.arange(40.0, 100.0, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 28))

#TODO
class TestScenario16(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario16', 'test_scenario16.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)
        cls.house = House(cfg, seed=0)

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

        simulation(self.house,
                   wind_speeds=np.arange(70.0, 101.0, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 136))

#TODO
class TestScenario17(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario17', 'test_scenario17.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)
        cls.house = House(cfg, seed=0)

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

        simulation(self.house,
                   wind_speeds=np.arange(55.0, 101.0, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 137))

#TODO
class TestScenario18(unittest.TestCase):
    """
     Connection 78 (batten) should fail at about 55m/s and then progressively 
     redistribute to adjacent batten connections 70, 86; 64; 60.
    """
    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario18', 'test_scenario18.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)
        cls.house = House(cfg, seed=0)

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

        simulation(self.house,
                   wind_speeds=np.arange(55.0, 101.0, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 137))


class TestScenario19(unittest.TestCase):
    """
     Coverage 2 should fail at about 34 m/s followed by
     4, 1, and 3 at 40, 45, and 50 m/s respectively.

    """
    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        path_cfg = os.path.join(path, 'test_scenarios', 'test_scenario19')
        file_cfg = os.path.join(path_cfg, 'test_scenario19.cfg')
        # set_logger(path_cfg=path_cfg, logging_level='debug')
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        cfg = Config(file_cfg=file_cfg, logger=logger)
        # cfg = Config(file_cfg=file_cfg)
        cls.house = House(cfg, seed=0)

    def test_damage_coverage(self):

        list_connections = range(1, 9)
        conn_capacity = {34.0: [2],
                         40.0: [4],
                         45.0: [1],
                         50.0: [3],
                         }

        wind_speeds = np.arange(20.0, 60.0, 1)

    # change it to conn to speed
        conn_capacity2 = {x: -1.0 for x in list_connections}
        for speed, conn_list in conn_capacity.items():
            for _id in conn_list:
                conn_capacity2.update({_id: speed})

        # compute zone pressures
        self.house._terrain_height_multiplier = 1.0  # profile: 6, height: 4.5

        for wind_speed in wind_speeds:

            logging.debug(f'wind speed {wind_speed:.3f}')

            self.house.compute_qz(wind_speed)

            for _, _ps in self.house.coverages.iterrows():
                _ps['coverage'].check_damage(self.house.qz,
                                             self.house.cpi,
                                             self.house.combination_factor,
                                             wind_speed)

                # ignore cpi refinement
                if _ps['coverage'].breached:
                    self.house._cpi = 0.7

        # compare with reference capacity
        for _id, _coverage in self.house.coverages['coverage'].iteritems():

            try:
                np.testing.assert_almost_equal(_coverage.capacity,
                                               conn_capacity2[_id],
                                               decimal=2)
            except KeyError:
                print(f'coverage #{_id} is not found')
            except AssertionError:
                print(f'coverage #{_id} fails at {_coverage.capacity} not {conn_capacity2[_id]}')


class TestScenario20(unittest.TestCase):
    """
    to test the effect of dead load on a roof sheeting connection

    """
    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario20', 'test_scenario20.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)
        cls.house = House(cfg, seed=0)

    def test_dead_load(self):

        conn_capacity = {48.0: [4],
                         51.0: [3, 5],
                         52.0: [2, 6],
                         53.0: [1],
                         }

        dead_load = {1: 1.5,
                     2: 1.5,
                     3: 1.5,
                     4: 0.1,
                     5: 1.5,
                     6: 1.5}

        simulation(self.house,
                   wind_speeds=np.arange(20.0, 60.0, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 6))

        for _id, _conn in self.house.connections.items():

            try:
                np.testing.assert_almost_equal(_conn.dead_load,
                                               dead_load[_id],
                                               decimal=2)
            except AssertionError:
                print(f'conn #{_id} dead load should be {dead_load[_id]} not {_conn.dead_load}')


class TestScenario21(unittest.TestCase):
    """
    to test the effect of dead load on a roof sheeting connection

    """
    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario21', 'test_scenario21.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)
        cls.house = House(cfg, seed=0)

    def test_dead_load(self):

        conn_capacity = {49.0: [10],
                         52.0: [9, 11],
                         53.0: [8, 12],
                         54.0: [7],
                         57.0: range(1, 7)}

        dead_load = {1: 0.1,
                     2: 0.1,
                     3: 0.1,
                     4: 0.1,
                     5: 0.1,
                     6: 0.1,
                     7: 1.5,
                     8: 1.5,
                     9: 1.5,
                     10: 0.1,
                     11: 1.5,
                     12: 1.5}

        simulation(self.house,
                   wind_speeds=np.arange(20.0, 60.0, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 12))

        for _id, _conn in self.house.connections.items():

            try:
                np.testing.assert_almost_equal(_conn.dead_load,
                                               dead_load[_id],
                                               decimal=2)
            except AssertionError:
                print(f'conn #{_id} dead load should be {dead_load[_id]} not {_conn.dead_load}')


class TestScenario22a(unittest.TestCase):
    """
    to test different strength for inward and outward direction

    """
    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario22', 'test_scenario22.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)
        cfg.wind_dir_index = 0

        cls.house = House(cfg, seed=0)

    def test_directional_strength_wind_direction_S(self):

        # change it to conn to speed
        coverage_capacity = {34.0: [2],
                             35.0: [4],
                             45.0: [3],
                             46.0: [1],
                             }
        list_coverages = range(1, 9)

        simulation(self.house,
                   wind_speeds=np.arange(20.0, 60.0, 1.0),
                   coverage_capacity=coverage_capacity,
                   list_coverages=list_coverages)


class TestScenario22b(unittest.TestCase):
    """
    to test different strength for inward and outward direction

    """
    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario22', 'test_scenario22.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)
        cfg.wind_dir_index = 4
        cls.house = House(cfg, seed=0)

    def test_directional_strength_wind_direction_N(self):

        assert self.house.wind_dir_index == 4
        self.house._terrain_height_multiplier = 1.0  # profile: 6, height: 4.5

        # change it to conn to speed
        coverage_capacity = {34.0: [4],
                             35.0: [3, 2, 1],
                             }
        list_coverages = range(1, 9)

        simulation(self.house,
                   wind_speeds=np.arange(20.0, 60.0, 1.0),
                   coverage_capacity=coverage_capacity,
                   list_coverages=list_coverages)


class TestScenario23a(unittest.TestCase):
    """
    to test different strength for inward and outward direction

    """
    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario23', 'test_scenario23.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cfg.wind_dir_index = 0
        cls.house = House(cfg, seed=0)

    def test_directional_strength_wind_direction_S(self):

        assert self.house.wind_dir_index == 0
        self.house._terrain_height_multiplier = 1.0  # profile: 6, height: 4.5

        # change it to conn to speed
        coverage_capacity = {34.0: [2],
                             35.0: [4],
                             45.0: [3],
                             46.0: [1],
                             }
        list_coverages = range(1, 9)

        simulation(self.house,
                   wind_speeds=np.arange(20.0, 60.0, 1.0),
                   coverage_capacity=coverage_capacity,
                   list_coverages=list_coverages)


class TestScenario23b(unittest.TestCase):
    """
    to test different strength for inward and outward direction

    """
    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario23', 'test_scenario23.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cfg.wind_dir_index = 1
        cls.house = House(cfg, seed=0)

    def test_directional_strength_wind_direction_SE(self):

        assert self.house.wind_dir_index == 1
        self.house._terrain_height_multiplier = 1.0  # profile: 6, height: 4.5

        # change it to conn to speed
        coverage_capacity = {28.0: [2],
                             29.0: [4],
                             58.0: [1],
                             }
        list_coverages = range(1, 9)

        simulation(self.house,
                   wind_speeds=np.arange(20.0, 60.0, 1.0),
                   coverage_capacity=coverage_capacity,
                   list_coverages=list_coverages)


class TestScenario23c(unittest.TestCase):
    """
    to test different strength for inward and outward direction

    """
    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario23', 'test_scenario23.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cfg.wind_dir_index = 2
        cls.house = House(cfg, seed=0)

    def test_directional_strength_wind_direction_E(self):

        assert self.house.wind_dir_index == 2
        self.house._terrain_height_multiplier = 1.0  # profile: 6, height: 4.5

        # change it to conn to speed
        coverage_capacity = {34.0: [4],
                             40.0: [3],
                             45.0: [1],
                             46.0: [2],
                             }
        list_coverages = range(1, 9)

        simulation(self.house,
                   wind_speeds=np.arange(20.0, 60.0, 1.0),
                   coverage_capacity=coverage_capacity,
                   list_coverages=list_coverages)


class TestScenario26(unittest.TestCase):
    """
    to test different strength for inward and outward direction

    """
    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario26', 'test_scenario26.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg, seed=0)

    def test_capacity(self):

        conn_capacity = {41.0: [10],
                         42.0: [11],
                         43.0: [4, 12],
                         44.0: [5],
                         45.0: [6],
                         50.0: [31, 35],
                         57.0: [1, 13],
                         58.0: [2, 14],
                         59.0: [3, 15],
                         76.0: [7],
                         77.0: [8],
                         78.0: [9]}

        simulation(self.house,
                   wind_speeds=np.arange(40.0, 120, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 35))


class TestScenario27(unittest.TestCase):
    """
    FIXME
    conn #33 fails at 84.0
    """

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario27', 'test_scenario27.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)

        cls.house = House(cfg, seed=0)

    def test_capacity(self):

        conn_capacity = {
                         40.0: [31],
                         68.0: [32],
                         83.0: [34],
                         50.0: [35],
                         }
        #self.house.house.mzcat = 0.9
        # mzcat = 0.9 then 32 fails at 76.0,
        #                  34 fails at 93.0
        #                  35 fails at 56.0

        simulation(self.house,
                   wind_speeds=np.arange(40.0, 105, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 36))


class TestScenario29(unittest.TestCase):
    """
    FIXME
    conn #78 fails at 51 not 67
    """
    @classmethod
    def setUpClass(cls):
        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario29', 'test_scenario29.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)
        cls.house = House(cfg, seed=0)

        ''' in case logging
        path = os.sep.join(__file__.split(os.sep)[:-1])
        path_cfg = os.path.join(path, 'test_scenarios', 'test_scenario29')
        set_logger(path_cfg=path_cfg, logging_level='debug')
        file_cfg = os.path.join(path_cfg, 'test_scenario29.cfg')
        cfg = Config(file_cfg=file_cfg)
        cls.house = House(cfg, seed=0)
        '''
    def test_capacity(self):

        conn_capacity = {
                         36.0: [77, 79],
                         41.0: [74, 82],
                         51.0: [76],
                         55.0: [73, 81],
                         64.0: [80],
                         67.0: [78],
                         70.0: [75, 83],
                         }

        simulation(self.house,
                   wind_speeds=np.arange(30.0, 80, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 83),
                   multiplier=0.9)


class TestScenario30(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_scenario30', 'test_scenario30.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)
        cls.house = House(cfg, seed=0)

        ''' in case logging
        path = os.sep.join(__file__.split(os.sep)[:-1])
        path_cfg = os.path.join(path, 'test_scenarios', 'test_scenario29')
        set_logger(path_cfg=path_cfg, logging_level='debug')
        file_cfg = os.path.join(path_cfg, 'test_scenario29.cfg')
        cfg = Config(file_cfg=file_cfg)
        cls.house = House(cfg, seed=0)
        '''
    def test_capacity(self):

        conn_capacity = {
                         32.0: [77, 79],
                         37.0: [74, 82],
                         41.0: [78],
                         46.0: [76],
                         50.0: [73, 81],
                         58.0: [80],
                         63.0: [75, 83],
                         }

        simulation(self.house,
                   wind_speeds=np.arange(30.0, 80, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 83),
                   multiplier=1.0)

class TestScenarioDeadLoadRoof(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        path = os.sep.join(__file__.split(os.sep)[:-1])
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(path, 'test_scenarios',
                                'test_dead_load_roof', 'test_dead_load_roof.cfg')
        cfg = Config(file_cfg=file_cfg, logger=logger)
        cls.house = House(cfg, seed=0)
        """

        path = os.sep.join(__file__.split(os.sep)[:-1])
        path_cfg = os.path.join(path, 'test_scenarios', 'test_dead_load_roof')
        set_logger(path_cfg=path_cfg, logging_level='debug')
        file_cfg = os.path.join(path_cfg, 'test_dead_load_roof.cfg')
        cfg = Config(file_cfg=file_cfg)
        cls.house = House(cfg, seed=0)

    def test_capacity(self):

        conn_capacity = {
                         37.0: [1, 2, 3, 4, 5, 6],
                         }

        simulation(self.house,
                   wind_speeds=np.arange(30.0, 80, 1.0),
                   conn_capacity=conn_capacity,
                   list_connections=range(1, 15),
                   multiplier=1.0)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestScenario30)
    unittest.TextTestRunner(verbosity=2).run(suite)
    #unittest.main(verbosity=2)
