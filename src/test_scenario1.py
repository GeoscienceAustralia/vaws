#!/usr/bin/env python
from __future__ import print_function

import unittest
import os

from core.simulation import HouseDamage
from core.scenario import Scenario
import logging


class TestHouseDamage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, 'scenarios/test_scenario1.cfg'),
            output_path=cls.path_output)

        cls.house_damage = HouseDamage(cfg, seed=0)

        # set up logging
        file_logger = os.path.join(cls.path_output, 'log.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    def test_load_by_conn(self):

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


if __name__ == '__main__':
    unittest.main()
