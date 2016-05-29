#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import unittest
import os
import filecmp
import pandas as pd
import numpy as np

from core.damage import WindDamageSimulator
import core.database as database
import core.scenario as scenario
from test_simulation import check_file_consistency, consistency_wind_debris, \
    consistency_houses_damaged, consistency_house_damage_idx, \
    consistency_house_cpi, consistency_wateringress, consistency_house_damage


def consistency_fragilites(path_reference, path_output):
    filename = 'fragilities.csv'
    file1 = os.path.join(path_reference, filename)
    file2 = os.path.join(path_output, filename)
    try:
        identical = filecmp.cmp(file1, file2)
    except OSError:
        print('{} does not exist'.format(file2))
    else:
        if not identical:
            data1 = pd.read_csv(file1)
            data2 = pd.read_csv(file2)

            for ((key0, value0), (key1, value1)) in zip(data1.iteritems(),
                                                        data2.iteritems()):

                try:
                    np.testing.assert_almost_equal(value0.values,
                                                   value1.values,
                                                   decimal=3)

                except AssertionError:
                    print('different: {}, {}, {}'.format(key0,
                                                         value0,
                                                         value1))


class options(object):

    def __init__(self):
        self.output_folder = None


class TestWindDamageSimulator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_reference = os.path.join(path, 'test/output')
        cls.path_output = os.path.join(path, 'output')

        for the_file in os.listdir(cls.path_output):
            file_path = os.path.join(cls.path_output, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        # model_db = os.path.join(path_, './core/output/model.db')
        # model_db = os.path.join(path_, '../data/model.db')
        model_db = os.path.join(path, 'model.db')
        cls.db = database.configure(model_db)

        scenario1 = scenario.loadFromCSV(os.path.join(path,
                                                      'scenarios/carl1.cfg'))
        scenario1.flags['random_seed'] = True
        scenario1.flags['dmg_distribute'] = True

        option = options()
        option.output_folder = cls.path_output

        cls.mySim = WindDamageSimulator(scenario1, option, cls.db, None, None)
        #cls.mySim.set_scenario(scenario1)
        cls.mySim.simulator_mainloop()

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

        # delete test/output
        # os.path.join(path_, 'test/output')

    def test_random_seed(self):
        self.assertEqual(self.mySim.cfg.flags['random_seed'], True)

    def test_consistency_house_cpi(self):
        consistency_house_cpi(self.path_reference, self.path_output)

    def test_consistency_house_damage(self):
        consistency_house_damage(self.path_reference, self.path_output)

    def test_consistency_house_damage_idx(self):
        consistency_house_damage_idx(self.path_reference, self.path_output)

    def test_consistency_fragilites(self):
        consistency_fragilites(self.path_reference, self.path_output)

    def test_consistency_houses_damaged(self):
        consistency_houses_damaged(self.path_output, self.path_output)

    def test_consistency_wateringress(self):
        consistency_wateringress(self.path_reference, self.path_output)

    def test_consistency_wind_debris(self):
        consistency_wind_debris(self.path_reference, self.path_output)


class TestWindDamageSimulator_No_Distribute(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_reference = os.path.join(path, 'test/output_no_dist')
        cls.path_output = os.path.join(path, 'output_no_dist')

        for the_file in os.listdir(cls.path_output):
            file_path = os.path.join(cls.path_output, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        # model_db = os.path.join(path_, './core/output/model.db')
        # model_db = os.path.join(path_, '../data/model.db')
        model_db = os.path.join(path, 'model.db')
        cls.db = database.configure(model_db)

        scenario1 = scenario.loadFromCSV(os.path.join(path,
                                                      'scenarios/carl1.cfg'))
        scenario1.flags['random_seed'] = True
        scenario1.flags['dmg_distribute'] = False

        option = options()
        option.output_folder = cls.path_output

        cls.mySim = WindDamageSimulator(scenario1, option, cls.db, None, None)
        #cls.mySim.set_scenario(scenario1)
        cls.mySim.simulator_mainloop()

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

        # delete test/output
        # os.path.join(path_, 'test/output')

    def test_random_seed(self):
        self.assertEqual(self.mySim.cfg.flags['random_seed'], True)

    def test_dmg_distribute(self):
        self.assertEqual(self.mySim.cfg.flags['dmg_distribute'], False)

    def test_consistency_house_cpi(self):
        consistency_house_cpi(self.path_reference, self.path_output)

    def test_consistency_house_damage(self):
        consistency_house_damage(self.path_reference, self.path_output)

    def test_consistency_house_damage_idx(self):
        consistency_house_damage_idx(self.path_reference, self.path_output)

    def test_consistency_fragilites(self):
        consistency_fragilites(self.path_reference, self.path_output)

    def test_consistency_houses_damaged(self):
        consistency_houses_damaged(self.path_output, self.path_output)

    def test_consistency_wateringress(self):
        consistency_wateringress(self.path_reference, self.path_output)

    def test_consistency_wind_debris(self):
        consistency_wind_debris(self.path_reference, self.path_output)

if __name__ == '__main__':
    unittest.main()
