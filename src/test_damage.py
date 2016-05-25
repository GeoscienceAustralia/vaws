#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import unittest
import os
import filecmp
import pandas as pd
from core.damage import WindDamageSimulator
import core.database as database
import core.scenario as scenario

class options(object):

    def __init__(self, output_folder):
        self.output_folder = output_folder


class TestWindDamageSimulator_No_Distribute(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path_ = '/'.join(__file__.split('/')[:-1])
        cls.path_reference = os.path.join(path_, 'test/output_no_dist')
        cls.path_output = os.path.join(path_, 'output_no_dist')

        for the_file in os.listdir(cls.path_output):
            file_path = os.path.join(cls.path_output, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        database.configure(os.path.join(path_, 'model.db'))
        scenario1 = scenario.loadFromCSV(os.path.join(path_,
                                                      'scenarios/carl1.csv'))
        scenario1.flags['SCEN_SEED_RANDOM'] = True
        scenario1.flags['SCEN_DMG_DISTRIBUTE'] = False

        options_ = options(cls.path_output)
        cls.mySim = WindDamageSimulator(options_, None, None)
        cls.mySim.set_scenario(scenario1)
        cls.mySim.simulator_mainloop()

    @classmethod
    def tearDownClass(cls):
        database.db.close()

    def test_run(self):

        list_of_files = ['house_cpi.csv',
                         'house_damage.csv',
                         'houses_damaged_at_v.csv',
                         'wateringress.csv',
                         'wind_debris.csv',
                         'fragilities.csv']

        for item in list_of_files:

            file1 = os.path.join(self.path_reference, item)
            file2 = os.path.join(self.path_output, item)

            try:
                identical = filecmp.cmp(file1, file2)
            except IOError:
                print('{} and/or {} not exist'.format(file1, file2))
            else:
                if not identical:
                    print('{} and {} are different'.format(file1, file2))

    def test_consistency_dmg_idx(self):

        filename = 'house_dmg_idx.csv'

        file1 = os.path.join(self.path_reference, filename)
        file2 = os.path.join(self.path_output, filename)

        identical = filecmp.cmp(file1, file2)

        if not identical:
            try:
                data1 = pd.read_csv(file1)
                data2 = pd.read_csv(file2)

                data1 = data1.sort_values(by='speed').reset_index(drop=True)
                data2 = data2.sort_values(by='speed').reset_index(drop=True)

                pd.util.testing.assert_frame_equal(data1, data2)

            except AssertionError:
                print ('{} and {} are different'.format(file1, file2))


class TestWindDamageSimulator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path_ = '/'.join(__file__.split('/')[:-1])
        cls.path_reference = os.path.join(path_, 'test/output')
        cls.path_output = os.path.join(path_, 'output')

        for the_file in os.listdir(cls.path_output):
            file_path = os.path.join(cls.path_output, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        database.configure(os.path.join(path_, 'model.db'))
        scenario1 = scenario.loadFromCSV(os.path.join(path_,
                                                      'scenarios/carl1.csv'))
        scenario1.flags['SCEN_SEED_RANDOM'] = True
        scenario1.flags['SCEN_DMG_DISTRIBUTE'] = True

        options_ = options(cls.path_output)
        cls.mySim = WindDamageSimulator(options_, None, None)
        cls.mySim.set_scenario(scenario1)
        cls.mySim.simulator_mainloop()

    @classmethod
    def tearDownClass(cls):
        database.db.close()

    def test_run(self):

        list_of_files = ['house_cpi.csv',
                         'house_damage.csv',
                         'houses_damaged_at_v.csv',
                         'wateringress.csv',
                         'wind_debris.csv',
                         'fragilities.csv']

        for item in list_of_files:

            file1 = os.path.join(self.path_reference, item)
            file2 = os.path.join(self.path_output, item)

            try:
                identical = filecmp.cmp(file1, file2)
            except IOError:
                print('{} and/or {} not exist'.format(file1, file2))
            else:
                if not identical:
                    print('{} and {} are different'.format(file1, file2))

    def test_consistency_dmg_idx(self):

        filename = 'house_dmg_idx.csv'

        file1 = os.path.join(self.path_reference, filename)
        file2 = os.path.join(self.path_output, filename)

        identical = filecmp.cmp(file1, file2)

        if not identical:
            try:
                data1 = pd.read_csv(file1)
                data2 = pd.read_csv(file2)

                data1 = data1.sort_values(by='speed').reset_index(drop=True)
                data2 = data2.sort_values(by='speed').reset_index(drop=True)

                pd.util.testing.assert_frame_equal(data1, data2)

            except AssertionError:
                print ('{} and {} are different'.format(file1, file2))

if __name__ == '__main__':
    unittest.main()
