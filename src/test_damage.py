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

class TestWindDamageSimulator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path_ = '/'.join(__file__.split('/')[:-1])
        cls.path_reference = os.path.join(path_, 'test/output')
        cls.path_output = os.path.join(path_, 'core/outputs')

        database.configure(os.path.join(path_, 'model.db'))
        scenario1 = scenario.loadFromCSV(os.path.join(path_,
                                                      'scenarios/carl1.csv'))
        scenario1.flags['SCEN_SEED_RANDOM'] = True

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
                self.assertTrue(filecmp.cmp(file1, file2))
            except AssertionError:
                print('{}:{}'.format(file1, file2))

    def test_consistency_dmg_idx(self):

        filename = 'house_dmg_idx.csv'

        file1 = os.path.join(self.path_reference, filename)
        file2 = os.path.join(self.path_output, filename)

        data1 = pd.read_csv(file1)
        data2 = pd.read_csv(file2)

        sorted_data1 = data1.sort_values(by='speed').reset_index(drop=True)
        sorted_data2 = data2.sort_values(by='speed').reset_index(drop=True)

        try:
            pd.util.testing.assert_frame_equal(sorted_data1, sorted_data2)
        except AssertionError:
            print ('{} and {} are different'.format(file1, file2))
        else:
            print ('{} and {} are identical'.format(file1, file2))

if __name__ == '__main__':
    unittest.main()
