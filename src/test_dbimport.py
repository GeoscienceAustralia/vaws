#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import unittest
import os
import filecmp
import pandas as pd

import core.database as database
import core.dbimport as dbimport


class options(object):

    def __init__(self):
        self.model_database = None
        self.data_folder = None


class TestDBimport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.path = '/'.join(__file__.split('/')[:-1])

        cls.ref_model = os.path.join(cls.path, 'test/model.db')
        cls.out_model = os.path.join(cls.path, 'core/output/model.db')

        cls.path_output = os.path.join(cls.path, 'core/output')
        cls.path_reference = os.path.join(cls.path, 'test/output')

        option = options()
        option.model_database = cls.out_model
        option.data_folder = os.path.join(cls.path, '../data')

        database.configure(option.model_database, flag_make=True)
        dbimport.import_model(option.data_folder, option.model_database)
        # database.db.close()

    @classmethod
    def tearDown(cls):
        database.db.close()

    def test_consistency_model_db(self):

        try:
            self.assertTrue(filecmp.cmp(self.ref_model,
                                        self.out_model))
        except AssertionError:
            print('{} and {} are different'.format(self.ref_model,
                                                   self.out_model))

'''
class TestDamagge(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.path = '/'.join(__file__.split('/')[:-1])
        cls.out_model = os.path.join(cls.path, './core/output/model.db')

        cls.path_output = os.path.join(cls.path, './core/output')
        cls.path_reference = os.path.join(cls.path, './test/output')

        # running with created database
        database.configure(cls.out_model)
        scenario1 = scenario.loadFromCSV(os.path.join(cls.path,
                                                      'scenarios/carl1.csv'))
        scenario1.flags['SCEN_SEED_RANDOM'] = True

        option = options()
        option.output_folder = cls.path_output

        mySim = WindDamageSimulator(option, None, None)
        mySim.set_scenario(scenario1)
        mySim.simulator_mainloop()

    def tearDown(self):
        database.db.close()

    def check_file_consistency(self, file1, file2, **kwargs):

        data1 = pd.read_csv(file1, **kwargs)
        data2 = pd.read_csv(file2, **kwargs)
        print('{}:{}'.format(file1, file2))
        # print('{}'.format(data1.head()))
        pd.util.testing.assert_frame_equal(data1, data2)

        try:
            self.assertTrue(filecmp.cmp(file1, file2))
        except AssertionError:
            print('{} and {} are different'.format(file1, file2))

    def test_consistency_house_cpi(self):
        filename = 'house_cpi.csv'
        file1 = os.path.join(self.path_reference, filename)
        file2 = os.path.join(self.path_output, filename)
        self.check_file_consistency(file1, file2)

    def test_consistency_house_damage(self):
        filename = 'house_damage.csv'
        file1 = os.path.join(self.path_reference, filename)
        file2 = os.path.join(self.path_output, filename)
        self.check_file_consistency(file1, file2)

    def test_consistency_fragilites(self):
        filename = 'fragilities.csv'
        file1 = os.path.join(self.path_reference, filename)
        file2 = os.path.join(self.path_output, filename)
        self.check_file_consistency(file1, file2)

    def test_consistency_houses_damaged(self):
        filename = 'houses_damaged_at_v.csv'
        file1 = os.path.join(self.path_reference, filename)
        file2 = os.path.join(self.path_output, filename)
        self.check_file_consistency(file1, file2, skiprows=3)

    def test_consistency_wateringress(self):
        filename = 'wateringress.csv'
        file1 = os.path.join(self.path_reference, filename)
        file2 = os.path.join(self.path_output, filename)
        self.check_file_consistency(file1, file2)

    def test_consistency_wind_debris(self):
        filename = 'wind_debris.csv'
        file1 = os.path.join(self.path_reference, filename)
        file2 = os.path.join(self.path_output, filename)
        self.check_file_consistency(file1, file2)
'''

if __name__ == '__main__':
    unittest.main()
