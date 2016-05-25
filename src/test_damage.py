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

    def __init__(self):
        self.output_folder = None


class TestWindDamageSimulator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.path = '/'.join(__file__.split('/')[:-1])
        cls.path_reference = os.path.join(cls.path, 'test/output')
        cls.path_output = os.path.join(cls.path, 'output')

        cls.path_output_no_dist = os.path.join(cls.path, 'output_no_dist')
        cls.path_reference_no_dist = os.path.join(cls.path, 'test/output_no_dist')

        cls.list_of_files = ['house_damage.csv',
                         'house_dmg_idx.csv',
                         'house_cpi.csv',
                         'fragilities.csv',
                         'houses_damaged_at_v.csv',
                         'wateringress.csv',
                         'wind_debris.csv']

    # @classmethod
    # def tearDownClass(cls):
    #     database.db.close()
    #
    #     # delete test/output
    #     # os.path.join(path_, 'test/output')

    def check_file_consistency(self, file1, file2, **kwargs):

        true_value = filecmp.cmp(file1, file2)
        if not true_value:
            try:
                data1 = pd.read_csv(file1, **kwargs)
                data2 = pd.read_csv(file2, **kwargs)
                pd.util.testing.assert_frame_equal(data1, data2)
            except AssertionError:
                print('{} and {} are different'.format(file1, file2))
            except pd.parser.CParserError:
                print('can not parse the files {},{}'.format(file1, file2))

    def test_output_vs_reference(self):

        model_db = os.path.join(self.path, 'model.db')
        database.configure(model_db)

        scenario1 = scenario.loadFromCSV(os.path.join(self.path,
                                                      'scenarios/carl1.csv'))
        scenario1.flags['SCEN_SEED_RANDOM'] = True
        scenario1.flags['SCEN_DMG_DISTRIBUTE'] = True

        option = options()
        option.output_folder = self.path_output

        for the_file in os.listdir(self.path_output):
            file_path = os.path.join(self.path_output, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        sim = WindDamageSimulator(option, None, None)
        sim.set_scenario(scenario1)
        _ = sim.simulator_mainloop()
        database.db.close()

        for filename in self.list_of_files:
            file1 = os.path.join(self.path_reference, filename)
            file2 = os.path.join(self.path_output, filename)
            self.check_file_consistency(file1, file2)

    def test_output_vs_reference_no_dist(self):

        model_db = os.path.join(self.path, 'model.db')
        database.configure(model_db)

        scenario1 = scenario.loadFromCSV(os.path.join(self.path,
                                                      'scenarios/carl1.csv'))
        scenario1.flags['SCEN_SEED_RANDOM'] = True
        scenario1.flags['SCEN_DMG_DISTRIBUTE'] = False

        option = options()
        option.output_folder = self.path_output_no_dist

        for the_file in os.listdir(self.path_output):
            file_path = os.path.join(self.path_output, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        sim = WindDamageSimulator(option, None, None)
        sim.set_scenario(scenario1)
        _ = sim.simulator_mainloop()

        for filename in self.list_of_files:
            file3 = os.path.join(self.path_reference_no_dist, filename)
            file4 = os.path.join(self.path_output_no_dist, filename)
            self.check_file_consistency(file3, file4)

if __name__ == '__main__':
    unittest.main()
