#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import unittest
import os
import filecmp
import pandas as pd

from core.database import DatabaseManager
from core.dbimport import import_model


# class options(object):
#
#     def __init__(self):
#         self.model_database = None
#         self.data_folder = None


class TestDBimport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.path = '/'.join(__file__.split('/')[:-1])

        cls.ref_model = os.path.join(cls.path, 'test/test_roof_sheeting2.db')
        cls.out_model = os.path.join(cls.path, './output/test.db')

        # cls.path_output = os.path.join(cls.path, 'core/output')
        # cls.path_reference = os.path.join(cls.path, 'test/output')

        # option = options()
        path_data_folder = os.path.join(cls.path,
                                        '../data/houses/test_roof_sheeting2')

        cls.db = DatabaseManager(cls.out_model, verbose=False)
        import_model(path_data_folder, cls.db)
        # database.db.close()

    @classmethod
    def tearDown(cls):
        cls.db.close()

    def test_consistency_model_db(self):

        try:
            self.assertTrue(filecmp.cmp(self.ref_model,
                                        self.out_model))
        except AssertionError:
            print('{} and {} are different'.format(self.ref_model,
                                                   self.out_model))

if __name__ == '__main__':
    unittest.main()
