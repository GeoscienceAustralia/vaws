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
        option.data_folder = os.path.join(cls.path, '../data')

        cls.db = database.DatabaseManager(cls.out_model, verbose=False)
        dbimport.import_model(option.data_folder, cls.db)
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
