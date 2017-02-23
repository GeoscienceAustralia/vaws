#!/usr/bin/env python

import unittest
import os
import filecmp

from vaws.database import DatabaseManager
from vaws.dbimport import import_model


class TestDBimport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.path = '/'.join(__file__.split('/')[:-1])

        cls.ref_model = os.path.join(cls.path, 'test/test_roof_sheeting2.db')
        cls.out_model = os.path.join(cls.path, './output/test.db')

        path_data_folder = os.path.join(cls.path,
                                        '../data/houses/test_roof_sheeting2')

        cls.db = DatabaseManager(cls.out_model, verbose=False)
        import_model(path_data_folder, cls.db)

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
