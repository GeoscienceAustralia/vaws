#!/usr/bin/env python
from __future__ import print_function

import unittest
import os
import numpy as np
import pandas as pd

from vaws.simulation import HouseDamage
from vaws.config import Config


class TestHouseDamage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_reference = path

        cls.cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_scenario15/test_scenario15.cfg'))

        cls.house_damage = HouseDamage(cfg=cls.cfg, seed=1)
        cls.house_damage.house.replace_cost = 45092.97

    # def test_calculate_qz(self):

        # self.assertEqual(self.house_damage.house.height, 4.5)
        # self.assertEqual(self.cfg.terrain_category, '2')

        # self.house_damage.set_wind_profile()
        # self.assertEqual(self.house_damage.profile, 5)
        # self.assertAlmostEqual(self.house_damage.mzcat, 0.9425, places=4)
        #
        # # regional_shielding_factor > 0.85
        # self.house_damage.regional_shielding_factor = 1.0
        # self.house_damage.calculate_qz(10.0)
        # self.assertAlmostEqual(self.house_damage.qz, 0.05472, places=4)
        #
        # # regional_shielding_factor < 0.85
        # self.house_damage.regional_shielding_factor = 0.5
        # self.house_damage.calculate_qz(10.0)
        # self.assertAlmostEqual(self.house_damage.qz, 0.21888, places=4)

    def test_calculate_damage_ratio(self):

        ref_dat = pd.read_csv(os.path.join(self.path_reference,
                                           'repair_cost_by_conn_type_group.csv'))

        for _, item in ref_dat.iterrows():

            dic_ = {'sheeting': item['dmg_ratio_sheeting'],
                    'batten': item['dmg_ratio_batten'],
                    'rafter': item['dmg_ratio_rafter']}

            repair_dic_ = {'sheeting': item['repair_cost_sheeting'],
                           'batten': item['repair_cost_batten'],
                           'rafter': item['repair_cost_rafter']}

            # assign damage area
            for group_name, group in self.house_damage.house.groups.iteritems():
                group.damaged_area = dic_[group.name] * group.costing_area

            self.house_damage.compute_damage_index(20.0)

            # for group_name, group in self.house_damage.house.groups.iteritems():

                # try:
                #     self.assertAlmostEqual(group.repair_cost,
                #                            repair_dic_[group.name])
                # except AssertionError:
                #     print('{}:{}:{}, {}, {}, {}:{}:{}'.format(
                #         dic_['sheeting'], dic_['batten'], dic_['rafter'],
                #         group.name,
                #         group.repair_cost,
                #         repair_dic_['sheeting'], repair_dic_['batten'],
                #         repair_dic_['rafter']))

            self.assertAlmostEqual(self.house_damage.di,
                                   min(item['loss_ratio'], 1.0),
                                   places=4)

if __name__ == '__main__':
    unittest.main()
