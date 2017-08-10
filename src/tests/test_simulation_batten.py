#!/usr/bin/env python
from __future__ import print_function

import unittest
import os
import numpy as np
# import filecmp
import pandas as pd

from vaws.house_damage import HouseDamage
# import vaws.database as database
from vaws.config import Config
# from vaws import zone
# from vaws import engine


def check_file_consistency(file1, file2, **kwargs):

    try:
        identical = filecmp.cmp(file1, file2)
    except OSError:
        print('{} does not exist'.format(file2))
    else:
        if not identical:
            try:
                data1 = pd.read_csv(file1, **kwargs)
                data2 = pd.read_csv(file2, **kwargs)
            except ValueError:
                print('No columns to parse from {}'.format(file2))
            else:
                try:
                    pd.util.testing.assert_frame_equal(data1, data2)
                except AssertionError:
                    print('{} and {} are different'.format(file1, file2))


def consistency_house_damage_idx(path_reference, path_output):

    file1 = os.path.join(path_reference, 'house_dmg_idx.csv')
    file2 = os.path.join(path_output, 'results_model.h5')

    data1 = pd.read_csv(file1)

    data2 = pd.read_hdf(file2, 'di')

    print('comparing {} and {}'.format(file1, file2))
    for i in range(data2.shape[0]):

        try:
            pd.util.testing.assert_frame_equal(data1[str(i)], data2.loc[i])
        except AssertionError:
            print('different at {}'.format(i))


def consistency_house_cpi(path_reference, path_output):
    filename = 'house_cpi.csv'
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
            try:
                np.testing.assert_almost_equal(data1.values, data2.values,
                                               decimal=3)
            except AssertionError:
                print('{} and {} are different'.format(file1, file2))


def consistency_house_damage(path_reference, path_output):
    # filename = 'house_damage.csv'
    filename = 'dmg_pct_by_conn_type.csv'
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

            for col in data1.columns:

                try:
                    assert pd.util.testing.Series.equals(data1[col], data2[col])

                except AssertionError:
                    try:
                        np.testing.assert_almost_equal(data1[col].values,
                                                       data2[col].values,
                                                       decimal=3)
                    except AssertionError:
                        print('{} and {} are different in {}'.format(
                            file1, file2, col))


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
            data2 = pd.read_csv(file2, index_col=0)
            map_dic = {'median': 'Median', 'sigma': 'Beta'}
            for state, value in data2.iterrows():
                for key, mapped in map_dic.iteritems():
                    str_ = state[0].upper() + state[1:] + ' ' + mapped
                    try:
                        ref_value = data1.loc[0, str_]
                        np.testing.assert_almost_equal(ref_value,
                                                       value[key],
                                                       decimal=3)
                    except KeyError:
                        print('invalid key: {}/{}'.format(str_, key))
                    except AssertionError:
                        print('different: {}, {}, {}'.format(state,
                                                             ref_value,
                                                             value[key]))


def consistency_houses_damaged(path_reference, path_output):
    # filename = 'houses_damaged_at_v.csv'
    filename = 'dmg_freq_by_conn_type.csv'
    file1 = os.path.join(path_reference, filename)
    file2 = os.path.join(path_output, filename)

    try:
        identical = filecmp.cmp(file1, file2)
    except OSError:
        print('{} does not exist'.format(file2))
    else:
        if not identical:
            data1 = pd.read_csv(file1, skiprows=3)
            data2 = pd.read_csv(file2, skiprows=3)

            for col in data1.columns:

                try:
                    np.testing.assert_almost_equal(data1[col].values,
                                                   data2[col].values,
                                                   decimal=3)
                except AssertionError:
                    print('{} and {} are different in {}'.format(
                        file1, file2, col))


def consistency_wateringress(path_reference, path_output):
    filename = 'wateringress.csv'
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
            for key, value in data2.iteritems():
                try:
                    np.testing.assert_almost_equal(value.values,
                                                   data1[key].values,
                                                   decimal=3)
                except TypeError:
                    try:
                        assert (value == data1[key]).all()
                    except AssertionError:
                        msg = 'Not equal: {}'.format(key)
                        print(msg)

                except AssertionError:
                    msg = 'Not equal: {}'.format(key)
                    print(msg)


def consistency_wind_debris(path_reference, path_output):
    filename = 'wind_debris.csv'
    file1 = os.path.join(path_reference, filename)
    file2 = os.path.join(path_output, filename)
    check_file_consistency(file1, file2)

"""
class TestDistributeMultiSwitchesOFF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cls.path_reference = os.path.join(path, 'test/output_roof_sheeting2_OFF')
        # cls.path_reference = os.path.join(path, 'test/output_no_dist')
        cls.path_output = os.path.join(path, 'output')

        for the_file in os.listdir(cls.path_output):
            file_path = os.path.join(cls.path_output, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        cfg = Scenario(cfg_file=os.path.join(path, 'scenarios/test_roof_sheeting2.cfg'),
                       output_path=cls.path_output)

        cfg.flags['random_seed'] = True
        cfg.parallel = False

        cfg.flags['dmg_distribute_sheeting'] = False
        components_list = ['rafter', 'batten', 'wallcladding',
                           'wallcollapse', 'wallracking']

        for component in components_list:
            cfg.flags['dmg_distribute_{}'.format(component)] = False

        _ = simulate_wind_damage_to_houses(cfg)

    def test_consistency_house_damage_idx(self):

        consistency_house_damage_idx(self.path_reference, self.path_output)


class TestDistributeMultiSwitchesOn(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])

        cls.path_reference = os.path.join(path, 'test/output_roof_sheeting2_ON')
        # cls.path_reference = os.path.join(path, 'test/output_no_dist')
        cls.path_output = os.path.join(path, 'output')

        for the_file in os.listdir(cls.path_output):
            file_path = os.path.join(cls.path_output, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        cfg = Scenario(cfg_file=os.path.join(path, 'scenarios/test_roof_sheeting2.cfg'),
                       output_path=cls.path_output)

        cfg.flags['random_seed'] = True
        cfg.parallel = False

        cfg.flags['dmg_distribute_sheeting'] = True
        components_list = ['rafter', 'batten', 'wallcladding',
                           'wallcollapse', 'wallracking']

        for component in components_list:
            cfg.flags['dmg_distribute_{}'.format(component)] = False

        _ = simulate_wind_damage_to_houses(cfg)

    def test_consistency_house_damage_idx(self):

        consistency_house_damage_idx(self.path_reference, self.path_output)

    # def test_consistency_house_cpi(self):
    #     consistency_house_cpi(self.path_reference, self.path_output)
    #
    # def test_consistency_house_damage(self):
    #     consistency_house_damage(self.path_reference, self.path_output)
    #
    # def test_consistency_fragilites(self):
    #     consistency_fragilites(self.path_reference, self.path_output)
    #
    # def test_consistency_houses_damaged(self):
    #     consistency_houses_damaged(self.path_reference, self.path_output)
    #
    # def test_consistency_wateringress(self):
    #     consistency_wateringress(self.path_reference, self.path_output)
    #
    # def test_consistency_wind_debris(self):
    #     consistency_wind_debris(self.path_reference, self.path_output)
"""

class TestHouseDamage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_reference = os.path.join(path, 'test')
        cls.path_output = os.path.join(path, 'output')

        cfg = Config(
            cfg_file=os.path.join(path, '../../scenarios/test_sheeting_batten/test_sheeting_batten.cfg'))

        cls.house_damage = HouseDamage(cfg, seed=1)

    def test_distribute_damage_by_row(self):

        # each time check whether the sum of effective area is the same
        ref_area = {}
        for _zone in self.house_damage.house.zones.itervalues():
            ref_area.setdefault(_zone.name[0], []).append(_zone.area)
            ref_area.setdefault(_zone.name[1], []).append(_zone.area)

        for key, value in ref_area.iteritems():
            ref_area[key] = np.array(value).sum()

        # # iteration over wind speed list
        for id_speed, wind_speed in enumerate(self.house_damage.cfg.speeds):

            # simulate sampled house
            self.house_damage.run_simulation(wind_speed)

            # check effective area
            area = {}
            for _zone in self.house_damage.house.zones.itervalues():
                area.setdefault(_zone.name[0], []).append(_zone.area)
                area.setdefault(_zone.name[1], []).append(_zone.area)

            for key, value in area.iteritems():
                sum_area = np.array(value).sum()
                try:
                    self.assertAlmostEqual(sum_area, ref_area[key])
                except AssertionError:
                    print('{}: {} is different from {}'.format(key, sum_area,
                                                           ref_area[key]))
                    self.assertEqual(0, 1)

    def test_cal_damage_index(self):
        pass
    # check cal_damage_index esp. factors_costing

if __name__ == '__main__':
    unittest.main()
