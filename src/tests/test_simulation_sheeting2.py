#!/usr/bin/env python
from __future__ import print_function

import unittest
import os
import numpy as np
# import filecmp
import pandas as pd

# from vaws.main import simulate_wind_damage_to_houses
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

class TestHouseDamage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_reference = os.path.join(path, 'test')
        cls.path_output = os.path.join(path, 'output')

        cfg = Scenario(
            cfg_file=os.path.join(path, '../../scenarios/test_roof_sheeting2.cfg'),
            output_path=cls.path_output)

        cls.house_damage = HouseDamage(cfg, seed=1)

    def test_calculate_qz(self):

        # default case
        assert self.house_damage.cfg.regional_shielding_factor == 1
        self.assertAlmostEqual(self.house_damage.house.mzcat, 0.923, places=3)

        self.house_damage.calculate_qz_Ms(10.0)
        self.assertAlmostEqual(self.house_damage.qz, 0.05117, places=3)
        self.assertAlmostEqual(self.house_damage.Ms, 1.0, places=3)

        # sampling mzcat
        # self.house_damage.regional_shielding_factor = 0.85

        # self.house_damage.calculate_qz_Ms(10.0)
        # self.assertAlmostEqual(self.house_damage.Ms, 0.95, places=3)
        # self.assertAlmostEqual(self.house_damage.qz, 0.0683, places=3)

    def test_distribute_damage_by_column(self):

        self.house_damage.run_simulation(wind_speed=20.0)

        sum_by_col_ref = {'A': 0.0, 'B': 0.0, 'C': 0.0}
        for _zone in self.house_damage.house.zones.itervalues():
            sum_by_col_ref[_zone.name[0]] += _zone.area

        _group = self.house_damage.house.groups['sheeting']

        self.assertEqual(_group.dist_dir, 'col')

        _group.damage_grid = \
            np.array([[True, False, False, False, False, False],
                      [False, False, True, False, True, False],
                      [False, False, False, False, False, True]])

        _group.distribute_damage()

        ref_area = np.array([[0.0, 0.2025+0.405, 0.405, 0.405, 0.405, 0.2025],
                            [0.405, 0.81+0.5*0.81, 0.0, 0.81+1.0*0.81, 0.0, 0.5*0.81],
                            [0.405, 0.81, 0.81, 0.81, 1.0*0.401+0.81, 0.0]])

        for row, col in zip(range(3), range(6)):
            self.assertAlmostEqual(self.house_damage.house.zone_by_grid[
                                       (row, col)].effective_area,
                                   ref_area[row, col])

        sum_by_col = {'A': 0.0, 'B': 0.0, 'C': 0.0}
        for _zone in self.house_damage.house.zones.itervalues():
            sum_by_col[_zone.name[0]] += _zone.effective_area

        self.assertEqual(sum_by_col, sum_by_col_ref)

    def test_distribute_damage_by_row(self):

        self.house_damage.run_simulation(wind_speed=20.0)

        _group = self.house_damage.house.groups['sheeting']

        _group.dist_dir = 'row'

        ref_dic = {'A1': 0.2025, 'B1': 0.405, 'C1': 0.405,
                   'A2': 0.405, 'B2': 0.81, 'C2': 0.81,
                   'A3': 0.405, 'B3': 0.81, 'C3': 0.81,
                   'A4': 0.405, 'B4': 0.81, 'C4': 0.81,
                   'A5': 0.405, 'B5': 0.81, 'C5': 0.81,
                   'A6': 0.2025, 'B6': 0.405, 'C6': 0.405}

        for _zone in self.house_damage.house.zone_by_name.itervalues():
            _zone.area = ref_dic[_zone.name]

        _group.damage_grid = \
            np.array([[True, False, False, False, True, False],
                      [False, False, True, False, True, False],
                      [False, False, False, False, False, True]])

        sum_by_row_ref = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
        for _zone in self.house_damage.house.zones.itervalues():
            sum_by_row_ref[_zone.grid[1]] += _zone.area

        _group.distribute_damage()

        ref_area = np.array(
            [[0.0, 0.405, 0.81, 0.405, 0.0, 0.2025],
             [0.6075, 0.81, 0.0, 0.81, 0.0, 0.81],
             [0.405, 0.81, 1.215, 0.81, 2.025, 0.0]])

        for row, col in zip(range(3), range(6)):
            print('{},{}'.format(row, col))
            self.assertAlmostEqual(self.house_damage.house.zone_by_grid[
                                        (row, col)].area,
                                   ref_area[row, col])
"""


if __name__ == '__main__':
    unittest.main()
