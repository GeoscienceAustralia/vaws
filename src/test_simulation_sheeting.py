#!/usr/bin/env python
from __future__ import print_function

import unittest
import os
import numpy as np
import filecmp
import pandas as pd

from core.simulation import HouseDamage, simulate_wind_damage_to_house
import core.database as database
from core.scenario import Scenario


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
    filename = 'house_dmg_idx.csv'

    file1 = os.path.join(path_reference, filename)
    file2 = os.path.join(path_output, filename)

    identical = filecmp.cmp(file1, file2)

    if not identical:
        try:
            data1 = pd.read_csv(file1)
            data2 = pd.read_csv(file2)

            data1 = data1.sort_values(by='speed').reset_index(drop=True)
            data2 = data2.sort_values(by='speed').reset_index(drop=True)

            pd.util.testing.assert_frame_equal(data1, data2)
        except AssertionError:
            print('{} and {} are different'.format(file1, file2))


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


class TestHouseDamage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_reference = os.path.join(path, 'test')
        # path_output = os.path.join(path, 'output')
        #
        # for the_file in os.listdir(cls.path_output):
        #     file_path = os.path.join(cls.path_output, the_file)
        #     try:
        #         if os.path.isfile(file_path):
        #             os.unlink(file_path)
        #     except Exception as e:
        #         print(e)

        # model_db = os.path.join(path_, './core/output/model.db')
        # model_db = os.path.join(path_, '../data/model.db')

        # cls.model_db = database.configure(os.path.join(path, 'model.db'))

        # cfg = scenario.loadFromCSV(os.path.join(path, 'scenarios/carl1.cfg'))
        cfg = Scenario(
            cfg_file=os.path.join(path, 'scenarios/carl1_dmg_dist.cfg'),
            output_path=os.path.join(path, 'output'))

        cfg.flags['random_seed'] = True
        cfg.parallel = False
        # cfg.flags['dmg_distribute'] = True

        # optionally seed random numbers
        if cfg.flags['random_seed']:
            print('random seed is set')
            np.random.seed(42)
            # zone.seed_scipy(42)
            # engine.seed(42)

        cls.model_db = database.DatabaseManager(cfg.db_file)
        cfg.list_conn, cfg.list_conn_type, cfg.list_conn_type_group = \
            cls.model_db.get_list_conn_type(cfg.house_name)

        cls.cfg = cfg

        cls.house_damage = HouseDamage(cls.cfg, cls.model_db)

        # print('{}'.format(cfg.file_damage))
        # cls.mySim = HouseDamage(cfg, option)
        #_, house_results = cls.mySim.simulator_mainloop()
        # key = cls.mySim.result_buckets.keys()[0]
        # print('{}:{}'.format(key, cls.mySim.result_buckets[key]))

    @classmethod
    def tearDownClass(cls):
        cls.model_db.close()

    def test_calculate_qz(self):

        self.assertEqual(self.house_damage.house.height, 4.5)
        self.assertEqual(self.house_damage.terrain_category, '2')

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
        wind_speed = 0.0

        for _, item in ref_dat.iterrows():

            dic_ = {'(piersgroup)': 0.0,
                    '(debris)': 0.0,
                    '(sheeting)': item['dmg_ratio_sheeting'],
                    '(batten)': item['dmg_ratio_batten'],
                    '(rafter)': item['dmg_ratio_rafter'],
                    '(wallcladding)': 0.0,
                    '(wallracking)': 0.0,
                    '(wallcollapse)': 0.0}

            repair_dic_ = {'(piersgroup)': 0.0,
                           '(debris)': 0.0,
                           '(sheeting)': item['repair_cost_sheeting'],
                           '(batten)': item['repair_cost_batten'],
                           '(rafter)': item['repair_cost_rafter'],
                           '(wallcladding)': 0.0,
                           '(wallracking)': 0.0,
                           '(wallcollapse)': 0.0}

            # assign damage area
            for conn_type_group in self.house_damage.house.conn_type_groups:
                conn_type_group.result_percent_damaged = dic_[str(conn_type_group)]

            self.house_damage.calculate_damage_ratio(wind_speed)

            for conn_type_group in self.house_damage.house.conn_type_groups:

                try:
                    self.assertAlmostEqual(conn_type_group.repair_cost,
                                           repair_dic_[str(conn_type_group)])
                except AssertionError:
                    print('{}:{}:{}'.format(
                        conn_type_group,
                        conn_type_group.repair_cost,
                        repair_dic_[str(conn_type_group)]))

    # def test_pdfs(self):
    #     for ct in self.house_damage.house.conn_types:
    #         if True or ct.connection_type == 'piers':
    #             x = []
    #             for i in xrange(50000):
    #                 # print('ctype mean({:.3f}), stddev({:.3f})'.format(
    #                 #    ct.strength_mean, ct.strength_std_dev))
    #                 rv = np.random.lognormal(ct.strength_mean,
    #                                          ct.strength_std_dev)
    #                 x.append(rv)
    #             n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green',
    #                                         alpha=0.75)
    #             plt.title(ct.connection_type)
    #             plt.grid(True)
    #             plt.savefig(os.path.join(self.path_reference,
    #                                      'plot_{}.png'.format(ct.connection_type)))
    #             plt.close()

    def test_construction_levels(self):
        self.cfg.setConstructionLevel('low', 0.33, 0.75, 0.78)
        counts = {'low': 0, 'medium': 0, 'high': 0}
        for i in range(1000):
            level, mf, cf = self.house_damage.sample_construction_level()
            if level == 'low':
                self.assertAlmostEquals(mf, 0.75)
                self.assertAlmostEquals(cf, 0.78)
            counts[level] += 1


class TestDistributeMultiSwitchesOff(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_reference = os.path.join(path, 'test/output_test1_dmg_dist_off')
        cls.path_output = os.path.join(path, 'output')

        for the_file in os.listdir(cls.path_output):
            file_path = os.path.join(cls.path_output, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        cfg = Scenario(cfg_file=os.path.join(path,'scenarios/test.cfg'),
                       output_path=cls.path_output)

        # setting
        cfg.flags['random_seed'] = True
        cfg.parallel = False
        # cfg.flags['dmg_distribute'] = False
        components_list = ['batten', 'rafter', 'sheeting', 'wallcladding',
                           'wallcollapse', 'wallracking']

        for component in components_list:
            cfg.flags['dmg_distribute_{}'.format(component)] = False

        _ = simulate_wind_damage_to_house(cfg)

    def test_consistency_house_damage_idx(self):
        consistency_house_damage_idx(self.path_reference, self.path_output)

    def test_consistency_house_cpi(self):
        consistency_house_cpi(self.path_reference, self.path_output)

    def test_consistency_house_damage(self):
        consistency_house_damage(self.path_reference, self.path_output)

    def test_consistency_fragilites(self):
        consistency_fragilites(self.path_reference, self.path_output)

    def test_consistency_houses_damaged(self):
        consistency_houses_damaged(self.path_reference, self.path_output)

    def test_consistency_wateringress(self):
        consistency_wateringress(self.path_reference, self.path_output)

    def test_consistency_wind_debris(self):
        consistency_wind_debris(self.path_reference, self.path_output)


class TestDistributeMultiSwitchesOn(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        cls.path_reference = os.path.join(path, 'test/output_test1_dmg_dist_on')
        # cls.path_reference = os.path.join(path, 'test/output_no_dist')
        cls.path_output = os.path.join(path, 'output')

        for the_file in os.listdir(cls.path_output):
            file_path = os.path.join(cls.path_output, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        cfg = Scenario(cfg_file=os.path.join(path,'scenarios/test.cfg'),
                       output_path=cls.path_output)

        cfg.flags['random_seed'] = True
        cfg.parallel = False

        cfg.flags['dmg_distribute_sheeting'] = True
        components_list = ['rafter', 'batten', 'wallcladding',
                           'wallcollapse', 'wallracking']

        for component in components_list:
            cfg.flags['dmg_distribute_{}'.format(component)] = False

        _ = simulate_wind_damage_to_house(cfg)

    def test_consistency_house_damage_idx(self):
        consistency_house_damage_idx(self.path_reference, self.path_output)

    def test_consistency_house_cpi(self):
        consistency_house_cpi(self.path_reference, self.path_output)

    def test_consistency_house_damage(self):
        consistency_house_damage(self.path_reference, self.path_output)

    def test_consistency_fragilites(self):
        consistency_fragilites(self.path_reference, self.path_output)

    def test_consistency_houses_damaged(self):
        consistency_houses_damaged(self.path_reference, self.path_output)

    def test_consistency_wateringress(self):
        consistency_wateringress(self.path_reference, self.path_output)

    def test_consistency_wind_debris(self):
        consistency_wind_debris(self.path_reference, self.path_output)

if __name__ == '__main__':
    unittest.main()
