# unit tests
import unittest
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from vaws.damage_costing import Costing, WaterIngressCosting, \
    compute_water_ingress_given_damage
from vaws.config import Config


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = '/'.join(__file__.split('/')[:-1])
        filename = 'damage_costing_data.csv'
        df_costing = pd.read_csv(os.path.join(path, '../../scenarios/test_roof_sheeting2/input/house/',
                                              filename))
        adic = df_costing.loc[0].to_dict()
        cls.costing1 = Costing(costing_name=adic['name'],
                               **adic)
        adic = df_costing.loc[4].to_dict()
        cls.costing2 = Costing(costing_name=adic['name'],
                               **adic)

    def test_env_func_type1(self):
        assert self.costing1.envelope_factor_formula_type == 1
        assert self.costing1.internal_factor_formula_type == 1

        area = 10.125
        env_rate = 72.4
        c1 = 0.3105
        c2 = -0.894300
        c3 = 1.601500

        self.assertAlmostEqual(self.costing1.surface_area, area)
        self.assertAlmostEqual(self.costing1.envelope_repair_rate, env_rate)
        self.assertAlmostEqual(self.costing1.envelope_coeff1, c1)
        self.assertAlmostEqual(self.costing1.envelope_coeff2, c2)
        self.assertAlmostEqual(self.costing1.envelope_coeff3, c3)

        self.assertAlmostEqual(self.costing1.compute_cost(0.0),
                               0.0)
        self.assertAlmostEqual(self.costing1.compute_cost(0.5),
                               451.5496, places=4)
        self.assertAlmostEqual(self.costing1.compute_cost(1.0),
                               746.0250, places=4)

    def test_env_func_type2(self):
        assert self.costing2.envelope_factor_formula_type == 2
        assert self.costing2.internal_factor_formula_type == 1

        area = 106.4
        env_rate = 243.72
        c1 = 1.0514
        c2 = -0.2271
        c3 = 0.0

        self.assertAlmostEqual(self.costing2.surface_area, area)
        self.assertAlmostEqual(self.costing2.envelope_repair_rate, env_rate)
        self.assertAlmostEqual(self.costing2.envelope_coeff1, c1)
        self.assertAlmostEqual(self.costing2.envelope_coeff2, c2)
        self.assertAlmostEqual(self.costing2.envelope_coeff3, c3)

        self.assertAlmostEqual(self.costing2.compute_cost(0.0),
                               0.0)
        self.assertAlmostEqual(self.costing2.compute_cost(0.5),
                               15956.3916, places=4)
        self.assertAlmostEqual(self.costing2.compute_cost(1.0),
                               27264.7029, places=4)


class WaterIngressTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = '/'.join(__file__.split('/')[:-1])
        cls.cfg = Config(
            cfg_file=os.path.join(path,
                                  '../../scenarios/test_scenario16/test_scenario16.cfg'))

    def test_compute_water_ingress_given_damage(self):

        di_array = [0, 0.15, 0.3, 0.7]
        speed_array = np.arange(20, 100, 1.0)
        a = np.zeros((len(di_array), len(speed_array)))

        for i, di in enumerate(di_array):
            for j, speed in enumerate(speed_array):
                a[i, j] = compute_water_ingress_given_damage(
                    di, speed, self.cfg.water_ingress_given_di)

        plt.figure()
        for j in range(a.shape[0]):
            plt.plot(speed_array, a[j, :],
                     label='DI={:.1f}'.format(di_array[j]))

        plt.legend()
        plt.xlabel('wind speed')
        plt.ylabel('Water ingress')
        plt.grid(1)
        plt.pause(1.0)
        plt.close()

    def test_water_costs(self):

        # thresholds = [0, 5.0, 18.0, 37.0, 67.0, 100.0]

        dic_wi = {0: 0,
                  4.3: 5.0,
                  5.6: 5.0,
                  22.12: 18.0,
                  50.0: 37.0,
                  67.1: 67.0,
                  99.0: 100.0}

        damage_name = 'Loss of roof sheeting'
        _df = self.cfg.water_ingress_costings[damage_name]

        for wi, expected in dic_wi.iteritems():
            idx = np.argsort(np.abs(_df.index - wi))[0]
            self.assertEquals(_df.iloc[idx].name, expected)

    def test_water_ingress_costings(self):

        file_water_ingress_costing = os.path.join(self.cfg.path_house_data,
                                                  'water_ingress_costing_data.csv')
        dic_ = {}
        tmp = pd.read_csv(file_water_ingress_costing)
        for key, grouped in tmp.groupby('name'):
            grouped = grouped.set_index('water_ingress')
            grouped['costing'] = grouped.apply(
                lambda row: WaterIngressCosting(costing_name=key,
                                                **row),
                axis=1)
            dic_[key] = grouped

        for _name, _df in dic_.iteritems():

            wi_array = [0, 5.0, 18.0, 37.0, 67.0, 100.0]
            di_array = np.arange(0.0, 1.1, 0.1)
            a = np.zeros((len(wi_array), len(di_array)))

            # _df = self.cfg.dic_water_ingress_costings[name]
            for i, wi in enumerate(wi_array):
                for j, di in enumerate(di_array):
                    idx = np.argsort(np.abs(_df.index - wi))[0]
                    a[i, j] = _df.iloc[idx]['costing'].compute_cost(di)

            plt.figure()
            for j in range(a.shape[0]):
                plt.plot(di_array, a[j, :],
                         label='WI={:.0f}'.format(wi_array[j]))

            plt.legend()
            plt.xlabel('Damage index')
            plt.ylabel('Cost due to water ingress')
            plt.title(_name)
            plt.grid(1)
            plt.pause(1.0)
            #plt.savefig('./wi_costing_{}'.format(_name))
            plt.close()

if __name__ == '__main__':
    unittest.main()

    # suite = unittest.TestLoader().loadTestsFromTestCase(WaterIngressTestCase)
    # unittest.TextTestRunner(verbosity=2).run(suite)
