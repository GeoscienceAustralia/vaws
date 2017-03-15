# unit tests
if __name__ == '__main__':
    import unittest
    import pandas as pd
    import os

    from vaws.damage_costing import Costing

    class MyTestCase(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            path = '/'.join(__file__.split('/')[:-1])
            filename = 'test_roof_sheeting2/damage_costing_data.csv'
            df_costing = pd.read_csv(os.path.join(path, '../../data/houses/',
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

            self.assertAlmostEqual(self.costing1.calculate_cost(0.0),
                                   0.0)
            self.assertAlmostEqual(self.costing1.calculate_cost(0.5),
                                   451.5496, places=4)
            self.assertAlmostEqual(self.costing1.calculate_cost(1.0),
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

            self.assertAlmostEqual(self.costing2.calculate_cost(0.0),
                                   0.0)
            self.assertAlmostEqual(self.costing2.calculate_cost(0.5),
                                   15956.3916, places=4)
            self.assertAlmostEqual(self.costing2.calculate_cost(1.0),
                                   27264.7029, places=4)

if __name__ == '__main__':
    unittest.main()

    # suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    # unittest.TextTestRunner(verbosity=2).run(suite)
