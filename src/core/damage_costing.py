"""
    Damage Costing Module - costing profiles for a "type of damage"
        - loaded from the database
        - imported from house
        - referenced by damage module to cost damages
"""
import copy


class Costing(object):
    def __init__(self, inst):
        """

        Args:
            inst: instance of database.DamageCosting
        """
        dic_ = copy.deepcopy(inst.__dict__)

        self.id = dic_['id']
        self.name = dic_['costing_name']
        self.area = dic_['area']

        self.env_factor_type = dic_['envelope_factor_formula_type']
        self.env_repair_rate = dic_['envelope_repair_rate']
        self.int_factor_type = dic_['internal_factor_formula_type']
        self.int_repair_rate = dic_['internal_repair_rate']

        self.env_c1 = dic_['env_coeff_1']
        self.env_c2 = dic_['env_coeff_2']
        self.env_c3 = dic_['env_coeff_3']

        self.int_c1 = dic_['int_coeff_1']
        self.int_c2 = dic_['int_coeff_2']
        self.int_c3 = dic_['int_coeff_3']

        if self.env_factor_type == 1:
            self.env_repair = self.__env_func_type1
        elif self.env_factor_type == 2:
            self.env_repair = self.__env_func_type2
        else:
            raise LookupError('Invalid env_factor_type: {}'.format(
                self.env_factor_type))

        if self.int_factor_type == 1:
            self.lining_repair = self.__lining_func_type1
        elif self.int_factor_type == 2:
            self.lining_repair = self.__lining_func_type2
        else:
            raise LookupError('Invalid int_factor_type: {}'.format(
                self.int_factor_type))

    def calculate_damage(self, x):
        assert 0 <= x <= 1
        return x * (self.area * self.env_repair(x) * self.env_repair_rate +
                    self.lining_repair(x) * self.int_repair_rate)

    def __env_func_type1(self, x):
        return self.env_c1 * x ** 2 + self.env_c2 * x + self.env_c3

    def __env_func_type2(self, x):
        try:
            return self.env_c1 * x ** self.env_c2
        except ZeroDivisionError:
            return 0.0

    def __lining_func_type1(self, x):
        return self.int_c1 * x ** 2 + self.int_c2 * x + self.int_c3

    def __lining_func_type2(self, x):
        try:
            self.int_c1 * x ** self.int_c2
        except ZeroDivisionError:
            return 0.0

# unit tests
if __name__ == '__main__':
    import unittest
    import database

    class MyTestCase(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            db_file = './test_roof_sheeting2.db'
            db_costing = database.DatabaseManager(db_file).session.query(
                database.DamageCosting).all()

            cls.costing1 = Costing(db_costing[0])
            cls.costing2 = Costing(db_costing[4])

        def test_env_func_type1(self):
            assert self.costing1.env_factor_type == 1

            area = 10.125
            env_rate = 72.4
            c1 = 0.3105
            c2 = -0.894300
            c3 = 1.601500

            self.assertAlmostEqual(self.costing1.area, area)
            self.assertAlmostEqual(self.costing1.env_repair_rate, env_rate)
            self.assertAlmostEqual(self.costing1.env_c1, c1)
            self.assertAlmostEqual(self.costing1.env_c2, c2)
            self.assertAlmostEqual(self.costing1.env_c3, c3)

            self.assertAlmostEqual(self.costing1.calculate_damage(0.0),
                                   0.0)
            self.assertAlmostEqual(self.costing1.calculate_damage(0.5),
                                   451.5496, places=4)
            self.assertAlmostEqual(self.costing1.calculate_damage(1.0),
                                   746.0250, places=4)

        def test_env_func_type2(self):
            assert self.costing2.env_factor_type == 2

            area = 106.4
            env_rate = 243.72
            c1 = 1.0514
            c2 = -0.2271

            self.assertAlmostEqual(self.costing2.area, area)
            self.assertAlmostEqual(self.costing2.env_repair_rate, env_rate)
            self.assertAlmostEqual(self.costing2.env_c1, c1)
            self.assertAlmostEqual(self.costing2.env_c2, c2)

            self.assertAlmostEqual(self.costing2.calculate_damage(0.0),
                                   0.0)
            self.assertAlmostEqual(self.costing2.calculate_damage(0.5),
                                   15956.3916, places=4)
            self.assertAlmostEqual(self.costing2.calculate_damage(1.0),
                                   27264.7029, places=4)


    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)