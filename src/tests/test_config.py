import unittest
from numpy.testing import assert_array_equal
from numpy import linspace
import pandas as pd
from pandas.util.testing import assert_frame_equal
import os
import StringIO
import tempfile

from vaws.config import Config
from vaws.stats import compute_logarithmic_mean_stddev


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        path = '/'.join(__file__.split('/')[:-1])
        scenario_filename1 = os.path.abspath(
            os.path.join(path, '../../scenarios/default/default.cfg'))
        cls.cfg = Config(cfg_file=scenario_filename1)
        cls.path_cfg = os.path.dirname(os.path.realpath(scenario_filename1))

    def test_debris(self):
        self.assertEquals(self.cfg.flags['debris'], True)

    def test_set_wind_dir_index(self):
        self.cfg.set_wind_dir_index('Random')
        self.assertEqual(self.cfg.wind_dir_index, 8)

        self.cfg.set_wind_dir_index('SW')
        self.assertEqual(self.cfg.wind_dir_index, 1)

        self.assertRaises(ValueError, self.cfg.set_wind_dir_index('dummy'))
        self.assertEqual(self.cfg.wind_dir_index, 8)

    def test_water_ingress(self):
        self.assertTrue(self.cfg.flags['water_ingress'])
        self.cfg.flags['water_ingress'] = False
        self.assertFalse(self.cfg.flags['water_ingress'])

    def test_path(self):
        self.assertEquals(self.cfg.path_cfg, self.path_cfg)
        self.assertEquals(self.cfg.path_output,
                          os.path.join(self.path_cfg, 'output'))
        self.assertEquals(self.cfg.path_house_data,
                          os.path.join(self.path_cfg, 'input', 'house'))
        self.assertEquals(self.cfg.path_wind_profiles,
                          os.path.join(self.path_cfg, 'input', 'gust_envelope_profiles'))
        self.assertEquals(self.cfg.path_debris,
                          os.path.join(self.path_cfg, 'input', 'debris'))

    def test_read_main(self):

        assert_array_equal(self.cfg.speeds,
                           linspace(self.cfg.wind_speed_min, self.cfg.wind_speed_max,
                           self.cfg.wind_speed_steps))
        self.assertEquals(self.cfg.wind_dir_index, 0)
        self.assertEquals(self.cfg.terrain_category, '2')

    def test_set_wind_profile(self):
        self.cfg.set_wind_profile('non_cyclonic')
        self.assertEquals(self.cfg.terrain_category, 'non_cyclonic')

        self.cfg.set_wind_profile('2.5')
        self.assertEquals(self.cfg.terrain_category, '2.5')

        self.assertRaises(AssertionError, self.cfg.set_wind_profile('dummy'))

    def test_read_fragility_thresholds(self):
        ref = pd.DataFrame([[0.02, 'b'],
                            [0.10, 'g'],
                            [0.35, 'y'],
                            [0.90, 'r']],
                           columns=['threshold', 'color'], index=['slight', 'medium', 'severe', 'complete'])
        assert_frame_equal(self.cfg.fragility_thresholds, ref)

    def test_set_region_name(self):
        self.cfg.set_region_name('Capital_city')
        self.assertEquals(self.cfg.region_name, 'Capital_city')

        self.cfg.set_region_name('Tropical_town')
        self.assertEquals(self.cfg.region_name, 'Tropical_town')

        self.assertRaises(AssertionError, self.cfg.set_region_name('dummy'))
        self.assertEquals(self.cfg.region_name, 'Capital_city')

    def test_return_norm_cdf(self):
        row = {'upper': 75.0, 'lower': 50.0}
        # mean = 62.5, std = 4.166
        a = self.cfg.return_norm_cdf(row)
        self.assertAlmostEqual(a(55.0), 0.03593, places=4)

    def test_get_diff_tuples(self):
        row = {'key1': {0: 3, 1: 4}, 'key2': {0: 2, 1: 5}}
        a = self.cfg.get_diff_tuples(row, 'key1', 'key2')
        self.assertEquals(a, (1, -1))

    def test_read_damage_costing_data(self):
        data = StringIO.StringIO("""
group_name,dist_order,dist_dir,damage_scenario
sheeting,1,col,Loss of roof sheeting
batten,2,row,Loss of roof sheeting & purlins
rafter,3,col,Loss of roof structure
        """)
        df_groups = pd.read_csv(data, index_col='group_name')

        file_damage_costing = StringIO.StringIO("""
name,surface_area,envelope_repair_rate,envelope_factor_formula_type,envelope_coeff1,envelope_coeff2,envelope_coeff3,internal_repair_rate,internal_factor_formula_type,internal_coeff1,internal_coeff2,internal_coeff3,water_ingress_order
Loss of roof sheeting,116,72.4,1,0.3105,-0.8943,1.6015,0,1,0,0,0,6
Loss of roof sheeting & purlins,116,184.23,1,0.3105,-0.8943,1.6015,0,1,0,0,0,7
Loss of roof structure,116,317,1,0.3105,-0.8943,1.6015,8320.97,1,-0.4902,1.4896,0.0036,3
Wall debris damage,106.4,375.37,1,0.8862,-1.6957,1.8535,0,1,0,0,0,4
        """)
        dic_costing, dic_costing_to_group, damage_order_by_water_ingress = \
            self.cfg.read_damage_costing_data(file_damage_costing, df_groups)

        for key, value in zip(['Loss of roof sheeting',
                               'Loss of roof sheeting & purlins',
                              'Loss of roof structure'],
                              [['sheeting'], ['batten'], ['rafter']]):
            self.assertEqual(dic_costing_to_group[key], value)

        self.assertEqual(damage_order_by_water_ingress,
                         ['Loss of roof structure',
                          'Loss of roof sheeting',
                          'Loss of roof sheeting & purlins'])

    def test_read_water_ingress_costing_data(self):
        pass

    def test_read_front_facing_walls(self):

        _file = tempfile.NamedTemporaryFile(mode='w+t')
        try:
            _file.writelines(['wind_dir,wall_name\n',
                              'S,1\n',
                              'SW,1,3\n',
                              'W,3\n',
                              'NW,3,5\n',
                              'N,5\n',
                              'NE,5,7\n',
                              'E,7\n',
                              'SE,1,7\n'])
            _file.seek(0)

            ref = {'S': [1], 'SW': [1, 3], 'W': [3], 'NW': [3, 5], 'N': [5],
                   'NE': [5, 7], 'E': [7], 'SE': [1, 7]}
            _dic = self.cfg.read_front_facing_walls(_file.name)
            self.assertDictEqual(_dic, ref)

        finally:
            _file.close()

    def test_read_damage_factorings(self):

        _file = tempfile.NamedTemporaryFile(mode='w+t')
        try:

            _file.writelines(['ParentGroup,FactorByGroup\n',
                              'batten, rafter\n',
                              'sheeting,rafter\n',
                              'sheeting,batten\n'])
            _file.seek(0)

            ref = {'batten': ['rafter'], 'sheeting': ['rafter', 'batten']}
            _dic = self.cfg.read_damage_factorings(_file.name)
            self.assertDictEqual(_dic, ref)

        finally:
            _file.close()

    def test_read_influences(self):

        _file = tempfile.NamedTemporaryFile(mode='w+t')
        try:

            _file.writelines(['conn_name, Zone, Coefficent\n',
                              '1, A1, 1\n',
                              '3, B1, 1\n',
                              '113, A13, 1, N15, 1\n',
                              '133, E13, 1, E14, 0,\n',
                              '136, F14, 1, F13, 0,\n',])

            _file.seek(0)

            ref = {1: {'A1': 1.0},
                   3: {'B1': 1.0},
                   113: {'A13': 1.0, 'N15': 1.0},
                   133: {'E13': 1.0, 'E14': 0.0},
                   136: {'F14': 1.0, 'F13': 0.0},
                   }

            _dic = self.cfg.read_influences(_file.name)
            self.assertDictEqual(_dic, ref)

        finally:
            _file.close()

    def test_read_influence_patches(self):

        _file = tempfile.NamedTemporaryFile(mode='w+t')
        try:

            _file.writelines(['Damaged connection,Connection,Zone,Inf factor,,\n',
                              '113,113, A13,0,N15,0\n',
                              '113,114,N15,1,,\n',
                              '113,121,B13,1,,\n',
                              '120,120,A14,0,G15,0\n',
                              '120,119,G15,1,,\n',
                              '120,124,A14,1,,\n'])

            _file.seek(0)

            ref = {113: {113: {'A13': 0.0, 'N15': 0.0},
                         114: {'N15': 1.0},
                         121: {'B13': 1.0}},
                   120: {120: {'A14': 0.0, 'G15': 0.0},
                         119: {'G15': 1.0},
                         124: {'A14': 1.0}},
                   }

            _dic = self.cfg.read_influence_patches(_file.name)
            self.assertDictEqual(_dic, ref)

        finally:
            _file.close()

    def test_read_water_ingress(self):

        pass

    def test_set_debris_types(self):

        _dic = self.cfg.dic_debris_regions[self.cfg.region_name]

        for key, value in self.cfg.debris_types.iteritems():

            self.assertEqual(value['cdav'], self.cfg.dic_debris_types[key]['cdav'])
            self.assertEqual(value['ratio'], _dic['{}_ratio'.format(key)])

            for item in ['frontalarea', 'mass']:
                _mean = _dic['{}_{}_mean'.format(key, item)]
                _sd = _dic['{}_{}_stddev'.format(key, item)]

                _mu, _std = compute_logarithmic_mean_stddev(_mean, _sd)

                self.assertAlmostEqual(_mu, value['{}_mu'.format(item)])
                self.assertAlmostEqual(_std, value['{}_std'.format(item)])

    def test_get_get_construction_level(self):

        ref = {'low': (0.33, 0.9, 0.58),
               'medium': (0.34, 1.0, 0.58),
               'high': (0.33, 1.1, 0.58)}
        for key, value in ref.iteritems():
            self.assertEqual(value, self.cfg.get_construction_level(key))

    def test_set_construction_level(self):

        self.cfg.set_construction_level(name='dummy', prob=0.5, mf=0.5, cf=0.5)
        self.assertDictEqual(self.cfg.construction_levels['dummy'],
                             {'probability': 0.5,
                              'mean_factor': 0.5,
                              'cov_factor': 0.5})

    def test_save_config(self):
        self.cfg.cfg_file += '.copy'
        self.cfg.save_config()

if __name__ == '__main__':
    unittest.main()

# suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
# unittest.TextTestRunner(verbosity=2).run(suite)