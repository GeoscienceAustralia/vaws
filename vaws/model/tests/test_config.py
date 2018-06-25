import unittest
from numpy.testing import assert_array_equal
from numpy import linspace
import pandas as pd
from pandas.util.testing import assert_frame_equal
import os
import logging
import StringIO
import tempfile

from vaws.model.config import Config
from vaws.model.stats import compute_logarithmic_mean_stddev


class TestConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        scenario_filename1 = os.path.abspath(os.path.join(
            path, 'test_scenarios', 'test_house', 'test_house.cfg'))
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        cls.cfg = Config(file_cfg=scenario_filename1, logger=logger)
        cls.path_cfg = os.path.dirname(os.path.realpath(scenario_filename1))

    def test_debris(self):
        self.assertEquals(self.cfg.flags['debris'], True)

    def test_set_wind_dir_index(self):
        self.cfg.wind_direction = 'Random'
        self.cfg.set_wind_dir_index()
        self.assertEqual(self.cfg.wind_dir_index, 8)

        self.cfg.wind_direction = 'SW'
        self.cfg.set_wind_dir_index()
        self.assertEqual(self.cfg.wind_dir_index, 1)

        self.cfg.wind_direction = 'dummy'
        self.assertRaises(ValueError, self.cfg.set_wind_dir_index())
        self.assertEqual(self.cfg.wind_dir_index, 8)

    def test_water_ingress(self):
        self.assertFalse(self.cfg.flags['water_ingress'])
        self.cfg.flags['water_ingress'] = True
        self.assertTrue(self.cfg.flags['water_ingress'])

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

        self.assertEqual(self.cfg.model_name, 'Group 4 House')

        self.assertEqual(self.cfg.no_models, 10)

        _speeds = linspace(self.cfg.wind_speed_min,
                           self.cfg.wind_speed_max,
                           self.cfg.wind_speed_steps)
        assert_array_equal(self.cfg.wind_speeds, _speeds)

        self.assertEquals(self.cfg.wind_dir_index, 0)

        self.assertEquals(self.cfg.regional_shielding_factor, 1.0)

        self.assertEquals(self.cfg.file_wind_profiles,
                          'cyclonic_terrain_cat2.csv')

    def test_set_wind_profile(self):
        self.cfg.file_wind_profiles = 'non_cyclonic.csv'
        self.cfg.set_wind_profiles()

        self.cfg.file_wind_profiles = 'dummy'
        self.assertRaises(IOError,
                          self.cfg.set_wind_profiles())

    def test_read_fragility_thresholds(self):
        ref = pd.DataFrame([[0.02, 'b'],
                            [0.10, 'g'],
                            [0.35, 'y'],
                            [0.90, 'r']],
                           columns=['threshold', 'color'],
                           index=['slight', 'medium', 'severe', 'complete'])
        assert_frame_equal(self.cfg.fragility, ref)

    def test_set_region_name(self):
        self.cfg.set_region_name('Capital_city')
        self.assertEquals(self.cfg.region_name, 'Capital_city')

        self.cfg.set_region_name('Tropical_town')
        self.assertEquals(self.cfg.region_name, 'Tropical_town')

        self.assertRaises(IOError,
                          self.cfg.set_region_name('dummy'))

    def test_return_norm_cdf(self):
        row = {'speed_at_full_wi': 75.0, 'speed_at_zero_wi': 50.0}
        # mean = 62.5, std = 4.166
        a = self.cfg.return_norm_cdf(row)
        self.assertAlmostEqual(a(55.0), 0.03593, places=4)
        self.assertAlmostEqual(a(62.5), 0.5, places=4)

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
        # df_groups = pd.read_csv(data, index_col='group_name')

        file_damage_costing = StringIO.StringIO("""
name,surface_area,envelope_repair_rate,envelope_factor_formula_type,envelope_coeff1,envelope_coeff2,envelope_coeff3,internal_repair_rate,internal_factor_formula_type,internal_coeff1,internal_coeff2,internal_coeff3,water_ingress_order
Loss of roof sheeting,116,72.4,1,0.3105,-0.8943,1.6015,0,1,0,0,0,6
Loss of roof sheeting & purlins,116,184.23,1,0.3105,-0.8943,1.6015,0,1,0,0,0,7
Loss of roof structure,116,317,1,0.3105,-0.8943,1.6015,8320.97,1,-0.4902,1.4896,0.0036,3
Wall debris damage,106.4,375.37,1,0.8862,-1.6957,1.8535,0,1,0,0,0,4
        """)
        dic_costing, damage_order_by_water_ingress = \
            self.cfg.read_damage_costing_data(file_damage_costing)

        # TODO
        # for key, value in zip(['Loss of roof sheeting',
        #                        'Loss of roof sheeting & purlins',
        #                       'Loss of roof structure'],
        #                       [['sheeting'], ['batten'], ['rafter']]):
        #     self.assertEqual(dic_costing_to_group[key], value)

        self.assertEqual(damage_order_by_water_ingress,
                         ['Loss of roof structure',
                          'Wall debris damage',
                          'Loss of roof sheeting',
                          'Loss of roof sheeting & purlins'])

    def test_read_water_ingress_costing_data(self):
        file_water_ingress_costing_data = StringIO.StringIO("""
name,water_ingress,base_cost,formula_type,coeff1,coeff2,coeff3
Loss of roof sheeting,0,0,1,0,0,1
Loss of roof sheeting,5,2989.97,1,0,0,1
Loss of roof sheeting,18,10763.89,1,0,0,1
Loss of roof sheeting,37,22125.78,1,0,0,1
Loss of roof sheeting,67,40065.59,1,0,0,1
Loss of roof sheeting,100,59799.39,1,0,0,1
Loss of roof sheeting & purlins,0,0,1,0,0,1
Loss of roof sheeting & purlins,5,2989.97,1,0,0,1
Loss of roof sheeting & purlins,18,10763.89,1,0,0,1
Loss of roof sheeting & purlins,37,22125.78,1,0,0,1
Loss of roof sheeting & purlins,67,40065.59,1,0,0,1
Loss of roof sheeting & purlins,100,59799.39,1,0,0,1
Loss of roof structure,0,0,1,0,0,1
Loss of roof structure,5,2335.65,2,0.9894,-0.0177,0
Loss of roof structure,18,8408.35,2,0.9832,-0.0561,0
Loss of roof structure,37,17867.86,2,0.9817,-0.0632,0
Loss of roof structure,67,33140.18,1,0.2974,-0.5077,1.212
Loss of roof structure,100,49939.74,1,0.0968,-0.2941,1.1967
WI only,0,0,1,0,0,1
WI only,5,2989.97,1,0,0,1
WI only,18,10763.89,1,0,0,1
WI only,37,22125.78,1,0,0,1
WI only,67,40065.59,1,0,0,1
WI only,100,59799.39,1,0,0,1
        """)

        file_conn_groups = StringIO.StringIO("""
group_name,dist_order,dist_dir,damage_scenario,trigger_collapse_at,set_zone_to_zero,water_ingress_order
sheeting,1,col,Loss of roof sheeting,0,1,6
batten,2,row,Loss of roof sheeting & purlins,0,1,7
rafter,3,patch,Loss of roof structure,0,0,3        
        """)
        groups = pd.read_csv(file_conn_groups, index_col=0)

        _dic = self.cfg.read_water_ingress_costing_data(
            file_water_ingress_costing_data)

        self.assertEqual(sorted(_dic.keys()),
                         ['Loss of roof sheeting',
                          'Loss of roof sheeting & purlins',
                          'Loss of roof structure',
                          'WI only'])

        #self.assertAlmostEqual()

    def test_read_front_facing_walls(self):

        _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
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
            os.unlink(_file.name)

    def test_read_damage_factorings(self):

        _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
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
            os.unlink(_file.name)

    def test_read_influences(self):

        _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
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
            os.unlink(_file.name)

    def test_read_influence_patches(self):

        _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
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
            os.unlink(_file.name)

    def test_read_water_ingress(self):
        thresholds = [0.1, 0.2, 0.5]
        index = [0.1, 0.2, 0.5, 1.1]
        speed_zero = [50.0, 35.0, 0.0, -20.0]
        speed_full = [75.0, 55.0, 40.0, 20.0]

        assert_array_equal(self.cfg.water_ingress_i_thresholds, thresholds)
        assert_array_equal(self.cfg.water_ingress.index, index)
        assert_array_equal(self.cfg.water_ingress['speed_at_zero_wi'].values,
                           speed_zero)
        assert_array_equal(self.cfg.water_ingress['speed_at_full_wi'].values,
                           speed_full)

        for ind, low, up in zip(index, speed_zero, speed_full):
            _mean = 0.5 * (low + up)
            est = self.cfg.water_ingress.loc[ind, 'wi'](_mean)
            self.assertAlmostEqual(est, 0.5)

    def test_set_debris_types(self):

        _dic = self.cfg.debris_regions[self.cfg.region_name]

        for key, value in self.cfg.debris_types.items():

            self.assertEqual(value['cdav'], self.cfg.debris_types[key]['cdav'])
            self.assertEqual(value['ratio'], _dic['{}_ratio'.format(key)])

            for item in ['frontal_area', 'mass']:
                _mean = _dic['{}_{}_mean'.format(key, item)]
                _sd = _dic['{}_{}_stddev'.format(key, item)]

                _mu, _std = compute_logarithmic_mean_stddev(_mean, _sd)

                self.assertAlmostEqual(_mu, value['{}_mu'.format(item)])
                self.assertAlmostEqual(_std, value['{}_std'.format(item)])

    def test_get_construction_level(self):

        ref = {'low': (0.9, 0.58),
               'medium': (1.0, 0.58),
               'high': (1.1, 0.58)}
        keys = ['mean_factor', 'cv_factor']
        for key, values in ref.items():
            for sub_key, value in zip(keys, values):
                self.assertEqual(value,
                                 self.cfg.construction_levels[key][sub_key])

    def test_set_construction_levels(self):

        self.cfg.construction_levels_levels = ['dummy']
        self.cfg.construction_levels_probs = [1.0]
        self.cfg.construction_levels_mean_factors = [0.5]
        self.cfg.construction_levels_cv_factors = [0.5]

        self.cfg.set_construction_levels()

        self.assertDictEqual(self.cfg.construction_levels['dummy'],
                             {'mean_factor': 0.5,
                              'cv_factor': 0.5})

    def test_save_config(self):
        self.cfg.file_cfg += '.copy'
        self.cfg.save_config()


class TestInputChecks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        path = os.sep.join(__file__.split(os.sep)[:-1])
        scenario_filename1 = os.path.abspath(os.path.join(
            path, 'test_scenarios', 'test_house', 'test_house.cfg'))
        cls.cfg = Config(file_cfg=scenario_filename1)
        cls.path_cfg = os.path.dirname(os.path.realpath(scenario_filename1))

    def test_flag_pressure(self):
        """Check that each flag_pressure entry is a valid entry. """

        self.cfg.file_conn_groups = StringIO.StringIO("""
group_name,dist_order,dist_dir,damage_scenario,flag_pressure
sheeting,1,col,Loss of roof sheeting,dummy
batten,2,row,Loss of roof sheeting and purlins,cpe_str
                """)
        self.assertRaises(Exception, self.cfg.set_groups)

    def test_damage_scenario(self):
        """
        Check that each damage_scenario entry is also listed in damage_costing_data.csv
        """
        self.cfg.file_conn_groups = StringIO.StringIO("""
group_name,dist_order,dist_dir,damage_scenario,flag_pressure
sheeting,1,col,Loss of roof sheeting,cpe
batten,2,row,dummy,cpe_str
                """)

        df_groups = self.cfg.set_groups()
        self.assertRaises(Exception, self.cfg.set_costings, df_groups)

    def test_dist_dir(self):
        """
        Check that each damage_scenario entry is also listed in damage_costing_data.csv
        """
        self.cfg.file_conn_groups = StringIO.StringIO("""
group_name,dist_order,dist_dir,damage_scenario,flag_pressure
sheeting,1,col,Loss of roof sheeting,cpe
batten,2,dummy,Loss of roof sheeting & purlins,cpe_str
                """)
        self.assertRaises(Exception, self.cfg.set_groups)

    def test_conn_types(self):
        """
        Check that each strength_mean, strength_std, dead_load_mean, dead_load_std and costing_area entry is >=0
        """

        self.cfg.file_conn_types = StringIO.StringIO("""
type_name,strength_mean,strength_std,dead_load_mean,dead_load_std,group_name,costing_area
sheetinggable,1.54,0.16334,0.02025,0.0246,sheeting,0.405
sheetingeave,4.62,0.28292,0.02025,-0.0246,sheeting,0.405
sheetingcorner,2.31,0.2,0.01013,0.0246,sheeting,-0.225
        """)

        self.assertRaises(Exception, self.cfg.set_types)

    def test_conn_types2(self):
        """Check that each group_name entry is also listed in conn_groups.csv"""

        self.cfg.file_conn_types = StringIO.StringIO("""
type_name,strength_mean,strength_std,dead_load_mean,dead_load_std,group_name,costing_area
sheetinggable,1.54,0.16334,0.02025,0.0246,sheeting,0.405
sheetingeave,4.62,0.28292,0.02025,0.0246,sheeting,0.405
sheetingcorner,2.31,0.2,0.01013,0.0246,dummy,0.225
        """)

        df_types = self.cfg.set_types()
        df_groups = self.cfg.set_groups()

        self.assertRaises(Exception, self.cfg.set_connections,
                          (df_types, df_groups))

    def test_connections(self):
        """Check that each conn_type entry is also listed in conn_types.csv"""

        self.cfg.file_conn_types = StringIO.StringIO("""
type_name,strength_mean,strength_std,dead_load_mean,dead_load_std,group_name,costing_area
sheetinggable,1.54,0.16334,0.02025,0.0246,sheeting,0.405
sheetingeave,4.62,0.28292,0.02025,0.0246,sheeting,0.405
sheetingcorner,2.31,0.2,0.01013,0.0246,sheeting,0.225
        """)

        df_types = self.cfg.set_types()

        try:
            _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

            _file.writelines(['conn_name, type_name, zone_loc, section, coords,\n',
                              '1, dummy, A1, 0, 0, 0, 0.45, 0, 0.45, 0.45, 0, 0.45\n',
                              '2, sheetinggable, A2, 0, 0, 0.45, 0.45, 0.45, 0.45, 1.35, 0, 1.35\n',
                              '3, sheetinggable, A3, 0, 0, 1.35, 0.45, 1.35, 0.45, 2.25, 0, 2.25\n',
                              '4, sheetinggable, A4, 0, 0, 2.25, 0.45, 2.25, 0.45, 3.15, 0, 3.15\n'])

            _file.close()

            dump = self.cfg.read_file_connections(_file.name)

            self.assertRaises(Exception, self.cfg.process_read_file_connections,
                              (dump, df_types))

        finally:
            os.unlink(_file.name)

    def test_connections_coords(self):
        """Check that each conn_type entry is also listed in conn_types.csv"""

        self.cfg.file_conn_types = StringIO.StringIO("""
type_name,strength_mean,strength_std,dead_load_mean,dead_load_std,group_name,costing_area
sheetinggable,1.54,0.16334,0.02025,0.0246,sheeting,0.405
sheetingeave,4.62,0.28292,0.02025,0.0246,sheeting,0.405
sheetingcorner,2.31,0.2,0.01013,0.0246,sheeting,0.225
        """)

        _ = self.cfg.set_types()

        try:
            _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

            _file.writelines(['conn_name, type_name, zone_loc, section, coords,\n',
                              '1, sheetinggable, A1, 0, 0, 0, 0.45, 0, 0.45, 0.45, 0\n',
                              '2, sheetinggable, A2, 0, 0, 0.45, 0.45, 0.45, 0.45, 1.35, 0, 1.35\n',
                              '3, sheetinggable, A3, 0, 0, 1.35, 0.45, 1.35, 0.45, 2.25, 0, 2.25\n',
                              '4, sheetinggable, A4, 0, 0, 2.25, 0.45, 2.25, 0.45, 3.15, 0, 3.15\n'])

            _file.close()

            _ = self.cfg.read_file_connections(_file.name)

        finally:
            os.unlink(_file.name)

    def test_coverage_types(self):
        """Check that each failure_strength_out_mean entry is <=0. """

        self.cfg.file_coverage_types = StringIO.StringIO("""
Name,failure_momentum_mean,failure_momentum_std,failure_strength_in_mean,failure_strength_in_std,failure_strength_out_mean,failure_strength_out_std
Glass_annealed_6mm,0.05,0.01,100,0.0,-100,0.0
Glass_heatstr_6mm,0.06,0.012,100,0.0,-100,0.0
Glass_temp_6mm,0.1,0.02,100,0.0,-100,0.0
Cement_sheeting,142.2,28.44,100,0.0,-100,0.0
Weatherboard,142.2,28.44,100,0.0,-100,0.0
Brick_veneer,321.6,64.32,100,0.0,-100,0.0
Timber_door,142.2,28.44,100,0.0,100,0.0
        """)

        self.assertRaises(ValueError, self.cfg.set_coverages)

    def test_coverages1(self):
        """Check that each area entry is >=0"""

        self.cfg.file_coverages = StringIO.StringIO("""
name,description,wall_name,area,coverage_type
1,window,1,3.6,Glass_annealed_6mm
2,door,1,1.8,Timber_door
3,window,1,1.89,Glass_annealed_6mm
4,window,1,-1.89,Glass_annealed_6mm
        """)

        self.assertRaises(ValueError, self.cfg.set_coverages)

    def test_coverages2(self):
        """Check each coverage_type entry is also listed in coverage_types"""

        self.cfg.file_coverages = StringIO.StringIO("""
name,description,wall_name,area,coverage_type
1,window,1,3.6,Glass_annealed_6mm
2,door,1,1.8,Timber_door
3,window,1,1.89,Glass_annealed_6mm
4,window,1,1.89,dummy
        """)

        self.assertRaises(ValueError, self.cfg.set_coverages)

    def test_coverages3(self):
        """Check each wall_name is valid"""

        try:
            _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

            _file.writelines(['wind_dir,wall_name\n',
                              'S,1\n',
                              'SW,1,3\n',
                              'W,3\n',
                              'NW,3,5\n',
                              'N,5\n',
                              'NE,5,7\n',
                              'E,7\n',
                              'SE,1,7\n'])

            _file.close()

            self.cfg.file_front_facing_walls = _file.name

            self.cfg.file_coverages = StringIO.StringIO("""
name,description,wall_name,area,coverage_type
1,window,1,3.6,Glass_annealed_6mm
2,door,1,1.8,Timber_door
3,window,1,1.89,Glass_annealed_6mm
4,window,dummy,1.89,Glass_annealed_6mm
                """)

            self.assertRaises(ValueError, self.cfg.set_coverages)

        finally:
            os.unlink(_file.name)

    def test_coverages_cpe(self):
        """Check that each numerical entry is between -5 and +5."""

        self.cfg.file_coverages_cpe = StringIO.StringIO("""ID,S,SW,W,NW,N,NE,E,SE
1,2.4,2.4,2.4,2.4,2.4,2.4,2.4,2.4
2,1.69,1.69,1.69,1.69,1.69,1.69,1.69,1.69
3,-1.14,-1.14,-1.14,-1.14,-1.14,-1.14,-1.14,-1.14
4,-1.45,-1.45,-1.45,-1.45,-1.45,-1.45,-1.45,-1.45
5,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9
6,-0.55,-0.55,-0.55,-0.55,-0.55,-0.55,-0.55,-9.55
        """)

        # warning instead of ValueError
        self.cfg.set_coverages_cpe()
        # self.assertRaises(ValueError, self.cfg.set_coverages_cpe)

    def test_damage_costing_data(self):
        """Both envelope_factor and internal_factor formula_type entry are
        either 1 or 2"""

        self.cfg.file_damage_costing_data = StringIO.StringIO("""name,surface_area,envelope_repair_rate,envelope_factor_formula_type,envelope_coeff1,envelope_coeff2,envelope_coeff3,internal_repair_rate,internal_factor_formula_type,internal_coeff1,internal_coeff2,internal_coeff3,water_ingress_order
Loss of roof sheeting,116,72.4,1,0.3105,-0.8943,1.6015,0,1,0,0,0,6
Loss of roof sheeting & purlins,116,184.23,-1,0.3105,-0.8943,1.6015,0,1,0,0,0,7
        """)

        self.assertRaises(ValueError, self.cfg.read_damage_costing_data,
                          self.cfg.file_damage_costing_data)

    def test_footprint(self):
        """Check that each row has two numeric values"""

        self.cfg.file_footprint = StringIO.StringIO("""footprint_coord
-4.0,1.0, 3.0
4.0, 6.5
4.0, -6.5
-4.0, -6.5
        """)

        self.assertRaises(ValueError, self.cfg.set_footprint)

    # def test_front_facing_walls(self):
    #     """each wall_name entry is a valid wall number"""
    #
    #     try:
    #         _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
    #
    #         _file.writelines(['wind_dir,wall_name\n',
    #                           'S,1\n',
    #                           'SW,1,3\n',
    #                           'W,3\n',
    #                           'NW,3,5\n',
    #                           'N,5\n',
    #                           'NE,5,7\n',
    #                           'E,7\n',
    #                           'SE,1,77\n'])
    #
    #         _file.close()
    #
    #         self.cfg.file_front_facing_walls = _file.name
    #
    #         self.assertRaises(AssertionError, self.cfg.set_front_facing_walls)
    #
    #     finally:
    #         os.unlink(_file.name)

    def test_house_data(self):
        """Check that cpe_cov and cpe_str_cov are between 0 and 1.
           Check that Cpe_k and Cpe_str_k are >0.
        """

        self.cfg.file_house_data = StringIO.StringIO("""name,Group 4 House
replace_cost,198859.27
height,4.5
cpe_cv,0.12
cpe_k,0.1
cpe_str_cv,-0.07
length,12.6
width,9.0
cpe_str_k,0.1
""")
        self.assertRaises(AssertionError, self.cfg.set_house)

    def test_influences(self):
        """Check that each influence coefficient is between -10 and 10
        """

        try:
            _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

            _file.writelines(['conn_name, Zone, Coefficent\n',
                              '1, A1, 1\n',
                              '3, B1, 1\n',
                              '113, A13, 1, N15, 1\n',
                              '133, E13, 1, E14, 0,\n',
                              '136, F14, 1, F13, -110,\n', ])
            _file.close()

            self.assertRaises(AssertionError,
                              self.cfg.read_influences, _file.name)

        finally:
            os.unlink(_file.name)

    def test_influences1(self):
        """Check that each conn_name is also listed in connections.csv.
        """

        try:
            _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

            _file.writelines(['conn_name, Zone, Coefficent\n',
                              '1, A1, 1\n',
                              'dummy, B1, 1\n',
                              '3, A1, 1, B1, 1\n',
                              '133, A1, 1, B1, 0,\n',
                              '136, A1, 1, B1, 0,\n', ])
            _file.close()

            self.cfg.file_influences = _file.name
            self.assertRaises(AssertionError,
                              self.cfg.set_influences)

        finally:
            os.unlink(_file.name)

    def test_influences2(self):
        """Check that each conn_name is also listed in connections.csv.
        """

        try:
            _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

            _file.writelines(['conn_name, Zone, Coefficent\n',
                              '1, A1, 1\n',
                              '2, B1, 1\n',
                              '3, dummy, 1, B1, 1\n',
                              '133, A1, 1, B1, 0,\n',
                              '136, A1, 1, B1, 0,\n', ])
            _file.close()

            self.cfg.file_influences = _file.name
            self.assertRaises(AssertionError,
                              self.cfg.set_influences)

        finally:
            os.unlink(_file.name)

    def test_influence_patches(self):
        """Check that each influence coefficient is between -10 and 10
        """

        try:
            _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

            _file.writelines(['Damaged connection,Connection,Zone,Inf factor,,\n',
                              '113,113, A13,0,N15,0\n',
                              '113,114,N15,1,,\n',
                              '113,121,B13,1,,\n',
                              '120,120,A14,0,G15,0\n',
                              '120,119,G15,1,,\n',
                              '120,124,A14,100.0,,\n'])
            _file.close()

            self.assertRaises(AssertionError,
                              self.cfg.read_influence_patches, _file.name)

        finally:
            os.unlink(_file.name)

    def test_influence_patches1(self):
        """Check that each conn_name is also listed in connections.csv.
        """

        try:
            _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

            _file.writelines(['Damaged connection,Connection,Zone,Inf factor,,\n',
                              '113,113,A1,0,A2,0\n',
                              'dummy,114,B1,1,,\n',
                              '113,121,B1,1,,\n',
                              '120,120,B1,0,B2,0\n',
                              '120,119,A3,1,,\n',
                              '120,124,A3,1.0,,\n'])
            _file.close()

            self.cfg.file_influence_patches = _file.name
            self.assertRaises(AssertionError,
                              self.cfg.set_influence_patches)

        finally:
            os.unlink(_file.name)

    def test_influence_patches2(self):
        """Check that each conn_name is also listed in connections.csv.
        """

        try:
            _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

            _file.writelines(['Damaged connection,Connection,Zone,Inf factor,,\n',
                              '113,113,A1,0,A2,0\n',
                              '113,dummy,B1,1,,\n',
                              '113,121,B1,1,,\n',
                              '120,120,B1,0,B2,0\n',
                              '120,119,A3,1,,\n',
                              '120,124,A3,1.0,,\n'])
            _file.close()

            self.cfg.file_influence_patches = _file.name
            self.assertRaises(AssertionError,
                              self.cfg.set_influence_patches)

        finally:
            os.unlink(_file.name)

    def test_influence_patches3(self):
        """Check that each conn_name is also listed in connections.csv.
        """

        try:
            _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

            _file.writelines(['Damaged connection,Connection,Zone,Inf factor,,\n',
                              '113,113,A1,0,A2,0\n',
                              '113,113,dummy,1,,\n',
                              '113,121,B1,1,,\n',
                              '120,120,B1,0,B2,0\n',
                              '120,119,A3,1,,\n',
                              '120,124,A3,1.0,,\n'])
            _file.close()

            self.cfg.file_influence_patches = _file.name
            self.assertRaises(AssertionError,
                              self.cfg.set_influence_patches)

        finally:
            os.unlink(_file.name)

    def test_water_ingress_costing_data(self):
        """Check that each name is also listed in damage_costing_data.csv,
        other than WI only"""

        self.cfg.file_water_ingress_costing_data = StringIO.StringIO("""name, water_ingress, base_cost, formula_type, coeff1, coeff2, coeff3
Loss of roof sheeting,0,0,1,0,0,1
Loss of roof sheeting & purlins,0,0,1,0,0,1
Loss of roof structure,0,0,1,0,0,1
Wall debris damage,0,0,1,0,0,1
Loss of wall cladding,0,0,1,0,0,1
Wall collapse,0,0,1,0,0,1
Wall racking,0,0,1,0,0,1
WI only,0,0,1,0,0,1
dummy,0,0,1,0,0,1
        """)

        self.assertRaises(AssertionError, self.cfg.set_water_ingress_costings)

    def test_water_ingress_costing_data1(self):
        """Check that each name is also listed in damage_costing_data.csv,
        other than WI only"""
        self.cfg.file_water_ingress_costing_data = StringIO.StringIO("""name, water_ingress, base_cost, formula_type, coeff1, coeff2, coeff3
Loss of roof sheeting,0,0,1,0,0,1
Loss of roof sheeting & purlins,0,0,1,0,0,1
Loss of roof structure,0,0,1,0,0,1
Wall debris damage,0,0,1,0,0,1
Loss of wall cladding,0,0,1,0,0,1
Wall collapse,0,0,1,0,0,1
Wall racking,0,0,1,0,0,1
        """)

        self.assertRaises(AssertionError, self.cfg.set_water_ingress_costings)

    def test_water_ingress_costing_data2(self):
        """Check that each formula_type entry is either 1 or 2"""
        self.cfg.file_water_ingress_costing_data = StringIO.StringIO("""name, water_ingress, base_cost, formula_type, coeff1, coeff2, coeff3
Loss of roof sheeting,0,0,1,0,0,1
Loss of roof sheeting & purlins,0,0,1,0,0,1
Loss of roof structure,0,0,1,0,0,1
Wall debris damage,0,0,1,0,0,1
Loss of wall cladding,0,0,1,0,0,1
Wall collapse,0,0,1,0,0,1
Wall racking,0,0,1,0,0,1
WI only,0,0,dummy,0,0,1
        """)

        self.assertRaises(ValueError,
                          self.cfg.read_water_ingress_costing_data,
                          self.cfg.file_water_ingress_costing_data)

    def test_zones(self):
        """
        Check that each area is >=0
        """

        for item in ['cpe_mean', 'cpe_str_mean', 'cpe_eave_mean', 'edge']:
            file_org = os.path.join(
                self.path_cfg, 'input', 'house', 'zones_{}.csv'.format(item))
            setattr(self.cfg, 'file_zones_{}'.format(item), file_org)

        try:
            _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

            _file.writelines(['name,area,cpi_alpha,wall_dir\n',
                              'A1,0.2025,0.0,0\n',
                              'A2,0.405,0.5,0\n',
                              'A3,0.405,1.0,0\n',
                              'A4,0.405,1.0,0\n',
                              'A5,0.405,1.0,0\n',
                              'A6,0.2475,1.0,0\n',
                              'A7,0.2475,1.0,0\n',
                              'A8,0.405,1.0,0\n',
                              'A9,-0.405,1.0,0\n'])

            _file.close()

            self.assertRaises(ValueError, self.cfg.read_file_zones, _file.name)

        finally:
            os.unlink(_file.name)

    def test_zones1(self):
        """Check that each cpi_alpha is between 0 and 1."""

        for item in ['cpe_mean', 'cpe_str_mean', 'cpe_eave_mean', 'edge']:
            file_org = os.path.join(
                self.path_cfg, 'input', 'house', 'zones_{}.csv'.format(item))
            setattr(self.cfg, 'file_zones_{}'.format(item), file_org)

        try:
            _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

            _file.writelines(['name,area,cpi_alpha,wall_dir\n',
                              'A1,0.2025,0.0,0\n',
                              'A2,0.405,0.5,0\n',
                              'A3,0.405,1.0,0\n',
                              'A4,0.405,1.0,0\n',
                              'A5,0.405,1.0,0\n',
                              'A6,0.2475,1.0,0\n',
                              'A7,0.2475,1.0,0\n',
                              'A8,0.405,1.0,0\n',
                              'A9,0.405,1.1,0\n'])

            _file.close()

            self.assertRaises(ValueError, self.cfg.read_file_zones, _file.name)

        finally:
            os.unlink(_file.name)

    def test_zones2(self):
        """Check that each zone has either 3 or 4 pairs of coordinates."""

        for item in ['cpe_mean', 'cpe_str_mean', 'cpe_eave_mean', 'edge']:
            file_org = os.path.join(
                self.path_cfg, 'input', 'house', 'zones_{}.csv'.format(item))
            setattr(self.cfg, 'file_zones_{}'.format(item), file_org)

        try:
            _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

            _file.writelines(['name,area,cpi_alpha,wall_dir,coords\n',
                              'A1,0.2025,0.0,0, 0, 0, 0.45, 0, 0.45, 0.45, 0, 0.45\n',
                              'A2,0.405,0.5,0,  0, 0.45, 0, 0.45, 0.45, 0, 0.45, 0.0\n',
                              'A3,0.405,1.0,0, 0, 0.45, 0, 0.45, 0.45, 0, 0.45, 0.0\n',
                              'A4,0.405,1.0,0, 0, 0.45, 0, 0.45, 0.45, 0, 0.45, 0.0\n',
                              'A5,0.405,1.0,0, 0, 0.45, 0, 0.45, 0.45, 0, 0.45, 0.0\n',
                              'A6,0.2475,1.0,0, 0, 0.45, 0, 0.45, 0.45, 0, 0.45, 0.0\n',
                              'A7,0.2475,1.0,0, 0, 0.45, 0, 0.45, 0.45, 0, 0.45, 0.0\n',
                              'A8,0.405,1.0,0, 0, 0.45, 0, 0.45, 0.45, 0, 0.45, 0.0\n',
                              'A9,0.405,1.0,0, 0, 0.45, 0, 0.45, 0.45, 0, 0.45, 0.0, 9.0\n'])

            _file.close()

            self.cfg.read_file_zones(_file.name)

        finally:
            os.unlink(_file.name)

    def test_zones3(self):
        """Check that each zone has either 3 or 4 pairs of coordinates."""

        for item in ['cpe_mean', 'cpe_str_mean', 'cpe_eave_mean', 'edge']:
            file_org = os.path.join(
                self.path_cfg, 'input', 'house', 'zones_{}.csv'.format(item))
            setattr(self.cfg, 'file_zones_{}'.format(item), file_org)

        try:
            _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

            _file.writelines(['name,area,cpi_alpha,wall_dir,coords\n',
                              'A1,0.2025,0.0,0, 0, 0, 0.45, 0, 0.45, 0.45, 0, 0.45\n',
                              'A2,0.405,0.5,0,  0, 0.45, 0, 0.45, 0.45, 0, 0.45, 0.0\n',
                              'A3,0.405,1.0,0, 0, 0.45, 0, 0.45, 0.45, 0, 0.45, 0.0\n',
                              'A4,0.405,1.0,0, 0, 0.45, 0, 0.45, 0.45, 0, 0.45, 0.0\n',
                              'A5,0.405,1.0,0, 0, 0.45, 0, 0.45, 0.45, 0, 0.45, 0.0\n',
                              'A6,0.2475,1.0,0, 0, 0.45, 0, 0.45, 0.45, 0, 0.45, 0.0\n',
                              'A7,0.2475,1.0,0, 0, 0.45, 0, 0.45, 0.45, 0, 0.45, 0.0\n',
                              'A8,0.405,1.0,0, 0, 0.45, 0, 0.45, 0.45, 0, 0.45, 0.0\n',
                              'A9,0.405,1.0,0, 0, 0.45, 0, 0.45\n'])

            _file.close()

            self.cfg.read_file_zones(_file.name)

        finally:
            os.unlink(_file.name)

    def test_zones_cpe_mean(self):
        """Check that each name is also listed in zones.csv. """

        try:
            _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

            _file.writelines(['name,area,cpi_alpha,wall_dir\n',
                              'A1,0.2025,0.0,0\n',
                              'A2,0.405,0.5,0\n',
                              'A3,0.405,1.0,0\n',
                              'A4,0.405,1.0,0\n'])

            _file.close()

            self.cfg.file_zones = _file.name

            self.cfg.file_zones_cpe_mean = StringIO.StringIO("""name,S,SW,W,NW,N,NE,E,SE
A1,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A2,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A3,-0.908,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
dummy,-0.5,-1.45,-0.85,-0.75,-0.5,-0.4,-0.2,-0.4
            """)

            self.cfg.file_zones_cpe_eave_mean = StringIO.StringIO("""name,S,SW,W,NW,N,NE,E,SE
A1,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A2,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A3,-0.908,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A4,-0.5,-1.45,-0.85,-0.75,-0.5,-0.4,-0.2,-0.4
                        """)

            self.cfg.file_zones_cpe_str_mean = StringIO.StringIO("""name,S,SW,W,NW,N,NE,E,SE
A1,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A2,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A3,-0.908,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A4,-0.5,-1.45,-0.85,-0.75,-0.5,-0.4,-0.2,-0.4
                        """)

            self.cfg.file_zones_edge = StringIO.StringIO("""name,S,SW,W,NW,N,NE,E,SE
A1,1,1,1,0,0,0,0,0
A2,1,1,1,0,0,0,0,0
A3,1,1,1,0,0,0,0,0
A4,0,1,0,0,0,0,0,0
""")
            self.assertRaises(AssertionError, self.cfg.set_zones)

        finally:
            os.unlink(_file.name)

    def test_zones_cpe_mean1(self):
        """Check that each zone has values for 8 directions."""

        try:
            _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

            _file.writelines(['name,area,cpi_alpha,wall_dir\n',
                              'A1,0.2025,0.0,0\n',
                              'A2,0.405,0.5,0\n',
                              'A3,0.405,1.0,0\n',
                              'A4,0.405,1.0,0\n'])

            _file.close()

            self.cfg.file_zones = _file.name

            # missing value @ A1
            self.cfg.file_zones_cpe_mean = StringIO.StringIO("""name,S,SW,W,NW,N,NE,E,SE
A1,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,,-0.4
A2,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A3,-0.908,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A4,-0.5,-1.45,-0.85,-0.75,-0.5,-0.4,-0.2,-0.4
            """)

            self.cfg.file_zones_cpe_eave_mean = StringIO.StringIO("""name,S,SW,W,NW,N,NE,E,SE
A1,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A2,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A3,-0.908,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A4,-0.5,-1.45,-0.85,-0.75,-0.5,-0.4,-0.2,-0.4
                        """)

            self.cfg.file_zones_cpe_str_mean = StringIO.StringIO("""name,S,SW,W,NW,N,NE,E,SE
A1,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A2,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A3,-0.908,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A4,-0.5,-1.45,-0.85,-0.75,-0.5,-0.4,-0.2,-0.4
                        """)

            self.cfg.file_zones_edge = StringIO.StringIO("""name,S,SW,W,NW,N,NE,E,SE
A1,1,1,1,0,0,0,0,0
A2,1,1,1,0,0,0,0,0
A3,1,1,1,0,0,0,0,0
A4,0,1,0,0,0,0,0,0
""")
            self.assertRaises(ValueError, self.cfg.set_zones)

        finally:
            os.unlink(_file.name)

    def test_zones_cpe_mean2(self):
        """Check that each Cpe is between -5 and 5."""

        try:
            _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

            _file.writelines(['name,area,cpi_alpha,wall_dir\n',
                              'A1,0.2025,0.0,0\n',
                              'A2,0.405,0.5,0\n',
                              'A3,0.405,1.0,0\n',
                              'A4,0.405,1.0,0\n'])

            _file.close()

            self.cfg.file_zones = _file.name

            # wrong value @ A1
            self.cfg.file_zones_cpe_mean = StringIO.StringIO("""name,S,SW,W,NW,N,NE,E,SE
A1,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.4,-99
A2,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A3,-0.908,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A4,-0.5,-1.45,-0.85,-0.75,-0.5,-0.4,-0.2,-0.4
            """)

            self.cfg.file_zones_cpe_eave_mean = StringIO.StringIO("""name,S,SW,W,NW,N,NE,E,SE
A1,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A2,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A3,-0.908,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A4,-0.5,-1.45,-0.85,-0.75,-0.5,-0.4,-0.2,-0.4
                        """)

            self.cfg.file_zones_cpe_str_mean = StringIO.StringIO("""name,S,SW,W,NW,N,NE,E,SE
A1,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A2,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A3,-0.908,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A4,-0.5,-1.45,-0.85,-0.75,-0.5,-0.4,-0.2,-0.4
                        """)

            self.cfg.file_zones_edge = StringIO.StringIO("""name,S,SW,W,NW,N,NE,E,SE
A1,1,1,1,0,0,0,0,0
A2,1,1,1,0,0,0,0,0
A3,1,1,1,0,0,0,0,0
A4,0,1,0,0,0,0,0,0
""")
            # warning instead of ValueError
            self.cfg.set_zones()
            # self.assertRaises(ValueError, self.cfg.set_zones)

        finally:
            os.unlink(_file.name)

    def test_zones_cpe_mean3(self):
        """Check that edge value is either 0 or 1"""

        try:
            _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

            _file.writelines(['name,area,cpi_alpha,wall_dir\n',
                              'A1,0.2025,0.0,0\n',
                              'A2,0.405,0.5,0\n',
                              'A3,0.405,1.0,0\n',
                              'A4,0.405,1.0,0\n'])

            _file.close()

            self.cfg.file_zones = _file.name

            self.cfg.file_zones_cpe_mean = StringIO.StringIO("""name,S,SW,W,NW,N,NE,E,SE
A1,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.4,-0.4
A2,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A3,-0.908,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A4,-0.5,-1.45,-0.85,-0.75,-0.5,-0.4,-0.2,-0.4
            """)

            self.cfg.file_zones_cpe_eave_mean = StringIO.StringIO("""name,S,SW,W,NW,N,NE,E,SE
A1,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A2,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A3,-0.908,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A4,-0.5,-1.45,-0.85,-0.75,-0.5,-0.4,-0.2,-0.4
                        """)

            self.cfg.file_zones_cpe_str_mean = StringIO.StringIO("""name,S,SW,W,NW,N,NE,E,SE
A1,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A2,-1.25,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A3,-0.908,-1.45,-1.3,-0.75,-0.35,-0.4,-0.2,-0.4
A4,-0.5,-1.45,-0.85,-0.75,-0.5,-0.4,-0.2,-0.4
                        """)
            # wrong value @ A4
            self.cfg.file_zones_edge = StringIO.StringIO("""name,S,SW,W,NW,N,NE,E,SE
A1,1,1,1,0,0,0,0,0
A2,1,1,1,0,0,0,0,0
A3,1,1,1,0,0,0,0,0
A4,0,1,0,0,0,0,0,-1
""")
            self.assertRaises(ValueError, self.cfg.set_zones)

        finally:
            os.unlink(_file.name)

if __name__ == '__main__':
    unittest.main()
