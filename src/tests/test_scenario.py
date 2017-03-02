import unittest
import os

from vaws.scenario import Scenario


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # cls.output_path = './output'

        path = '/'.join(__file__.split('/')[:-1])
        scenario_filename1 = os.path.abspath(
            os.path.join(path, '../../scenarios/test_scenario1.cfg'))

        cls.cfg = Scenario(cfg_file=scenario_filename1)

        # cls.scenario_filename3 = os.path.abspath(os.path.join(path,
        #                                          '../../scenarios/test.cfg'))

    def test_debris(self):
        # s1.storeToCSV(self.file3)
        self.assertEquals(self.cfg.flags['debris'], False)

    def test_wind_directions(self):
        self.cfg.set_wind_dir_index('Random')
        self.assertEqual(self.cfg.wind_dir_index, 8)

        self.cfg.set_wind_dir_index('SW')
        self.assertEqual(self.cfg.wind_dir_index, 1)

    def test_water_ingress(self):
        self.assertFalse(self.cfg.flags['water_ingress'])
        self.cfg.flags['water_ingress'] = True
        self.assertTrue(self.cfg.flags['water_ingress'])

    def test_ctgenables(self):
        self.assertTrue(
            self.cfg.flags['conn_type_group_{}'.format('sheeting')])
        self.cfg.setOptCTGEnabled('batten', False)
        self.assertFalse(
            self.cfg.flags['conn_type_group_{}'.format('batten')])

        # s.storeToCSV(self.scenario_filename3)
        # s2 = Scenario(cfg_file=self.scenario_filename3)
        # self.assertFalse(s2.flags['conn_type_group_{}'.format('batten')])
        # self.assertTrue(s2.flags['conn_type_group_{}'.format('sheeting')])

        # def test_construction_levels(self):
        #     s1 = Scenario(cfg_file=self.scenario_filename1)
        #     s1.setConstructionLevel('low', 0.33, 0.42, 0.78)

        # s1.storeToCSV(self.scenario_filename3)
        # s = Scenario(cfg_file=self.scenario_filename3)
        # self.assertAlmostEquals(
        #     s.construction_levels['low']['mean_factor'], 0.42)

if __name__ == '__main__':
    unittest.main()

# suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
# unittest.TextTestRunner(verbosity=2).run(suite)