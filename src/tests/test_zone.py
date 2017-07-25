import unittest
import numpy as np

from vaws.zone import Zone, str2num


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.zone = Zone(zone_name='N12')

        cls.zone.area = 0.2025
        cls.zone.cpi_alpha = 0.0

        for idx in range(8):
            cls.zone.cpe_mean[idx] = -0.1
            cls.zone.cpe_str_mean[idx] = -0.05
            cls.zone.cpe_eave_mean[idx] = 0.0
            cls.zone.is_roof_edge[idx] = 1

        cls.rnd_state = np.random.RandomState(seed=13)

    def test_get_grid(self):
        col, row = self.zone.get_grid_from_zone_location(self.zone.name)
        self.assertEquals(col, 13)  # N
        self.assertEquals(row, 11)  # 12
        # self.assertEquals(self.zone.get_zone_location_from_grid((col, row)),
        #                  self.zone.name)

    def test_is_wall(self):
        self.assertEqual(self.zone.is_wall_zone, False)

    def test_calc_zone_pressures(self):
        self.zone.sample_zone_cpe(wind_dir_index=3,
                                  cpe_cov=0.12,
                                  cpe_k=0.1,
                                  cpe_str_cov=0.07,
                                  big_a=0.486,
                                  big_b=1.145,
                                  rnd_state=self.rnd_state)

        self.assertAlmostEqual(self.zone.cpe, -0.1084, places=4)
        self.assertAlmostEqual(self.zone.cpe_eave, 0.0000, places=4)
        self.assertAlmostEqual(self.zone.cpe_str, -0.0474, places=4)

        wind_speed = 40.0
        mzcat = 0.9235
        qz = 0.6 * 1.0e-3 * (wind_speed * mzcat) ** 2

        self.zone.calc_zone_pressures(wind_dir_index=3,
                                      cpi=0.0,
                                      qz=qz,
                                      Ms=1.0,
                                      building_spacing=0)

        self.assertAlmostEqual(self.zone.pressure, -0.1276, places=4)

    # def test_num2str(self):
    #     self.assertEqual(num2str(1), 'A')
    #     self.assertEqual(num2str(27), 'AA')

    def test_str2num(self):
        self.assertEqual(str2num('B'), 2)
        self.assertEqual(str2num('AB'), 28)

if __name__ == '__main__':
    unittest.main()

# suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
# unittest.TextTestRunner(verbosity=2).run(suite)
