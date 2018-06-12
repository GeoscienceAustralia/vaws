import unittest
import numpy as np
import StringIO
import pandas as pd

from vaws.model.zone import Zone, str2num, get_grid_from_zone_location, get_zone_location_from_grid


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.rnd_state = np.random.RandomState(seed=13)

        item = dict(area=0.2025,
                    cpi_alpha=0.0,
                    wind_dir_index=0,
                    shielding_multiplier=1.0,
                    building_spacing=0,
                    flag_differential_shielding=False,
                    cpe_mean={k: -0.1 for k in range(8)},
                    cpe_str_mean={k: -0.05 for k in range(8)},
                    cpe_eave_mean={k: 0.0 for k in range(8)},
                    is_roof_edge={k: 1 for k in range(8)},
                    cpe_cv=0.12,
                    cpe_k=0.1,
                    big_a=0.486,
                    big_b=1.145,
                    cpe_str_cv=0.07,
                    cpe_str_k=0.1,
                    big_a_str=0.486,
                    big_b_str=1.145,
                    rnd_state=cls.rnd_state)

        cls.zone = Zone(name='N12', **item)


    def test_get_grid(self):
        col, row = get_grid_from_zone_location(self.zone.name)
        self.assertEquals(col, 13)  # N
        self.assertEquals(row, 11)  # 12
        # self.assertEquals(self.zone.get_zone_location_from_grid((col, row)),
        #                  self.zone.name)

    # def test_is_wall(self):
    #     self.assertEqual(self.zone.is_wall_zone, False)

    def test_calc_zone_pressures(self):
        # self.zone.sample_cpe(

        self.assertAlmostEqual(self.zone.cpe, -0.1084, places=4)
        self.assertAlmostEqual(self.zone.cpe_eave, 0.0000, places=4)
        self.assertAlmostEqual(self.zone.cpe_str, -0.0474, places=4)

        wind_speed = 40.0
        mzcat = 0.9235
        qz = 0.6 * 1.0e-3 * (wind_speed * mzcat) ** 2

        self.zone.calc_zone_pressure(cpi=0.0, qz=qz, combination_factor=1.0)

        self.assertAlmostEqual(self.zone.pressure_cpe, -0.0888, places=4)
        self.assertAlmostEqual(self.zone.pressure_cpe_str, -0.0388, places=4)

    # def test_num2str(self):
    #     self.assertEqual(num2str(1), 'A')
    #     self.assertEqual(num2str(27), 'AA')

    def test_str2num(self):
        self.assertEqual(str2num('B'), 2)
        self.assertEqual(str2num('AB'), 28)

    def test_set_differential_shieding(self):

        reference_data = StringIO.StringIO("""Differential shielding flag,Building spacing,Ms,Zone_edge_flag,Expected factor,Comments
TRUE, 40, 1, 1, 1, No shielding
TRUE, 40, 1, 0, 1, No shielding
TRUE, 40, 0.95, 1, 1, JDH recommendation 1 - retain shielding for leading edges of upwind roofs
TRUE, 40, 0.95, 0, 1.108, JDH recommendation 1 - neglect shielding to all other surfaces
TRUE, 40, 0.85, 1, 1, JDH recommendation 1 - retain shielding for leading edges of upwind roofs
TRUE, 40, 0.85, 0, 1.384, JDH recommendation 1 - neglect shielding to all other surfaces
TRUE, 20, 1, 1, 1, No shielding
TRUE, 20, 1, 0, 1, No shielding
TRUE, 20, 0.95, 1, 0.709, JDH recommendation 3 - reduce Ms to 0.8 for leading edges of upwind roofs
TRUE, 20, 0.95, 0, 1, JDH recommendation 3 - retain Ms = 0.95 for all other surfaces
TRUE, 20, 0.85, 1, 0.678, JDH recommendation 2 - reduce Ms to 0.7 for leading edges of upwind roofs
TRUE, 20, 0.85, 0, 1, JDH recommendation 2 - retain Ms = 0.85 for all other surfaces
FALSE, 40, 1, 1, 1, Diff shielding not considered
FALSE, 40, 1, 0, 1, Diff shielding not considered
FALSE, 40, 0.95, 1, 1, Diff shielding not considered
FALSE, 40, 0.95, 0, 1, Diff shielding not considered
FALSE, 40, 0.85, 1, 1, Diff shielding not considered
FALSE, 40, 0.85, 0, 1, Diff shielding not considered
FALSE, 20, 1, 1, 1, Diff shielding not considered
FALSE, 20, 1, 0, 1, Diff shielding not considered
FALSE, 20, 0.95, 1, 1, Diff shielding not considered
FALSE, 20, 0.95, 0, 1, Diff shielding not considered
FALSE, 20, 0.85, 1, 1, Diff shielding not considered
FALSE, 20, 0.85, 0, 1, Diff shielding not considered""")

        reference_data = pd.read_csv(reference_data)

        for irow, row in reference_data.iterrows():

            item = dict(wind_dir_index=0,
                        shielding_multiplier=row['Ms'],
                        building_spacing=row['Building spacing'],
                        flag_differential_shielding = row['Differential shielding flag'],
                        is_roof_edge={0: row['Zone_edge_flag']})

            _zone = Zone(name='dummy', **item)
            outcome = _zone.differential_shielding

            try:
                self.assertAlmostEqual(outcome, row['Expected factor'], places=3)
            except AssertionError:
                print('{}: expecting {} but returned {}'.format(irow, row['Expected factor'], outcome))

if __name__ == '__main__':
    unittest.main()

# suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
# unittest.TextTestRunner(verbosity=2).run(suite)
