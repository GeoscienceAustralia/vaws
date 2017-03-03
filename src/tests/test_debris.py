import unittest
import os
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from shapely.geometry import Point, Polygon

from vaws.scenario import Scenario
from vaws.debris import Debris


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = '/'.join(__file__.split('/')[:-1])
        cfg_file = os.path.join(path, '../../scenarios/test_roof_sheeting2.cfg')
        cls.cfg = Scenario(cfg_file=cfg_file)

        cls.footprint_inst = Polygon([(-6.5, 4.0), (6.5, 4.0), (6.5, -4.0),
                                      (-6.5, -4.0), (-6.5, 4.0)])

        ref04 = Polygon([(-24.0, 6.5), (4.0, 6.5), (4.0, -6.5),
                         (-24.0, -6.5), (-24.0, 6.5)])

        ref37 = Polygon([(-1.77, 7.42), (7.42, -1.77), (1.77, -7.42),
                         (-27.42, 1.77), (-1.77, 7.42)])

        ref26 = Polygon([(-26.5, 4.0), (6.5, 4.0), (6.5, -4.0),
                         (-26.5, -4.0), (-26.5, 4.0)])

        ref15 = Polygon([(-27.42, -1.77), (1.77, 7.42), (7.42, 1.77),
                         (-1.77, -7.42), (-27.42, -1.77)])

        cls.ref_footprint = {0: ref04, 1: ref15, 2: ref26, 3: ref37,
                             4: ref04, 5: ref15, 6: ref26, 7: ref37}

    # def test_debris_types(self)
    #     expectednames = ['Compact', 'Sheet', 'Rod']
    #     expectedcdavs = [0.65, 0.9, 0.8]
    #     i = 0
    #     for dt in model_db.qryDebrisTypes():
    #         self.assertEquals(dt[0], expectednames[i])
    #         self.assertAlmostEquals(dt[1], expectedcdavs[i])
    #         i += 1
    #
    # def test_debris_regions(self):
    #     expectednames = ['Capital_city', 'Tropical_town']
    #     expectedalphas = [0.1585, 0.103040002286434]
    #     i = 0
    #     for r in qryDebrisRegions(model_db):
    #         self.assertEquals(r.name, expectednames[i])
    #         self.assertAlmostEquals(r.alpha, expectedalphas[i])
    #         self.assertTrue(r.cr < r.rr)
    #         self.assertTrue(r.rr < r.pr)
    #         i += 1
    #     self.assertEquals(qryDebrisRegionByName('Foobar', model_db), None)
    #     self.assertNotEquals(qryDebrisRegionByName('Capital_city', model_db), None)
    #
    # def test_with_render(self):
    #     # this is the minimum case
    #     house_inst = house.queryHouseWithName('Group 4 House', model_db)
    #     region_name = 'Capital_city'
    #     v = 55.0
    #     mgr = DebrisManager(model_db, house_inst, region_name)
    #     mgr.set_wind_direction_index(1)
    #     mgr.run(v, True)
    #     mgr.render(v)

    def test_create_sources(self):

        self.cfg.debris_radius = 100.0
        self.cfg.debris_angle = 45.0
        self.cfg.building_spacing = 20.0
        self.cfg.flags['debris_staggered_sources'] = False

        sources1 = Debris.create_sources(self.cfg.debris_radius,
                                         self.cfg.debris_angle,
                                         self.cfg.building_spacing,
                                         False)
        self.assertEquals(len(sources1), 13)

        plt.figure()
        for source in sources1:
            plt.scatter(source.x, source.y)
        # plt.show()
        plt.pause(1.0)
        plt.close()

        # staggered source
        sources2 = Debris.create_sources(self.cfg.debris_radius,
                                         self.cfg.debris_angle,
                                         self.cfg.building_spacing,
                                         True)

        self.assertEquals(len(sources2), 15)

        plt.figure()
        for source in sources2:
            plt.scatter(source.x, source.y)
        # plt.show()
        plt.pause(1.0)
        plt.close()

    def test_footprint04(self):

        _debris = Debris(cfg=self.cfg)
        _debris.footprint = (self.footprint_inst, 0)

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        p = PolygonPatch(_debris.footprint, fc='red', alpha=0.5)
        ax.add_patch(p)
        x, y = self.ref_footprint[0].exterior.xy
        ax.plot(x, y, 'b-')
        ax.set_xlim([-40, 20])
        ax.set_ylim([-20, 20])
        plt.title('Wind direction: 0')
        # plt.show()
        plt.pause(1.0)
        plt.close()

    def test_footprint15(self):

        _debris = Debris(cfg=self.cfg)
        _debris.footprint = (self.footprint_inst, 1)

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        p = PolygonPatch(_debris.footprint, fc='red', alpha=0.5)
        ax.add_patch(p)
        x, y = self.ref_footprint[1].exterior.xy
        ax.plot(x, y, 'b-')
        ax.set_xlim([-40, 20])
        ax.set_ylim([-20, 20])
        plt.title('Wind direction: 1')
        # plt.show()
        plt.pause(1.0)
        plt.close()

    def test_footprint26(self):

        _debris = Debris(cfg=self.cfg)
        _debris.footprint = (self.footprint_inst, 2)

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        p = PolygonPatch(_debris.footprint, fc='red', alpha=0.5)
        ax.add_patch(p)
        x, y = self.ref_footprint[2].exterior.xy
        ax.plot(x, y, 'b-')
        ax.set_xlim([-40, 20])
        ax.set_ylim([-20, 20])
        plt.title('Wind direction: 2')
        # plt.show()
        plt.pause(1.0)
        plt.close()

    def test_footprint37(self):

        _debris = Debris(cfg=self.cfg)
        _debris.footprint = (self.footprint_inst, 3)

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        p = PolygonPatch(_debris.footprint, fc='red', alpha=0.5)
        ax.add_patch(p)
        x, y = self.ref_footprint[3].exterior.xy
        ax.plot(x, y, 'b-')
        ax.set_xlim([-40, 20])
        ax.set_ylim([-20, 20])
        plt.title('Wind direction: 3')
        # plt.show()
        plt.pause(1.0)
        plt.close()

    def test_footprint_non_rect(self):
        """ Not working yet """

        footprint_inst = Polygon(
            [(-6.5, 4.0), (6.5, 4.0), (6.5, 0.0), (0.0, 0.0),
             (0.0, -4.0), (-6.5, -4.0), (-6.5, 4.0)])

        _debris = Debris(cfg=self.cfg)
        _debris.footprint = (footprint_inst, 0)

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        p = PolygonPatch(_debris.footprint, fc='red', alpha=0.5)
        ax.add_patch(p)
        x, y = footprint_inst.exterior.xy
        ax.plot(x, y, 'b-')
        ax.set_xlim([-40, 20])
        ax.set_ylim([-20, 20])
        plt.title('Wind direction: 0')
        # plt.show()
        plt.pause(1.0)
        plt.close()

    def test_contains(self):
        rect = Polygon([(-15, 4), (15, 4), (15, -4), (-15, -4)])
        self.assertTrue(rect.contains(Point(0, 0)))
        self.assertFalse(rect.contains(Point(-100, -1.56)))
        self.assertFalse(rect.contains(Point(10.88, 4.514)))
        self.assertFalse(rect.contains(Point(7.773, 12.66)))

if __name__ == '__main__':
    unittest.main()

# suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
# unittest.TextTestRunner(verbosity=2).run(suite)
