"""
Debris Module - adapted from JDH Consulting and Martin's work
    - given sim:wind_speed, sim:wind_dir,
            scen:h:target_height, scen:h:target_width,
            scen:building_spacing, scen:debris_angle, scen:debris_radius,
            scen:flighttime_mean, scen:flighttime_stddev
    - generate sources of debris.
    - generate debris items from those sources.
    - track items and sample target collisions.
    - handle collisions (impact)
"""
import numpy as np
from shapely.geometry import Point
import math

import curve
import engine
from rect import Rect, Point

# lookup table mapping (0-7) to wind direction desc
# dirs = ['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE', 'Random']

# FIXME!!! mapping wind direciton to facing wall will provided as an input
# lookup table mapping wind direction (1-8) to list of front facing wall
# directions
# facing = {1: [1], 2: [1, 3], 3: [3], 4: [3, 5], 5: [5], 6: [5, 7], 7: [7],
#           8: [1, 7]}


# lookup table for source render colors
# src_cols = ('y', 'm', 'c', 'g', 'b', 'k')


# class DebrisItem(object):
#     def __init__(self, col, shape):
#         self.col = col
#         self.shape = shape


# class DebrisSource(object):
#     def __init__(self, x, y, col):
#         self.yord = float(y)
#         self.xord = float(x)
#         self.col = col


class Debris(object):
    def __init__(self,
                 cfg,
                 rnd_state):

        self.debris_radius = cfg.debris_radius
        self.debris_angle = cfg.debris_angle
        self.debris_extension = cfg.debris_extension
        self.building_spacing = cfg.building_spacing
        self.no_source_items = cfg.source_items
        self.flight_time_mu = cfg.flighttime_mu  # mean of log x
        self.flight_time_std = cfg.flighttime_std  # std of log x
        # self.wind_incr = (wind_max - wind_min) / float(wind_steps)
        self.source_items = cfg.source_items
        self.sources = list()

        # added
        # self.wind_dir_index = None
        # self.front_facing_walls = None
        # self.footprint_rect = None
        # self.result_breached = None
        # self.result_dmgperc = None
        # self.result_nv = None
        # self.result_breached = None
        # self.result_num_items = None
        # self.result_scores = None
        # self.result_dmgperc = None
        # self.result_items = None

        # FIXME !! NO VALUE is assigned to restrict_yord
        # generate grid
        self.create_sources(cfg.debris_radius,
                               cfg.debris_angle,
                               cfg.building_spacing,
                               cfg.flags['debris_staggered_sources'])

    def create_sources(self, radius, angle, spacing, flag_staggered,
                 restrict_yord=False):

        """

        Args:
            radius:
            angle:
            spacing:
            flag_staggered:
            restrict_yord:

        Returns:

        """
        xord = spacing
        yord = 0
        yordlim = radius / 6.0

        if flag_staggered:
            while xord <= radius:
                yordmax = xord * math.tan(math.radians(angle) / 2.0)
                if restrict_yord:
                    yordmax = min(yordlim, yordmax)

                if xord / spacing % 2:
                    self.sources.append(Point(xord, yord))
                    yord += spacing
                    while yord <= yordmax:
                        self.sources.append(Point(xord, yord))
                        self.sources.append(Point(xord, -yord))
                        yord += spacing
                else:
                    yord += spacing / 2
                    while yord <= yordmax:
                        self.sources.append(Point(xord, yord))
                        self.sources.append(Point(xord, -yord))
                        yord += spacing

                yord = 0
                xord += spacing

        else:
            while xord <= radius:
                self.sources.append(Point(xord, yord))
                yordmax = xord * math.tan(math.radians(angle) / 2.0)

                if restrict_yord:
                    yordmax = min(yordmax, yordlim)

                while yord <= yordmax - spacing:
                    yord += spacing
                    self.sources.append(Point(xord, yord))
                    self.sources.append(Point(xord, -yord))
                yord = 0
                xord += spacing

    def set_footprint_polygon(self, wind_dir_index):
        # self.wind_dir_index = wind_dir_index
        #
        # # determine front facing walls
        # self.front_facing_walls = []
        # for fw in facing[wind_dir_index + 1]:
        #     wall = self.house.getWallByDirection(fw)
        #     self.front_facing_walls.append(wall)
        #
        # # calculate footprint rect from wind_direction and house dimension
        # rect = None
        # if wind_dir_index in [0, 4]:
        #     rect = Rect(self.house.width, self.house.length, 0,
        #                 self.debris_extension)
        # elif wind_dir_index in [2, 6]:
        #     rect = Rect(self.house.length, self.house.width, 0,
        #                 self.debris_extension)
        # elif wind_dir_index in [1, 5]:
        #     rect = Rect(self.house.length, self.house.width, 45.0,
        #                 self.debris_extension)
        # elif wind_dir_index in [3, 7]:
        #     rect = Rect(self.house.length, self.house.width, -45.0,
        #                 self.debris_extension)
        # self.footprint_rect = rect

    def run(self, wind_speed, verbose=False):
        """ Returns several results as data members
        """
        self.result_nv = 0
        self.result_num_items = 0
        self.result_dmgperc = 0
        self.result_breached = False
        self.result_items = []

        # A = [self.region.beta, self.region.alpha]
        # estimate difference of loss ratio
        # read from pdf of vulnerability

        # determine how many items each source will have
        mean_delta = None
        item_mean = int(mean_delta * self.source_items)
        # item_mean can be computed using the pdf rather than the difference.

        if item_mean:

            # sample a poisson for each source
            num_itemsarr = np.random.poisson(item_mean, size=len(self.sources))

            # determine how many item buckets we need
            self.result_num_items = num_itemsarr.sum()
            self.result_scores = np.zeros(self.result_num_items)

            # loop through sources
            item_index = 0
            for src_index, num_source_items in enumerate(num_itemsarr):
                if num_source_items > 0:
                    self.result_nv += self.run_source(self.sources[src_index],
                                                      wind_speed,
                                                      num_source_items,
                                                      item_index,
                                                      self.result_items,
                                                      self.result_scores,
                                                      verbose)
                    item_index += num_source_items

            # process results if we have any items falling within our footprint.
            if self.result_nv > 0:
                self.check_impacts()

        self.gather_results()

    def run_source(self, source, wind_speed, num_items, item_index, items,
                   scores, verbose):
        """ Returns number of impacts from this source. """
        nv = 0
        for i in xrange(item_index, num_items + item_index):

            type_ = self.dt_sheet
            d100 = np.random.random_integers(1, 100)
            if d100 <= self.region.cr:
                type_ = self.dt_compact
            elif d100 <= self.region.rr:
                type_ = self.dt_rod

            momentum, X, Y = engine.debris_generate_item(wind_speed,
                                                         source.xord,
                                                         source.yord,
                                                         self.flighttime_mean,
                                                         self.flighttime_stddev,
                                                         type_.typeid,
                                                         type_.cdav,
                                                         type_.mass_mean,
                                                         type_.mass_stddev,
                                                         type_.fa_mean,
                                                         type_.fa_stddev)

            # we only need to store items when we are running in verbose mode
            #  (for plotting)
            if verbose:
                item = DebrisItem(source.col, type_.plot_shape)
                item.impact_point = Point(X, Y)
                items.append(item)

            if abs(X) < self.building_spacing:
                if self.footprint_rect.contains(Point(X, Y)):
                    nv += 1
            scores[i] = momentum
        return nv

    def gather_results(self):
        """ Calculate total area of envelope damaged, as a percentage
        """
        area = 0
        wall_area = self.house.getWallArea()
        for wall in self.front_facing_walls:
            for cov in wall.coverages:
                if not cov.result_intact:
                    if cov.description == 'window':
                        area += cov.area
                    else:
                        thisarea = cov.result_num_impacts if \
                            cov.result_num_impacts <= cov.area else cov.area
                        area += thisarea

        self.result_dmgperc = area / wall_area
        return self.result_dmgperc

    def check_impacts(self):
        """ Check to see if any of the Nv impacts break coverages.
        """
        Nv = self.result_nv
        A = 0.0
        for wall in self.front_facing_walls:
            A += wall.area
        self.result_scores.sort()

        # Loop through coverages rolling dice until all impacts have been
        # 'drained'
        for wall in self.front_facing_walls:
            for cov in wall.coverages:
                q = cov.area

                Ed = engine.lognormal(cov.type.failure_momentum_mean,
                                       cov.type.failure_momentum_stddev)

                # Ed = lognorm_rv_given_mean_stddev(
                #     cov.type.failure_momentum_mean,
                #     cov.type.failure_momentum_stddev)

                #Cum_Ed = engine.percentileofscore(self.result_scores.tolist(),
                #                                  Ed)
                Cum_Ed = (self.result_scores <= Ed).sum()\
                          /float(len(self.result_scores))

                if cov.description == 'window':
                    Pd = 1.0 - math.exp(-Nv * (q / A) * (1.0 - Cum_Ed))
                    dice = np.random.random()
                    if dice <= Pd:
                        cov.result_intact = False
                        self.result_breached = True
                else:
                    mean_num_impacts = Nv * (q / A) * (1 - Cum_Ed)
                    if mean_num_impacts > 0:
                        sampled_impacts = engine.poisson(mean_num_impacts)
                        # sampled_impacts = poisson(mean_num_impacts, random_state=24)
                        cov.result_num_impacts += sampled_impacts
                        if cov.result_num_impacts > 0:
                            cov.result_intact = False

    def render(self, v):
        """ Render a simple (but useful) plot showing a debris run.
        """
        points = []
        if self.result_items is not None:
            for item in self.result_items:
                sz = 30
                item.impact_point.set_plot(item.col, item.shape, sz, 0.5)
                points.append(item.impact_point)
        srcs = []
        for src in self.sources:
            srcs.append(Point(src.xord, src.yord, src.col, 'o', 300, 0.3))

        title = ('Debris Sample Field for Wind Speed: {0:.3f} m/s '
                 'in Region: {1}').format(v, self.region.name)
        self.footprint_rect.render(title, points, srcs)

    def get_breached(self):
        """ Returns True if any window was broken.
        """
        return self.result_breached


# unit tests
if __name__ == '__main__':
    import unittest
    # import house

    class MyTestCase(unittest.TestCase):
        # def test_debris_types(self):
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

        # def test_grid(self):
        #     self.assertEquals(len(Debris.create_sources(100.0, 45.0, 20.0, False)), 13)
        #
        # def test_staggered(self):
        #     self.assertEquals(len(Debris.sources(100.0, 45.0, 20.0, True)), 15)
        #
        # def test_plot(self):
        #     from matplotlib.pyplot import show, scatter
        #     sources = Debris.generate_debris_source(100.0, 45.0, 20.0, True)
        #     a = np.array(sources)
        #     scatter(a[:, 0], a[:, 1], s=50)
        #     show()


    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
