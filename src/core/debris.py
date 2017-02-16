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
import math
from shapely.geometry import Point, Polygon, LineString
from shapely.affinity import rotate
from shapely.ops import cascaded_union

# FIXME!!! mapping wind direciton to facing wall will provided as an input
# lookup table mapping wind direction (1-8) to list of front facing wall
# directions
# facing = {1: [1], 2: [1, 3], 3: [3], 4: [3, 5], 5: [5], 6: [5, 7], 7: [7],
#           8: [1, 7]}


class Debris(object):

    param1_by_type = {'Compact': 0.2060, 'Sheet': 0.072, 'Rod': 0.0723}
    param2_by_type = {'Compact': 0.011, 'Sheet': 0.3456, 'Rod': 0.2376}

    def __init__(self, cfg):

        self.cfg = cfg
        self.num_impacts = None

        self._footprint = None

    @property
    def footprint(self):
        return self._footprint

    @footprint.setter
    def footprint(self, _tuple):
        """
        create house footprint by wind direction
        Note that debris source model is generated assuming wind blows from E.

        Args:
            polygon_inst:
            wind_dir_index:

        Returns:

        """
        polygon_inst, wind_dir_index = _tuple
        assert isinstance(polygon_inst, Polygon)
        assert isinstance(wind_dir_index, int)

        # # calculate footprint rect from wind_direction and house dimension
        stretch = 20.0  # FIXME!! HARD-CODED

        angle = {0: 90.0, 4: 90.0,    # S, N
                 2: 0.0, 6: 0.0,      # E, W
                 1: 45.0, 5: 45.0,  # SW, NE
                 3: -45.0, 7: -45.0}    # SE, NW

        rotated = rotate(polygon_inst, angle[wind_dir_index])
        points = [(_x-stretch, _y) for _x, _y in rotated.exterior.coords]
        self._footprint = cascaded_union([rotated,
                                          Polygon(points)]).convex_hull

    @staticmethod
    def get_angle(x_coord, y_coord):
        """
        compute anlge (in degrees) between a vector and unit vector (-1, 0)
        Args:
            x_coord:
            y_coord:

        Returns:
        """

        v0 = np.array([x_coord, y_coord])
        v1 = np.array([-1.0, 0.0])
        angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
        return np.degrees(angle)

    def run(self, wind_speed, rnd_state):
        """ Returns several results as data members
        """

        self.num_impacts = 0  # average number of impacts on the bldg footprint

        no_item_mean = self.cal_number_of_debris_items(wind_speed)

        # sample a poisson for each source
        no_item_by_source = rnd_state.poisson(no_item_mean,
                                              size=len(self.cfg.debris_sources))

        # loop through sources
        for no_item, source in zip(no_item_by_source, self.cfg.debris_sources):

            list_debris = rnd_state.choice(self.cfg.debris_types.keys(),
                                           size=no_item, replace=True)

            for _debris_type in list_debris:
                self.generate_debris_item(wind_speed,
                                          source,
                                          _debris_type)

                #     if abs(X) < self.cfg.building_spacing:
                #         if self.footprint_rect.contains(Point(X, Y)):
                #             nv += 1
                #     scores[i] = momentum
                # return nv

        # process results if we have any items falling within our footprint.
        if self.result_nv > 0:
            self.check_impacts()

        self.gather_results()

    def cal_number_of_debris_items(self, wind_speed):
        """


        """
        mean_delta = None
        wind_speed += wind_speed
        no_item_mean = int(mean_delta * self.cfg.source_items)
        # item_mean can be computed using the pdf rather than the difference.
        return no_item_mean

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

                # Cum_Ed = engine.percentileofscore(self.result_scores.tolist(),
                #                                  Ed)
                Cum_Ed = (self.result_scores <= Ed).sum() \
                         / float(len(self.result_scores))

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

    def generate_debris_item(self, wind_speed, source, debris_type_str,
                             rnd_state):
        """

        Args:
            wind_speed:
            source:
            rnd_state:

        Returns:

        """

        try:
            debris = self.cfg.debris_types[debris_type_str]
        except KeyError:
            print '{} is not found in the defined debris types'.format(
                debris_type_str)
        else:
            param1 = self.param1_by_type[debris_type_str]
            param2 = self.param2_by_type[debris_type_str]

        mass = rnd_state.lognormal(debris['mass_mu'], debris['mass_std'])

        fa = rnd_state.lognormal(debris['frontalarea_mu'],
                                 debris['frontalarea_std'])

        flight_time = rnd_state.lognormal(self.cfg.flight_time_mu,
                                          self.cfg.flight_time_std)

        c_t = 9.81 * flight_time / wind_speed
        c_k = 1.2 * wind_speed * wind_speed / (2 * 9.81 * mass / fa)
        c_kt = c_k * c_t

        flight_distance = math.pow(wind_speed, 2) / 9.81 / c_k * (
            param1 * math.pow(c_kt, 2) + param2 * c_kt)

        item_momentum = self.debris_trajectory(debris['cdav'],
                                               fa,
                                               flight_distance,
                                               mass,
                                               rnd_state,
                                               wind_speed)





        # Sample Impact Location
        sigma_x = flight_distance / 3.0
        sigma_y = flight_distance / 12.0
        rho_xy = 1.0

        cov_matrix = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
                      [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]

        # try:
        x, y = rnd_state.multivariate_normal(mean=[0.0, 0.0], cov=cov_matrix)
        # except RuntimeWarning:
        # print('cov_matrix: {}'.format(cov_matrix))

        pt_debris = Point(source.x - flight_distance + x, source.y + y)

        if self.footprint.contains(pt_debris) or \
                self.footprint.touches(pt_debris):
            self.num_impacts += 1

    @staticmethod
    def debris_trajectory(cdav, fa, flight_distance, mass, rnd_state,
                          wind_speed):
        """

        Args:
            cdav: average drag coefficient
            fa: frontal area
            flight_distance:
            mass:
            rnd_state:
            wind_speed:

        Returns:

        """
        # calculate um/vs, ratio of hor. vel. of debris to local wind speed
        rho_a = 1.2  # air density
        param_b = math.sqrt(rho_a * cdav * fa / mass)

        _mean = 1 - math.exp(-param_b * math.sqrt(flight_distance))
        try:
            dispersion = max(1.0 / _mean, 1.0 / (1.0 - _mean)) + 3.0
        except ZeroDivisionError:
            dispersion = 4.0
            _mean -= 0.001

        beta_a = _mean * dispersion
        beta_b = dispersion * (1.0 - _mean)
        item_momentum = mass * wind_speed * rnd_state.beta(beta_a, beta_b)

        return item_momentum

    @staticmethod
    def create_sources(radius, angle, bldg_spacing, flag_staggered,
                       restrict_y_cord=False):
        """
        define a debris generation region for a building
        Args:
            radius:
            angle: (in degree)
            bldg_spacing:
            flag_staggered:
            restrict_y_cord:
           # FIXME !! NO VALUE is assigned to restrict_yord

        Returns:

        """

        x_cord = bldg_spacing
        y_cord = 0.0
        y_cord_lim = radius / 6.0

        sources = list()
        if flag_staggered:
            while x_cord <= radius:
                y_cord_max = x_cord * math.tan(math.radians(angle) / 2.0)

                if restrict_y_cord:
                    y_cord_max = min(y_cord_lim, y_cord_max)

                if x_cord / bldg_spacing % 2:
                    sources.append(Point(x_cord, y_cord))
                    y_cord += bldg_spacing
                    while y_cord <= y_cord_max:
                        sources.append(Point(x_cord, y_cord))
                        sources.append(Point(x_cord, -y_cord))
                        y_cord += bldg_spacing
                else:
                    y_cord += bldg_spacing / 2.0
                    while y_cord <= y_cord_max:
                        sources.append(Point(x_cord, y_cord))
                        sources.append(Point(x_cord, -y_cord))
                        y_cord += bldg_spacing

                y_cord = 0.0
                x_cord += bldg_spacing

        else:
            while x_cord <= radius:
                sources.append(Point(x_cord, y_cord))
                y_cord_max = x_cord * math.tan(math.radians(angle) / 2.0)

                if restrict_y_cord:
                    y_cord_max = min(y_cord_max, y_cord_lim)

                while y_cord <= y_cord_max - bldg_spacing:
                    y_cord += bldg_spacing
                    sources.append(Point(x_cord, y_cord))
                    sources.append(Point(x_cord, -y_cord))
                y_cord = 0
                x_cord += bldg_spacing

        return sources


# unit tests
if __name__ == '__main__':
    import unittest
    import os
    import matplotlib.pyplot as plt
    from descartes import PolygonPatch

    from scenario import Scenario

    class MyTestCase(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            path = '/'.join(__file__.split('/')[:-1])
            cfg_file = os.path.join(path, '../scenarios/test_roof_sheeting2.cfg')
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
            plt.show()

            # staggered source
            sources2 = Debris.create_sources(self.cfg.debris_radius,
                                             self.cfg.debris_angle,
                                             self.cfg.building_spacing,
                                             True)

            self.assertEquals(len(sources2), 15)

            plt.figure()
            for source in sources2:
                plt.scatter(source.x, source.y)
            plt.show()

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
            plt.show()

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
            plt.show()

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
            plt.show()

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
            plt.show()

        def test_footprint_non_rect(self):

            footprint_inst = Polygon([(-6.5, 4.0), (6.5, 4.0), (6.5, 0.0), (0.0, 0.0),
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
            plt.show()



    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
