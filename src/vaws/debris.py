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

class Debris(object):

    param1_by_type = {'Compact': 0.2060, 'Sheet': 0.072, 'Rod': 0.0723}
    param2_by_type = {'Compact': 0.011, 'Sheet': 0.3456, 'Rod': 0.2376}

    def __init__(self, cfg):

        self.cfg = cfg
        self.no_impacts = None

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
                 1: 45.0, 5: 45.0,    # SW, NE
                 2: 0.0, 6: 0.0,      # E, W
                 3: -45.0, 7: -45.0}  # SE, NW

        rotated = rotate(polygon_inst, angle[wind_dir_index])
        points = [(_x - stretch, _y) for _x, _y in rotated.exterior.coords]
        self._footprint = cascaded_union([rotated,
                                          Polygon(points)]).convex_hull

    # @staticmethod
    # def get_angle(x_coord, y_coord):
    #     """
    #     compute anlge (in degrees) between a vector and unit vector (-1, 0)
    #     Args:
    #         x_coord:
    #         y_coord:
    #
    #     Returns:
    #     """
    #
    #     v0 = np.array([x_coord, y_coord])
    #     v1 = np.array([-1.0, 0.0])
    #     angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    #     return np.degrees(angle)

    def run(self, wind_speed, rnd_state):
        """ Returns several results as data members
        """

        self.no_impacts = 0  # average number of impacts on the bldg footprint

        no_item_mean = self.cal_number_of_debris_items(wind_speed)

        # sample a poisson for each source
        no_item_by_source = rnd_state.poisson(no_item_mean,
                                              size=len(self.cfg.debris_sources))

        # loop through sources
        for no_item, source in zip(no_item_by_source, self.cfg.debris_sources):

            list_debris = rnd_state.choice(self.cfg.debris_types.keys(),
                                           size=no_item, replace=True)

            for _debris_type in list_debris:
                self.generate_debris_item(wind_speed, source, _debris_type,
                                          rnd_state)

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
        area = 0.0
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
            debris_type_str:
            rnd_state:

        Returns:

        """

        try:
            debris = self.cfg.debris_types[debris_type_str]
        except KeyError:
            print '{} is not found in the debris types'.format(debris_type_str)
        else:
            param1 = self.param1_by_type[debris_type_str]
            param2 = self.param2_by_type[debris_type_str]

            mass = rnd_state.lognormal(debris['mass_mu'], debris['mass_std'])

            frontal_area = rnd_state.lognormal(debris['frontalarea_mu'],
                                               debris['frontalarea_std'])

            flight_time = rnd_state.lognormal(self.cfg.flight_time_mu,
                                              self.cfg.flight_time_std)

            c_t = 9.81 * flight_time / wind_speed
            c_k = 1.2 * wind_speed * wind_speed/(2 * 9.81 * mass / frontal_area)
            c_kt = c_k * c_t

            flight_distance = math.pow(wind_speed, 2) / 9.81 / c_k * (
                param1 * math.pow(c_kt, 2) + param2 * c_kt)

            item_momentum = self.debris_trajectory(debris['cdav'],
                                                   frontal_area,
                                                   flight_distance,
                                                   mass,
                                                   wind_speed,
                                                   rnd_state)

            # Sample Impact Location
            sigma_x = flight_distance / 3.0
            sigma_y = flight_distance / 12.0
            rho_xy = 1.0

            cov_matrix = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
                          [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]

            # try:
            x, y = rnd_state.multivariate_normal(mean=[0.0, 0.0],
                                                 cov=cov_matrix)
            # except RuntimeWarning:
            # print('cov_matrix: {}'.format(cov_matrix))

            pt_debris = Point(source.x - flight_distance + x, source.y + y)

            if self.footprint.contains(pt_debris) or \
                    self.footprint.touches(pt_debris):
                self.no_impacts += 1

    @staticmethod
    def debris_trajectory(cdav, frontal_area, flight_distance, mass,
                          wind_speed, rnd_state):
        """

        Args:
            cdav: average drag coefficient
            frontal_area:
            flight_distance:
            mass:
            wind_speed:
            rnd_state:

        Returns:

        """
        # calculate um/vs, ratio of hor. vel. of debris to local wind speed
        rho_a = 1.2  # air density
        param_b = math.sqrt(rho_a * cdav * frontal_area / mass)
        _mean = 1 - math.exp(-param_b * math.sqrt(flight_distance))

        try:
            dispersion = max(1.0 / _mean, 1.0 / (1.0 - _mean)) + 3.0
        except ZeroDivisionError:
            dispersion = 4.0
            _mean -= 0.001

        beta_a = _mean * dispersion
        beta_b = dispersion * (1.0 - _mean)
        return mass * wind_speed * rnd_state.beta(beta_a, beta_b)

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

