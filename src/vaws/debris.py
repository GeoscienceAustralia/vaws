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

import logging

from math import pow, radians, tan, sqrt, exp
from shapely.geometry import Point, Polygon
from shapely.affinity import rotate
# from shapely.ops import cascaded_union

from curve import vulnerability_weibull


class Debris(object):

    def __init__(self, cfg):

        self.cfg = cfg

        self.front_facing_walls = None
        self.area_walls = None
        self.coverages = None

        self.debris_items = list()
        self.breached = None  # only due to window breakage
        self.damaged_area = None
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
            _tuple: (polygon_inst, wind_dir_index)

        Returns:

        """
        polygon_inst, wind_dir_index = _tuple
        assert isinstance(polygon_inst, Polygon)
        assert isinstance(wind_dir_index, int)

        angle = {0: 90.0, 4: 90.0,  # S, N
                 1: 45.0, 5: 45.0,  # SW, NE
                 2: 0.0, 6: 0.0,  # E, W
                 3: -45.0, 7: -45.0}  # SE, NW

        rotated = rotate(polygon_inst, angle[wind_dir_index])

        '''
        option 1: stretch
        # calculate footprint rect from wind_direction and house dimension
        stretch = self.cfg.debris_extension

        points = [(_x - stretch, _y) for _x, _y in rotated.exterior.coords]
        self._footprint = cascaded_union([rotated,
                                          Polygon(points)]).convex_hull
        '''

        # option 2: rotate only
        self._footprint = rotated

        self.front_facing_walls = self.cfg.dic_front_facing_walls[
            self.cfg.wind_dir[wind_dir_index]]

        self.area_walls = 0.0
        for wall_name in self.front_facing_walls:
            self.area_walls += self.cfg.dic_walls[wall_name]

        id_ = self.cfg.df_coverages.apply(
            lambda row: row['wall_name'] in self.front_facing_walls, axis=1)

        self.coverages = self.cfg.df_coverages.loc[id_].copy()

        self.coverages['cum_prop_area'] = \
            self.coverages['area'].cumsum() / self.area_walls

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

        no_item_mean = self.cal_number_of_debris_items(wind_speed)
        self.damaged_area = 0.0

        # sample a poisson for each source
        no_item_by_source = rnd_state.poisson(no_item_mean,
                                              size=len(self.cfg.debris_sources))

        logging.debug('no_item_by_source: {}'.format(no_item_by_source))

        # loop through sources
        for no_item, source in zip(no_item_by_source, self.cfg.debris_sources):

            list_debris = rnd_state.choice(self.cfg.debris_types.keys(),
                                           size=no_item, replace=True)

            logging.debug('source: {}, list_debris: {}'.format(source, list_debris))

            for _debris_type in list_debris:
                self.generate_debris_item(wind_speed, source, _debris_type,
                                          rnd_state)

    def cal_number_of_debris_items(self, wind_speed):
        """
        dV = f * dD
        where dV: incr. number of debris items,
              dD: incr in wind damage
              f : a constant factor

        """
        # tropical town
        incr_damage = vulnerability_weibull(alpha_=0.10304,
                                            beta_=4.18252,
                                            x=wind_speed,
                                            flag='pdf')
        logging.debug('incr_damage: {}'.format(incr_damage))
        no_item_mean = int(incr_damage * self.cfg.source_items)

        return no_item_mean

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

            mass = rnd_state.lognormal(debris['mass_mu'], debris['mass_std'])

            frontal_area = rnd_state.lognormal(debris['frontalarea_mu'],
                                               debris['frontalarea_std'])

            flight_time = rnd_state.lognormal(self.cfg.flight_time_log_mu,
                                              self.cfg.flight_time_log_std)

            flight_distance = self.cal_flight_distance(debris_type_str,
                                                       flight_time,
                                                       frontal_area,
                                                       mass,
                                                       wind_speed)

            logging.debug('debris type:{}, area:{}, time:{}, distance:{}'.format(
                debris_type_str, frontal_area, flight_time, flight_distance))

            # determine landing location for a debris item
            # sigma_x, y are taken from Wehner et al. (2010)
            sigma_x = flight_distance / 3.0
            sigma_y = flight_distance / 12.0
            cov_matrix = [[pow(sigma_x, 2.0), 0.0], [0.0, pow(sigma_y, 2.0)]]

            x, y = rnd_state.multivariate_normal(mean=[0.0, 0.0],
                                                 cov=cov_matrix)
            # reference point: target house
            pt_debris = Point(x + source.x - flight_distance, y + source.y)

            self.debris_items.append((pt_debris, debris_type_str, source))

            if (self.footprint.contains(pt_debris) or
                    self.footprint.touches(pt_debris)):

                logging.debug('x:{}, y:{}: inside the footprint'.format(
                    pt_debris.x, pt_debris.y))

                item_momentum = self.cal_debris_mementum(debris['cdav'],
                                                         frontal_area,
                                                         flight_distance,
                                                         mass,
                                                         wind_speed,
                                                         rnd_state)

                # determine coverage type
                rv_ = rnd_state.uniform()
                _id = self.coverages[self.coverages['cum_prop_area'] > rv_].index[0]
                _coverage = self.coverages.loc[_id]

                # check whether it impacts or not using failure momentum
                _capacity = rnd_state.lognormal(
                    *_coverage['log_failure_momentum'])

                logging.debug('rv:{}, coverage type:{}, capacity:{}, demand:{}'.format(
                    rv_, _coverage['description'], _capacity, item_momentum))

                if _capacity < item_momentum:

                    if _coverage['description'] == 'window':
                        self.breached = True
                        self.damaged_area += _coverage['area']

                    else:
                        self.damaged_area += min(frontal_area,
                                                 _coverage['area'])

            else:
                logging.debug('pt outside')

    @staticmethod
    def cal_flight_distance(debris_type_str, flight_time, frontal_area, mass,
                            wind_speed, flag_poly=2):
        """
        calculate flight distance based on the methodology in Appendix of
        Lin and Vanmarcke (2008)

        Note that the coefficients of fifth order polynomials are from
        Lin and Vanmarcke (2008), while ones of quadratic form are proposed.

        Args:
            debris_type_str:
            flight_time:
            frontal_area:
            mass:
            wind_speed:

        Returns:

        """
        try:
            assert wind_speed > 0

        except AssertionError:
            return 0.0

        else:

            assert flag_poly == 2 or flag_poly == 5

            rho_air = 1.2  # air density
            g_const = 9.81  # acceleration of gravity

            # in increasing order
            coeff_by_type = {2: {'Compact': (0.011, 0.2060),
                                 'Sheet': (0.3456, 0.072),
                                 'Rod': (0.2376, 0.0723)},
                             5: {'Sheet': (0.456, -0.148, 0.024, -0.0014),
                                 'Compact': (0.405, -0.036, -0.052, 0.008),
                                 'Rod': (0.4005, -0.16, 0.036, -0.0032)}}

            # dimensionless time
            t_star = g_const * flight_time / wind_speed

            # Tachikawa Number: rho*(V**2)/(2*g*h_m*rho_m)
            # assume h_m * rho_m == mass / frontal_area
            k_star = rho_air * pow(wind_speed, 2.0) / (
                2.0 * g_const * mass / frontal_area)

            # dimensionless hor. displacement
            # k*x_star = k*g*x/V**2
            # x = (k*x_star)*(V**2)/(k*g)
            convert_to_dim = pow(wind_speed, 2.0) / (k_star * g_const)

            kt_star = k_star * t_star

            less_dis = None
            if flag_poly == 2:
                less_dis = sum([coeff_by_type[2][debris_type_str][i] *
                                pow(kt_star, i + 1) for i in range(2)])
            elif flag_poly == 5:
                less_dis = sum([coeff_by_type[5][debris_type_str][i] *
                                pow(kt_star, i + 2) for i in range(4)])

            return convert_to_dim * less_dis

    @staticmethod
    def cal_debris_mementum(cdav, frontal_area, flight_distance, mass,
                            wind_speed, rnd_state):
        """
        The ratio of horizontal velocity of the windborne debris object
         to the wind gust velocity is related to the horizontal distance
         travelled, x as below

         um/vs approx.= 1-exp(-b*sqrt(x))

         where um: horizontal velocity of the debris object
               vs: local gust wind speed
               x: horizontal distance travelled
               b: a parameter, sqrt(rho*CD*A/m)

         The ratio is assumed to follow a beta distribution with mean, and

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
        rho_air = 1.2  # air density
        param_b = sqrt(rho_air * cdav * frontal_area / mass)
        _mean = 1.0 - exp(-param_b * sqrt(flight_distance))

        # dispersion here means a + b of Beta(a, b)
        try:
            assert 0 <= _mean <= 1
        except AssertionError:
            print '{}:{}:{}'.format(_mean, param_b, flight_distance)

        try:
            dispersion = max(1.0 / _mean, 1.0 / (1.0 - _mean)) + 3.0
        except ZeroDivisionError:
            dispersion = 4.0
            _mean -= 0.001

        # mu = a / (a+b), var = (a*b)/[(a+b)**2*(a+b+1)]
        beta_a = _mean * dispersion
        beta_b = dispersion * (1.0 - _mean)

        # momentum of object: mass*vs = mass*vs*(um/vs)
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
                y_cord_max = x_cord * tan(radians(angle) / 2.0)

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
                y_cord_max = x_cord * tan(radians(angle) / 2.0)

                if restrict_y_cord:
                    y_cord_max = min(y_cord_max, y_cord_lim)

                while y_cord <= y_cord_max - bldg_spacing:
                    y_cord += bldg_spacing
                    sources.append(Point(x_cord, y_cord))
                    sources.append(Point(x_cord, -y_cord))
                y_cord = 0
                x_cord += bldg_spacing

        return sources

    '''
    def check_impacts(self, rnd_state):
        """ Check to see if any of the Nv impacts break coverages.
        """

        no_impacts = len(self.momentums)

        for wall_name in self.front_facing_walls:

            _id = self.cfg.df_coverages['wall_name'] == wall_name

            for _, coverage in self.cfg.df_coverages.loc[_id].iterrows():

                rv_ = rnd_state.lognormal(*(
                    coverage['lognormal_failure_momentum']+(no_impacts,)))

                ccdf_momentum = (self.momentums > rv_).sum() / float(no_impacts)

                mean_rate = no_impacts * coverage['area'] / self.area_walls * \
                            ccdf_momentum

                # Once window is damaged, then total area is redeemed as
                # damaged area while non-wondow is only

                if coverage.description == 'window':
                    prob_damage = 1.0 - exp(-mean_rate)

                    dice = np.random.random()
                    if dice <= Pd:
                        self.result_breached = True

                else:
                    sampled_impacts = rnd_state.poisson(mean_rate)

                    cov.result_num_impacts += sampled_impacts
                        if cov.result_num_impacts > 0:
                            cov.result_intact = False

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
    '''
