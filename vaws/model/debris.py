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
import pandas as pd

from numpy import rint, random, array
from numbers import Number
from math import pow, radians, tan, sqrt, exp
from shapely.geometry import Point, Polygon, LineString
from shapely.affinity import rotate


class Debris(object):

    rho_air = 1.2  # air density
    g_const = 9.81  # acceleration of gravity

    flight_distance_coeff = {2: {'Compact': [0.011, 0.2060],
                                 'Sheet': [0.3456, 0.072],
                                 'Rod': [0.2376, 0.0723]},
                             5: {'Sheet': [0.456, -0.148, 0.024, -0.0014],
                                 'Compact': [0.405, -0.036, -0.052, 0.008],
                                 'Rod': [0.4005, -0.16, 0.036, -0.0032]}}

    flight_distance_power = {2: [1, 2],
                             5: [2, 3, 4, 5]}

    angle_by_idx = {0: 90.0, 4: 90.0,  # S, N
                    1: 45.0, 5: 45.0,  # SW, NE
                    2: 0.0, 6: 0.0,  # E, W
                    3: -45.0, 7: -45.0}  # SE, NW

    def __init__(self, cfg):

        self.cfg = cfg

        # assigned by footprint.setter
        self._footprint = None
        self.front_facing_walls = None

        # assigned by coverages.setter
        self._coverages = None
        self.area = None
        self.coverage_idx = None
        self.coverage_prob = None

        # assigned by rnd_state.setter
        self._rnd_state = None

        # assigned by no_items_mean.setter
        self._no_items_mean = None  # mean value for poisson dist

        self.debris_items = []  # container of items over wind steps

        # vary over wind speeds
        self.no_items = 0  # total number of debris items generated
        self.no_touched = 0
        # self.breached = 0  # only due to window breakage
        self.damaged_area = 0.0  # total damaged area by debris items

    @property
    def no_items_mean(self):
        return self._no_items_mean

    @no_items_mean.setter
    def no_items_mean(self, value):
        """

        dN = f * dD
        where dN: incr. number of debris items,
              dD: incr in vulnerability (or damage index)
              f : a constant factor

              if we use dD/dV (pdf of vulnerability), then
                 dN = f * (dD/dV) * dV

        Args:
            value:

        Returns:

        """
        assert isinstance(value, Number)

        self._no_items_mean = rint(self.cfg.source_items * value)

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

            self.footprint
            self.front_facing_walls

        """
        polygon_inst, wind_dir_idx = _tuple
        assert isinstance(polygon_inst, Polygon)
        assert isinstance(wind_dir_idx, int)

        self._footprint = rotate(polygon_inst, self.__class__.angle_by_idx[
            wind_dir_idx])

        self.front_facing_walls = self.cfg.front_facing_walls[
            self.cfg.wind_dir[wind_dir_idx]]

    @property
    def coverages(self):
        return self._coverages

    @coverages.setter
    def coverages(self, _df):

        assert isinstance(_df, pd.DataFrame)

        _tf = _df['wall_name'].isin(self.front_facing_walls)
        self._coverages = _df.loc[_tf, 'coverage'].to_dict()

        self.area = 0.0
        self.coverage_idx = []
        self.coverage_prob = []
        for key, value in self._coverages.iteritems():
            self.area += value.area
            self.coverage_idx.append(key)
            self.coverage_prob.append(self.area)

        self.coverage_prob = array(self.coverage_prob)/self.area

    @property
    def rnd_state(self):
        return self._rnd_state

    @rnd_state.setter
    def rnd_state(self, value):
        """

        Args:
            value:

        Returns:

        """

        assert isinstance(value, random.RandomState)
        self._rnd_state = value

    def run(self, wind_speed):
        """

        Args:
            wind_speed:

        Returns:

        """

        self.no_touched = 0

        # sample a poisson for each source
        no_items_by_source = self.rnd_state.poisson(
            self.no_items_mean, size=len(self.cfg.debris_sources))

        self.no_items = no_items_by_source.sum()
        logging.debug('no_item_by_source at speed {:.3f}: {}, {}'.format(
            wind_speed, self.no_items, self.no_items_mean))

        # loop through sources
        for no_item, source in zip(no_items_by_source, self.cfg.debris_sources):

            list_debris = self.rnd_state.choice(self.cfg.debris_types_keys,
                                                size=no_item, replace=True)

            # logging.debug('source: {}, list_debris: {}'.format(source, list_debris))

            for _debris_type in list_debris:
                self.generate_debris_item(wind_speed, source, _debris_type)

    def generate_debris_item(self, wind_speed, source, debris_type_str):
        """

        Args:
            wind_speed:
            source:
            debris_type_str:

        Returns:

        """

        # try:
        debris = self.cfg.debris_types[debris_type_str]

        # except KeyError:
        #     logging.warning('invalid debris type: {}'.format(debris_type_str))
        #
        # else:

        mass = self.rnd_state.lognormal(debris['mass_mu'],
                                        debris['mass_std'])

        frontal_area = self.rnd_state.lognormal(debris['frontal_area_mu'],
                                                debris['frontal_area_std'])

        flight_time = self.rnd_state.lognormal(self.cfg.flight_time_log_mu,
                                               self.cfg.flight_time_log_std)

        flight_distance = self.compute_flight_distance(debris_type_str,
                                                       flight_time,
                                                       frontal_area,
                                                       mass,
                                                       wind_speed)

        # logging.debug('debris type:{}, area:{}, time:{}, distance:{}'.format(
        #    debris_type_str, frontal_area, flight_time, flight_distance))

        # determine landing location for a debris item
        # sigma_x, y are taken from Wehner et al. (2010)
        sigma_x = flight_distance / 3.0
        sigma_y = flight_distance / 12.0
        cov_matrix = [[pow(sigma_x, 2.0), 0.0], [0.0, pow(sigma_y, 2.0)]]

        x, y = self.rnd_state.multivariate_normal(mean=[0.0, 0.0],
                                                  cov=cov_matrix)
        # reference point: target house
        pt_debris = Point(x + source.x - flight_distance, y + source.y)
        line_debris = LineString([source, pt_debris])

        self.debris_items.append(line_debris)

        if line_debris.intersects(self.footprint):

            self.no_touched += 1

            item_momentum = self.compute_debris_momentum(debris['cdav'],
                                                         frontal_area,
                                                         flight_distance,
                                                         mass,
                                                         wind_speed,
                                                         self.rnd_state)

            self.check_debris_impact(frontal_area, item_momentum)

            logging.debug('debris impact from {}, {}'.format(
                pt_debris.x, pt_debris.y))

    def check_debris_impact(self, frontal_area, item_momentum):
        """

        Args:
            frontal_area:
            item_momentum:

        Returns:
            self.breached

        """

        # determine coverage type
        _rv = self.rnd_state.uniform()
        _id = self.coverage_idx[(self.coverage_prob < _rv).sum()]
        _coverage = self.coverages[_id]

        #logging.debug('coverage id: {} due to rv: {} vs prob: {}'.format(
        #    _id, _rv, self.coverage_prob))

        # check impact using failure momentum
        try:
            _capacity = self.rnd_state.lognormal(*_coverage.log_failure_momentum)
        except ValueError:
            _capacity = exp(_coverage.log_failure_momentum[0])

        if _capacity < item_momentum:
            # history of coverage is ignored
            if _coverage.description == 'window':
                _coverage.breached_area = _coverage.area
                _coverage.breached = 1
            else:
                _coverage.breached_area = min(frontal_area, _coverage.area)

            logging.debug('coverage {} breached by debris b/c {:.3f} < {:.3f} -> area: {:.3f}'.format(
                _coverage.name, _capacity, item_momentum, _coverage.breached_area))

    def compute_flight_distance(self, debris_type_str, flight_time,
                                frontal_area, mass, wind_speed, flag_poly=2):
        """
        calculate flight distance based on the methodology in Appendix of
        Lin and Vanmarcke (2008)

        Args:
            debris_type_str:
            flight_time:
            frontal_area:
            mass:
            wind_speed:
            flag_poly:

        Returns:

        Notes:
            The coefficients of fifth order polynomials are from
        Lin and Vanmarcke (2008), while quadratic form are proposed by Martin.

        """
        try:
            assert wind_speed > 0
        except AssertionError:
            return 0.0

        else:

            assert flag_poly in self.__class__.flight_distance_power

            # dimensionless time
            t_star = self.__class__.g_const * flight_time / wind_speed

            # Tachikawa Number: rho*(V**2)/(2*g*h_m*rho_m)
            # assume h_m * rho_m == mass / frontal_area
            k_star = self.__class__.rho_air * pow(wind_speed, 2.0) / (
                2.0 * self.__class__.g_const * mass / frontal_area)
            kt_star = k_star * t_star

            kt_star_powered = array([pow(kt_star, i)
                for i in self.__class__.flight_distance_power[flag_poly]])
            coeff = array(self.__class__.flight_distance_coeff[
                              flag_poly][debris_type_str])
            less_dis = (coeff * kt_star_powered).sum()

            # dimensionless hor. displacement
            # k*x_star = k*g*x/V**2
            # x = (k*x_star)*(V**2)/(k*g)
            convert_to_dim = pow(wind_speed, 2.0) / (
                k_star * self.__class__.g_const)

            return convert_to_dim * less_dis

    def compute_debris_momentum(self, cdav, frontal_area, flight_distance, mass,
                                wind_speed, rnd_state):
        """
        calculate momentum of debris object

        Args:
            cdav: average drag coefficient
            frontal_area:
            flight_distance:
            mass:
            wind_speed:
            rnd_state:

        Returns: momentum of debris object

        Notes:
         The ratio of horizontal velocity of the windborne debris object
         to the wind gust velocity is related to the horizontal distance
         travelled, x as below

         um/vs approx.= 1-exp(-b*sqrt(x))

         where um: horizontal velocity of the debris object
               vs: local gust wind speed
               x: horizontal distance travelled
               b: a parameter, sqrt(rho*CD*A/m)

         The ratio is assumed to follow a beta distribution with mean, and

        """
        # calculate um/vs, ratio of hor. vel. of debris to local wind speed
        param_b = sqrt(self.__class__.rho_air * cdav * frontal_area / mass)
        _mean = 1.0 - exp(-param_b * sqrt(flight_distance))

        # dispersion here means a + b of Beta(a, b)
        try:
            assert 0.0 <= _mean <= 1.0
        except AssertionError:
            logging.warning('invalid mean of beta dist.: {} with b: {},'
                            'flight_distance: {}'.format(_mean, param_b,
                                                         flight_distance))

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

        sources = []
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

