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

from __future__ import division, print_function

import logging

import numpy as np
import math
from shapely import geometry

from vaws.model.constants import (RHO_AIR, G_CONST, FLIGHT_DISTANCE_POWER,
                                  FLIGHT_DISTANCE_COEFF, DEBRIS_TYPES_KEYS)
from vaws.model.stats import sample_lognormal


class Debris(object):

    def __init__(self, debris_source, debris_type, debris_property, wind_speed,
                 rnd_state, flag_poly=2):

        self.source = debris_source
        self.type = debris_type
        self.property = debris_property
        self.wind_speed = wind_speed
        self.rnd_state = rnd_state
        self.flag_poly = flag_poly

        self._mass = None
        self._frontal_area = None
        self._flight_time = None
        self._landing = None
        self._momentum = None
        self._flight_distance = None

        self.impact = None

    @property
    def cdav(self):
        return self.property['cdav']

    @property
    def mass(self):
        if self._mass is None:
            self._mass = sample_lognormal(mu_lnx=self.property['mass_mu'],
                                          std_lnx=self.property['mass_std'],
                                          rnd_state=self.rnd_state)
        return self._mass

    @property
    def frontal_area(self):
        if self._frontal_area is None:
            self._frontal_area = sample_lognormal(
                mu_lnx=self.property['frontal_area_mu'],
                std_lnx=self.property['frontal_area_std'],
                rnd_state=self.rnd_state)
        return self._frontal_area

    @property
    def flight_time(self):
        if self._flight_time is None:
            self._flight_time = sample_lognormal(
                mu_lnx=self.property['flight_time_mu'],
                std_lnx=self.property['flight_time_std'],
                rnd_state=self.rnd_state)
        return self._flight_time

    @property
    def landing(self):
        if self._landing is None:
            # determine landing location for a debris item
            # sigma_x, y are taken from Wehner et al. (2010)
            x = self.rnd_state.normal(loc=0,
                                      scale=self.flight_distance / 3.0)
            y = self.rnd_state.normal(loc=0,
                                      scale=self.flight_distance / 12.0)

            # cov_matrix = [[pow(sigma_x, 2.0), 0.0], [0.0, pow(sigma_y, 2.0)]]
            # x, y = self.rnd_state.multivariate_normal(mean=[0.0, 0.0],
            #                                           cov=cov_matrix)
            # reference point: target house
            self._landing = geometry.Point(
                x + self.source.x - self.flight_distance, y + self.source.y)
        return self._landing

    @property
    def trajectory(self):
        return geometry.LineString([self.source, self.landing])

    @property
    def momentum(self):
        """
        calculate momentum of debris object

        Args:
            debris_property:
            rnd_state: None, integer, or np.random.RandomState
            wind_speed:

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
        if self._momentum is None:
            # mu = a / (a+b), var = (a*b)/[(a+b)**2*(a+b+1)]
            beta_a, beta_b = self.compute_coeff_beta_dist()

            # momentum of object: mass*vs = mass*vs*(um/vs)
            self._momentum = self.mass * self.wind_speed * self.rnd_state.beta(beta_a, beta_b)
        return self._momentum

    @property
    def flight_distance(self):
        """
        calculate flight distance based on the methodology in Appendix of
        Lin and Vanmarcke (2008)

        Args:
            debris_type:
            debris_property:
            wind_speed:

        Returns:

        Notes:
            The coefficients of fifth order polynomials are from
        Lin and Vanmarcke (2008), while quadratic form are proposed by Martin.

        """
        if self._flight_distance is None:

            self._flight_distance = 0.0

            if self.wind_speed > 0:

                # dimensionless time
                t_star = G_CONST * self.flight_time / self.wind_speed

                # Tachikawa Number: rho*(V**2)/(2*g*h_m*rho_m)
                # assume h_m * rho_m == mass / frontal_area
                k_star = RHO_AIR * self.wind_speed ** 2 / (
                    2.0 * G_CONST * self.mass / self.frontal_area)
                kt_star = k_star * t_star

                kt_star_powered = [kt_star ** i for i in
                                   FLIGHT_DISTANCE_POWER[self.flag_poly]]
                coeff = FLIGHT_DISTANCE_COEFF[self.flag_poly][self.type]
                less_dis = sum([x*y for x, y in zip(coeff, kt_star_powered)])

                # dimensionless hor. displacement
                # k*x_star = k*g*x/V**2
                # x = (k*x_star)*(V**2)/(k*g)
                convert_to_dim = self.wind_speed ** 2 / (k_star * G_CONST)
                self._flight_distance = convert_to_dim * less_dis

        return self._flight_distance

    def check_impact(self, footprint, boundary):

        land_within_footprint = footprint.contains(self.landing)
        intersect_within_boundary = (self.trajectory.intersects(footprint)
                                     and boundary.contains(self.landing))
        if land_within_footprint or intersect_within_boundary:
            self.impact = 1
        else:
            self.impact = 0

    def check_coverages(self, coverages, prob_coverages):

        msg = '{coverage} breached by debris with {momentum:.3f} -> {area:.3f}'

        if self.impact:

            _coverage = self.rnd_state.choice(coverages, p=prob_coverages)

            # check impact using failure momentum
            if _coverage.momentum_capacity < self.momentum:
                # history of coverage is ignored
                if _coverage.description == 'window':
                    _coverage.breached_area = _coverage.area
                    _coverage.breached = 1
                else:
                    _coverage.breached_area = min(
                        # frontal_area * self.__class__.amplification_factor,
                        1.0, _coverage.area)

                logging.debug(msg.format(coverage=_coverage.name,
                                         momentum=self.momentum,
                                         area=_coverage.area))

    def compute_coeff_beta_dist(self):
        """
        calculate momentum of debris object

        Args:
            debris_property:
                cdav: average drag coefficient
                frontal_area:
                flight_distance:
                mass:

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
        param_b = math.sqrt(RHO_AIR * self.cdav * self.frontal_area / self.mass)
        _mean = 1.0 - math.exp(-param_b * math.sqrt(self.flight_distance))

        # dispersion here means a + b of Beta(a, b)
        try:
            assert 0.0 <= _mean <= 1.0
        except AssertionError:
            logging.warning('invalid mean of beta dist.: {} with b: {},'
                            'flight_distance: {}'.format(_mean, param_b,
                                                         self.flight_distance))

        try:
            dispersion = max(1.0 / _mean, 1.0 / (1.0 - _mean)) + 3.0
        except ZeroDivisionError:
            dispersion = 4.0
            _mean -= 0.001

        # mu = a / (a+b), var = (a*b)/[(a+b)**2*(a+b+1)]
        # beta_a = _mean * dispersion
        # beta_b = dispersion * (1.0 - _mean)

        return _mean * dispersion, dispersion * (1.0 - _mean)


def generate_debris_items(cfg, mean_no_debris_items, wind_speed, rnd_state):

    debris_items = []

    for source in cfg.debris_sources:

        no_items = rnd_state.poisson(mean_no_debris_items)

        debris_types = rnd_state.choice(DEBRIS_TYPES_KEYS,
                                        size=no_items,
                                        replace=True,
                                        p=cfg.debris_types_ratio)

        for debris_type in debris_types:

            _debris = Debris(debris_source=source,
                             debris_type=debris_type,
                             debris_property=cfg.debris_types[debris_type],
                             wind_speed=wind_speed,
                             rnd_state=rnd_state)

            debris_items.append(_debris)

    return debris_items


def check_coverages(debris_items, coverages, rnd_state, coverage_area):
    # Complementary CDF of impact momentum

    debris_momentums = np.array([x.momentum for x in debris_items])
    no_debris_items = len(debris_items)
    no_debris_impacts = sum([x.impact for x in debris_items])

    for _coverage in coverages:

        # Complementary CDF of impact momentum
        ccdf = (debris_momentums > _coverage.momentum_capacity).sum()/no_debris_items
        poisson_rate = no_debris_impacts * _coverage.area / coverage_area * ccdf

        if _coverage.description == 'window':
            prob_damage = 1.0 - math.exp(-1.0*poisson_rate)
            rv = rnd_state.rand()
            if rv < prob_damage:
                _coverage.breached_area = _coverage.area
                _coverage.breached = 1
        else:
            # assume area: no_impacts * size(1) * amplification_factor(1)
            sampled_impacts = rnd_state.poisson(poisson_rate)
            _coverage.breached_area = min(sampled_impacts, _coverage.area)


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
            y_cord_max = x_cord * math.tan(math.radians(angle) / 2.0)

            if restrict_y_cord:
                y_cord_max = min(y_cord_lim, y_cord_max)

            if x_cord / bldg_spacing % 2:
                sources.append(geometry.Point(x_cord, y_cord))
                y_cord += bldg_spacing
                while y_cord <= y_cord_max:
                    sources.append(geometry.Point(x_cord, y_cord))
                    sources.append(geometry.Point(x_cord, -y_cord))
                    y_cord += bldg_spacing
            else:
                y_cord += bldg_spacing / 2.0
                while y_cord <= y_cord_max:
                    sources.append(geometry.Point(x_cord, y_cord))
                    sources.append(geometry.Point(x_cord, -y_cord))
                    y_cord += bldg_spacing

            y_cord = 0.0
            x_cord += bldg_spacing

    else:
        while x_cord <= radius:
            sources.append(geometry.Point(x_cord, y_cord))
            y_cord_max = x_cord * math.tan(math.radians(angle) / 2.0)

            if restrict_y_cord:
                y_cord_max = min(y_cord_max, y_cord_lim)

            while y_cord <= y_cord_max - bldg_spacing:
                y_cord += bldg_spacing
                sources.append(geometry.Point(x_cord, y_cord))
                sources.append(geometry.Point(x_cord, -y_cord))
            y_cord = 0
            x_cord += bldg_spacing

    return sources


def check_impact_by_debris(debris_item, footprint, boundary):
    """

    :param debris_item:
    :param footprint:
    :param boundary:
    :return:
    """

    debris_item.check_impact(footprint=footprint, boundary=boundary)

    return debris_item




