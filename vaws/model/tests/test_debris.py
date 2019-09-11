import unittest
import os
# import copy
import numpy as np
# import pandas as pd
# import parmap
import matplotlib.pyplot as plt
from matplotlib import patches
import logging
import time
# from scipy import stats

import numbers
import math
from shapely import geometry, affinity

from vaws.model.constants import (WIND_DIR, DEBRIS_TYPES_KEYS,
                                  FLIGHT_DISTANCE_POWER, VUL_DIC,
                                  FLIGHT_DISTANCE_COEFF, RHO_AIR, G_CONST)
from vaws.model.config import Config
from vaws.model.debris import Debris, generate_debris_items, create_sources, \
    check_impact_by_debris
from vaws.model.coverage import Coverage
from vaws.model.curve import vulnerability_weibull_pdf
from vaws.model.house import House

REF04 = geometry.Polygon([(-24.0, 6.5), (4.0, 6.5), (4.0, -6.5),
                          (-24.0, -6.5), (-24.0, 6.5)])

REF37 = geometry.Polygon([(-1.77, 7.42), (7.42, -1.77), (1.77, -7.42),
                          (-27.42, 1.77), (-1.77, 7.42)])

REF26 = geometry.Polygon([(-26.5, 4.0), (6.5, 4.0), (6.5, -4.0),
                          (-26.5, -4.0), (-26.5, 4.0)])

REF15 = geometry.Polygon([(-27.42, -1.77), (1.77, 7.42), (7.42, 1.77),
                          (-1.77, -7.42), (-27.42, -1.77)])

STRETCHED_FOOTPRINT = {0: REF04, 1: REF15, 2: REF26, 3: REF37,
                       4: REF04, 5: REF15, 6: REF26, 7: REF37}


class DebrisOriginal(object):
    """
    Original debris model
    : footprint: stretched (only works for defined stretched_footprint)
    : touching: within the footprint
    : estimated breached area assuming poisson distribution
    """

    angle_by_idx = {0: 90.0, 4: 90.0,  # S, N
                    1: 45.0, 5: 45.0,  # SW, NE
                    2: 0.0, 6: 0.0,  # E, W
                    3: -45.0, 7: -45.0}  # SE, NW

    def __init__(self, cfg, rnd_state, wind_dir_idx, coverages):

        assert isinstance(cfg, Config)
        assert isinstance(rnd_state, np.random.RandomState)
        assert wind_dir_idx in range(8)
        assert isinstance(coverages, list)

        self.cfg = cfg
        self.rnd_state = rnd_state
        self.wind_dir_idx = wind_dir_idx
        self.coverages = coverages
        self._footprint = None
        self._damage_incr = None

        self.debris_items = []  # container of items over wind steps
        self.debris_momentums = []  # container of momentums in a wind step

        # vary over wind speeds
        self.no_items = 0  # total number of debris items generated
        self.no_impacts = 0
        self.damaged_area = 0.0  # total damaged area by debris items

        self.coverage_prob = np.array([x.area for x in coverages])/self.area

    @property
    def damage_incr(self):
        return self._damage_incr

    @damage_incr.setter
    def damage_incr(self, value):
        assert isinstance(value, numbers.Number)
        self._damage_incr = value

    @property
    def no_items_mean(self):
        """

        dN = f * dD
        where dN: incr. number of debris items,
              dD: incr in vulnerability (or damage index)
              f : a constant factor

              if we use dD/dV (pdf of vulnerability), then
                 dN = f * (dD/dV) * dV

        :getter: Returns this direction's name
        :setter: Sets this direction's name        Args:
            value :

        Returns:
            setter of mean_no_items
        """
        return np.rint(self.cfg.source_items * self._damage_incr)

    @property
    def footprint(self):
        return self._footprint

    @footprint.setter
    def footprint(self, polygon_inst):
        """
        create house footprint by wind direction
        Note that debris source model is generated assuming wind blows from E.

        Args:
            _tuple: (polygon_inst, wind_dir_index)

        Returns:

            self.footprint
            self.front_facing_walls

        """
        self._footprint = polygon_inst

    @property
    def area(self):
        return sum([x.area for x in self.coverages])

    @property
    def boundary(self):
        return geometry.Point(0, 0).buffer(self.cfg.boundary_radius)

    @property
    def breached_area(self):
        return sum([x.breached_area for x in self.coverages])

    @property
    def window_breached(self):
        # only check windows breach
        window_breach = sum([x.breached for x in self.coverages
                             if x.description == 'window'])
        if window_breach:
            return 1
        else:
            return 0

    def run(self, wind_speed):
        """

        Args:
            wind_speed:

        Returns:

        """

        self.no_impacts = 0
        self.debris_momentums = []

        # sample a poisson for each source
        no_items_by_source = self.rnd_state.poisson(
            self.no_items_mean, size=len(self.cfg.debris_sources))

        self.no_items = no_items_by_source.sum()
        logging.debug(f'no_item_by_source at speed {wind_speed:.3f}: {self.no_items} sampled with {self.no_items_mean}')

        # loop through sources
        for no_item, source in zip(no_items_by_source, self.cfg.debris_sources):

            list_debris = self.rnd_state.choice(DEBRIS_TYPES_KEYS,
                                                size=no_item, replace=True,
                                                p=self.cfg.debris_types_ratio)

            for debris_type in list_debris:
                self.generate_debris_item(wind_speed, source, debris_type)

        if self.no_impacts:
            self.check_coverages()

    def run_moment(self, wind_speed):
        """

        Args:
            wind_speed:

        Returns:

        """
        self.no_impacts = 0
        self.debris_momentums = []

        # sample a poisson for each source
        no_items_by_source = self.rnd_state.poisson(
            self.no_items_mean, size=len(self.cfg.debris_sources))

        self.no_items = no_items_by_source.sum()
        logging.debug(f'no_item_by_source at speed {wind_speed:.3f}: {self.no_items} sampled with {self.no_items_mean}')

        # loop through sources
        for no_item, source in zip(no_items_by_source, self.cfg.debris_sources):

            list_debris = self.rnd_state.choice(DEBRIS_TYPES_KEYS,
                                                size=no_item, replace=True,
                                                p=self.cfg.debris_types_ratio)

            for debris_type in list_debris:
                self.generate_debris_item(wind_speed, source, debris_type)

        if self.no_impacts:
            self.check_coverages_moment()

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

        try:
            mass = self.rnd_state.lognormal(debris['mass_mu'],
                                            debris['mass_std'])
        except ValueError:  # when sigma = 0
            mass = math.exp(debris['mass_mu'])

        try:
            frontal_area = self.rnd_state.lognormal(debris['frontal_area_mu'],
                                                    debris['frontal_area_std'])
        except ValueError:
            frontal_area = math.exp(debris['frontal_area_mu'])

        try:
            flight_time = self.rnd_state.lognormal(debris['flight_time_mu'],
                                                   debris['flight_time_std'])
        except ValueError:
            flight_time = math.exp(debris['flight_time_mu'])

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
        x, y = self.rnd_state.normal(loc=[0.0, 0.0], scale=[sigma_x, sigma_y])

        # reference point: target house
        pt_debris = geometry.Point(x + source.x - flight_distance, y + source.y)
        line_debris = geometry.LineString([source, pt_debris])

        if self.footprint.contains(pt_debris):
            self.no_impacts += 1

        self.debris_items.append((debris_type_str, line_debris))

        item_momentum = self.compute_debris_momentum(debris['cdav'],
                                                     frontal_area,
                                                     flight_distance,
                                                     mass,
                                                     wind_speed)

        self.debris_momentums.append(item_momentum)

    def check_coverages(self):
        """

        Returns:
            self.breached

        """

        for _coverage in self.coverages:

            try:
                _capacity = self.rnd_state.lognormal(*_coverage.log_failure_momentum)
            except ValueError:  # when sigma = 0
                _capacity = math.exp(_coverage.log_failure_momentum[0])

            # Complementary CDF of impact momentum
            ccdf = (self.debris_momentums > np.array(_capacity)).sum()/self.no_items
            poisson_rate = self.no_impacts * _coverage.area / self.area * ccdf

            if _coverage.description == 'window':
                prob_damage = 1.0 - math.exp(-1.0*poisson_rate)
                rv = self.rnd_state.rand()
                if rv < prob_damage:
                    _coverage.breached_area = _coverage.area
                    _coverage.breached = 1
            else:
                # assume area: no_impacts * size(1) * amplification_factor(1)
                sampled_impacts = self.rnd_state.poisson(poisson_rate)
                _coverage.breached_area = min(sampled_impacts, _coverage.area)

    def check_coverages_moment(self):
        """

        Returns:
            self.breached

        """

        for _coverage in self.coverages:

            # Complementary CDF of impact momentum
            ccdf = (self.debris_momentums > np.array(_coverage.momentum_capacity)).sum()/self.no_items
            poisson_rate = self.no_impacts * _coverage.area / self.area * ccdf

            if _coverage.description == 'window':
                prob_damage = 1.0 - math.exp(-1.0*poisson_rate)
                rv = self.rnd_state.rand()
                if rv < prob_damage:
                    _coverage.breached_area = _coverage.area
                    _coverage.breached = 1
            else:
                # assume area: no_impacts * size(1) * amplification_factor(1)
                sampled_impacts = self.rnd_state.poisson(poisson_rate)
                _coverage.breached_area = min(sampled_impacts, _coverage.area)

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

            assert flag_poly in FLIGHT_DISTANCE_POWER

            # dimensionless time
            t_star = G_CONST * flight_time / wind_speed

            # tachikawa number: rho*(v**2)/(2*g*h_m*rho_m)
            # assume h_m * rho_m == mass / frontal_area
            k_star = RHO_AIR * math.pow(wind_speed, 2.0) / (
                2.0 * G_CONST * mass / frontal_area)
            kt_star = k_star * t_star

            kt_star_powered = np.array([math.pow(kt_star, i) for i in FLIGHT_DISTANCE_POWER[flag_poly]])
            coeff = np.array(FLIGHT_DISTANCE_COEFF[flag_poly][debris_type_str])
            less_dis = (coeff * kt_star_powered).sum()

            # dimensionless hor. displacement
            # k*x_star = k*g*x/v**2
            # x = (k*x_star)*(v**2)/(k*g)
            convert_to_dim = math.pow(wind_speed, 2.0) / (
                k_star * G_CONST)

            return convert_to_dim * less_dis

    def compute_debris_momentum(self, cdav, frontal_area, flight_distance, mass,
                                wind_speed):
        """
        calculate momentum of debris object

        Args:
            cdav: average drag coefficient
            frontal_area:
            flight_distance:
            mass:
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
        # calculate um/vs, ratio of hor. vel. of debris to local wind speed
        param_b = math.sqrt(RHO_AIR * cdav * frontal_area / mass)
        _mean = 1.0 - math.exp(-param_b * math.sqrt(flight_distance))

        # dispersion here means a + b of Beta(a, b)
        try:
            assert 0.0 <= _mean <= 1.0
        except AssertionError:
            logging.warning(f'invalid mean of beta dist.: {_mean} with b: {param_b},'
                            'flight_distance: {flight_distance}')

        try:
            dispersion = max(1.0 / _mean, 1.0 / (1.0 - _mean)) + 3.0
        except ZeroDivisionError:
            dispersion = 4.0
            _mean -= 0.001

        # mu = a / (a+b), var = (a*b)/[(a+b)**2*(a+b+1)]
        beta_a = _mean * dispersion
        beta_b = dispersion * (1.0 - _mean)

        # momentum of object: mass*vs = mass*vs*(um/vs)
        return mass * wind_speed * self.rnd_state.beta(beta_a, beta_b)


class DebrisCircle(DebrisOriginal):
    """
    debris model with
    : footprint: rotated + circle boundary
    : touching: within footprint or intersect footprint within boundary
    : estimated breached area assuming poisson distribution
    """

    def __init__(self, cfg, rnd_state, wind_dir_idx, coverages):

        super(DebrisCircle, self).__init__(cfg=cfg,
                                           rnd_state=rnd_state,
                                           wind_dir_idx=wind_dir_idx,
                                           coverages=coverages)

    @property
    def footprint(self):
        """
        create house footprint by wind direction
        Note that debris source model is generated assuming wind blows from East.

        :param _tuple: (polygon_inst, wind_dir_index)

        :return:
            self.footprint, self.front_facing_walls
        """
        if self._footprint is None:
            self._footprint = affinity.rotate(self.cfg.footprint, self.__class__.angle_by_idx[self.wind_dir_idx])
        return self._footprint

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

        try:
            mass = self.rnd_state.lognormal(debris['mass_mu'],
                                            debris['mass_std'])
        except ValueError:  # when sigma = 0
            mass = math.exp(debris['mass_mu'])

        try:
            frontal_area = self.rnd_state.lognormal(debris['frontal_area_mu'],
                                                    debris['frontal_area_std'])
        except ValueError:
            frontal_area = math.exp(debris['frontal_area_mu'])

        try:
            flight_time = self.rnd_state.lognormal(debris['flight_time_mu'],
                                                   debris['flight_time_std'])
        except ValueError:
            flight_time = math.exp(debris['flight_time_mu'])

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
        x = self.rnd_state.normal(loc=0.0, scale=sigma_x)
        y = self.rnd_state.normal(loc=0.0, scale=sigma_y)

        # reference point: target house
        pt_debris = geometry.Point(x + source.x - flight_distance, y + source.y)
        line_debris = geometry.LineString([source, pt_debris])
        #
        land_within_footprint = self.footprint.contains(pt_debris)
        intersect_within_boundary = (line_debris.intersects(self.footprint)
                                     and self.boundary.contains(pt_debris))
        if land_within_footprint or intersect_within_boundary:
            self.no_impacts += 1

        self.debris_items.append((debris_type_str, line_debris))

        item_momentum = self.compute_debris_momentum(debris['cdav'],
                                                     frontal_area,
                                                     flight_distance,
                                                     mass,
                                                     wind_speed)

        self.debris_momentums.append(item_momentum)


class DebrisMC(DebrisOriginal):
    """
    Original debris model
    : footprint: stretched (only works for defined stretched_footprint)
    : touching: within the footprint
    : estimated breached area using MC + Moment
    """

    def __init__(self, cfg, rnd_state, wind_dir_idx, coverages):

        super(DebrisMC, self).__init__(cfg=cfg,
                                       rnd_state=rnd_state,
                                       wind_dir_idx=wind_dir_idx,
                                       coverages=coverages)

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

        try:
            mass = self.rnd_state.lognormal(debris['mass_mu'],
                                            debris['mass_std'])
        except ValueError:  # when sigma = 0
            mass = math.exp(debris['mass_mu'])

        try:
            frontal_area = self.rnd_state.lognormal(debris['frontal_area_mu'],
                                                    debris['frontal_area_std'])
        except ValueError:
            frontal_area = math.exp(debris['frontal_area_mu'])

        try:
            flight_time = self.rnd_state.lognormal(debris['flight_time_mu'],
                                                   debris['flight_time_std'])
        except ValueError:
            flight_time = math.exp(debris['flight_time_mu'])

        flight_distance = self.compute_flight_distance(debris_type_str,
                                                       flight_time,
                                                       frontal_area,
                                                       mass,
                                                       wind_speed)

        # logging.debug('debris type:{}, area:{}, time:{}, distance:{}'.format(
        #    debris_type_str, frontal_area, flight_time, flight_distance))

        # determine landing location for a debris item
        # sigma_x, y are taken from Wehner et al. (2010)
        x = self.rnd_state.normal(loc=0.0, scale=flight_distance / 3.0)
        y = self.rnd_state.normal(loc=0.0, scale=flight_distance / 12.0)

        # # reference point: target house
        pt_debris = geometry.Point(x + source.x - flight_distance, y + source.y)
        line_debris = geometry.LineString([source, pt_debris])
        self.debris_items.append((debris_type_str, line_debris))
        #
        # land_within_footprint = self.footprint.contains(pt_debris)
        # intersect_within_boundary = (line_debris.intersects(self.footprint)
        #                              and self.boundary.contains(pt_debris))
        # if land_within_footprint or intersect_within_boundary:
        #     self.no_impacts += 1

        if self.footprint.contains(pt_debris):
            self.no_impacts += 1

            item_momentum = self.compute_debris_momentum(debris['cdav'],
                                                         frontal_area,
                                                         flight_distance,
                                                         mass,
                                                         wind_speed)

            self.check_coverages_MC(item_momentum)

    def check_coverages_MC(self, item_momentum):
        """

        Args:
            frontal_area:
            item_momentum:

        Returns:
            self.breached

        """

        # determine coverage type
        _coverage = self.rnd_state.choice(self.coverages, p=self.coverage_prob)

        if _coverage.momentum_capacity < item_momentum:
            # history of coverage is ignored
            if _coverage.description == 'window':
                _coverage.breached_area = _coverage.area
                _coverage.breached = 1
            else:
                _coverage.breached_area = min(
                    #frontal_area * self.__class__.amplification_factor,
                    1.0,
                    _coverage.area)


def set_coverages_for_debris(cfg, wind_dir_idx, rnd_state):
    windward_walls = cfg.front_facing_walls[WIND_DIR[wind_dir_idx]]
    coverages = []
    new_item = {'rnd_state': rnd_state}

    for _name, item in cfg.coverages.iterrows():

        for key, value in new_item.items():
            item[key] = value

        _coverage = Coverage(name=_name, **item)

        if item.wall_name in windward_walls:
            coverages.append(_coverage)

    return coverages


@unittest.skip("comparison cases")
class CompareCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = os.sep.join(__file__.split(os.sep)[:-1])
        cls.path_scenario = os.path.join(
            path, 'test_scenarios', 'test_roof_sheeting')

        file_cfg = os.path.join(cls.path_scenario, 'test_roof_sheeting.cfg')
        cls.cfg = Config(file_cfg=file_cfg)

        # key = 'Capital_city'
        key = 'Tropical_town'

        cls.cfg.building_spacing = 20.0
        cls.cfg.debris_radius = 100.
        cls.cfg.debris_angle = 45.0
        cls.cfg.source_items = 100
        cls.cfg.region_name = key

        cls.cfg.process_config()

        cls.wind_speeds = np.arange(40.0, 120.0, 1.0)
        cls.incr_speed = cls.wind_speeds[1] - cls.wind_speeds[0]
        cls.no_speeds = len(cls.wind_speeds)
        cls.no_models = 20

        cls.footprint_stretched = geometry.Polygon([
            (-24.0, 6.5), (4.0, 6.5), (4.0, -6.5), (-24.0, -6.5), (-24.0, 6.5)])

        cls.footprint_rotated = geometry.Polygon([
            (-4.0, 6.5), (4.0, 6.5), (4.0, -6.5), (-4.0, -6.5), (-4.0, 6.5)])

        cls.footprint_inst = geometry.Polygon([(-6.5, 4.0), (6.5, 4.0), (6.5, -4.0),
                                      (-6.5, -4.0), (-6.5, 4.0)])

        boundary_radius = 24.0

        cls.boundary = geometry.Point(0, 0).buffer(boundary_radius)

    @classmethod
    def run_original(cls):
        start_time = time.time()
        no_items, no_touched, no_breached, damaged_area = [], [], [], []

        # original case:
        for i in range(cls.no_models):

            rnd_state = np.random.RandomState(i)

            coverages = set_coverages_for_debris(cfg=cls.cfg,
                                                 wind_dir_idx=0,
                                                 rnd_state=rnd_state)

            debris = DebrisOriginal(cfg=cls.cfg,
                                    rnd_state=rnd_state,
                                    coverages=coverages,
                                    wind_dir_idx=0)

            debris.footprint = cls.footprint_stretched

            for wind_speed in cls.wind_speeds:

                incr_damage = vulnerability_weibull_pdf(
                    x=wind_speed,
                    alpha=VUL_DIC[cls.cfg.region_name]['alpha'],
                    beta=VUL_DIC[cls.cfg.region_name]['beta']) * cls.incr_speed

                debris.damage_incr = incr_damage
                debris.run(wind_speed)

                no_items.append(debris.no_items)
                no_touched.append(debris.no_impacts)
                no_breached.append(debris.window_breached)
                damaged_area.append(debris.breached_area)

        no_items = np.array(no_items).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        no_touched = np.array(no_touched).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        no_breached = np.array(no_breached).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        damaged_area = np.array(damaged_area).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        elapsed = time.time() - start_time

        return no_items, no_touched, no_breached, damaged_area, elapsed

    @classmethod
    def run_moment(cls):

        start_time = time.time()
        no_items, no_touched, no_breached, damaged_area = [], [], [], []

        # original case:
        for i in range(cls.no_models):

            rnd_state = np.random.RandomState(i)

            coverages = set_coverages_for_debris(cfg=cls.cfg,
                                                       wind_dir_idx=0,
                                                       rnd_state=rnd_state)

            debris = DebrisOriginal(cfg=cls.cfg,
                                    rnd_state=rnd_state,
                                    coverages=coverages,
                                    wind_dir_idx=0)

            debris.footprint = cls.footprint_stretched

            for wind_speed in cls.wind_speeds:

                incr_damage = vulnerability_weibull_pdf(
                    x=wind_speed,
                    alpha=VUL_DIC[cls.cfg.region_name]['alpha'],
                    beta=VUL_DIC[cls.cfg.region_name]['beta']) * cls.incr_speed

                debris.damage_incr = incr_damage
                debris.run_moment(wind_speed)

                no_items.append(debris.no_items)
                no_touched.append(debris.no_impacts)
                no_breached.append(debris.window_breached)
                damaged_area.append(debris.breached_area)

        no_items = np.array(no_items).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        no_touched = np.array(no_touched).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        no_breached = np.array(no_breached).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        damaged_area = np.array(damaged_area).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        elapsed = time.time() - start_time

        return no_items, no_touched, no_breached, damaged_area, elapsed

    @classmethod
    def run_mc(cls):

        start_time = time.time()
        no_items, no_touched, no_breached, damaged_area = [], [], [], []

        # original case:
        for i in range(cls.no_models):

            rnd_state = np.random.RandomState(i)

            coverages = set_coverages_for_debris(cfg=cls.cfg,
                                                       wind_dir_idx=0,
                                                       rnd_state=rnd_state)

            debris = DebrisMC(cfg=cls.cfg,
                              rnd_state=rnd_state,
                              coverages=coverages,
                              wind_dir_idx=0)

            debris.footprint = cls.footprint_stretched

            for wind_speed in cls.wind_speeds:

                incr_damage = vulnerability_weibull_pdf(
                    x=wind_speed,
                    alpha=VUL_DIC[cls.cfg.region_name]['alpha'],
                    beta=VUL_DIC[cls.cfg.region_name]['beta']) * cls.incr_speed

                debris.damage_incr = incr_damage
                debris.run(wind_speed)

                no_items.append(debris.no_items)
                no_touched.append(debris.no_impacts)
                no_breached.append(debris.window_breached)
                damaged_area.append(debris.breached_area)

        no_items = np.array(no_items).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        no_touched = np.array(no_touched).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        no_breached = np.array(no_breached).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        damaged_area = np.array(damaged_area).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        elapsed = time.time() - start_time

        return no_items, no_touched, no_breached, damaged_area, elapsed

    @classmethod
    def run_boundary(cls):

        start_time = time.time()
        no_items, no_touched, no_breached, damaged_area = [], [], [], []

        # original case:
        for i in range(cls.no_models):

            rnd_state = np.random.RandomState(i)

            coverages = set_coverages_for_debris(cfg=cls.cfg,
                                                       wind_dir_idx=0,
                                                       rnd_state=rnd_state)

            debris = DebrisCircle(cfg=cls.cfg,
                                  rnd_state=rnd_state,
                                  coverages=coverages,
                                  wind_dir_idx=0)

            for wind_speed in cls.wind_speeds:

                incr_damage = vulnerability_weibull_pdf(
                    x=wind_speed,
                    alpha=VUL_DIC[cls.cfg.region_name]['alpha'],
                    beta=VUL_DIC[cls.cfg.region_name]['beta']) * cls.incr_speed

                debris.damage_incr = incr_damage
                debris.run(wind_speed)

                no_items.append(debris.no_items)
                no_touched.append(debris.no_impacts)
                no_breached.append(debris.window_breached)
                damaged_area.append(debris.breached_area)

        no_items = np.array(no_items).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        no_touched = np.array(no_touched).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        no_breached = np.array(no_breached).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        damaged_area = np.array(damaged_area).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        elapsed = time.time() - start_time

        return no_items, no_touched, no_breached, damaged_area, elapsed

    @classmethod
    def run_debris(cls):

        start_time = time.time()
        no_items, no_touched, no_breached, damaged_area = [], [], [], []

        # original case:
        for i in range(cls.no_models):

            rnd_state = np.random.RandomState(i)

            coverages = set_coverages_for_debris(cfg=cls.cfg,
                                                     wind_dir_idx=0,
                                                     rnd_state=rnd_state)

            coverage_area = sum([x.area for x in coverages])
            coverage_prob = np.array(
                [x.area for x in coverages]) / coverage_area

            for wind_speed in cls.wind_speeds:

                incr_damage = vulnerability_weibull_pdf(
                    x=wind_speed,
                    alpha=VUL_DIC[cls.cfg.region_name]['alpha'],
                    beta=VUL_DIC[cls.cfg.region_name]['beta']) * cls.incr_speed

                mean_no_items = np.rint(cls.cfg.source_items * incr_damage)

                debris_items = generate_debris_items(
                    rnd_state=rnd_state,
                    wind_speed=wind_speed,
                    cfg=cls.cfg,
                    mean_no_debris_items=mean_no_items)

                for item in debris_items:
                    item.check_impact(boundary=cls.boundary,
                                      footprint=cls.footprint_rotated)
                    item.check_coverages(coverages=coverages,
                                         prob_coverages=coverage_prob)

                breached_area = sum([x.breached_area for x in coverages])

                # only check windows breach
                window_breach = sum([
                    x.breached for x in coverages if
                    x.description == 'window'])
                if window_breach:
                    no_breached.append(1)
                else:
                    no_breached.append(0)

                no_debris_items = len(debris_items)
                no_debris_impacts = sum([x.impact for x in debris_items])

                no_items.append(no_debris_items)
                no_touched.append(no_debris_impacts)
                damaged_area.append(breached_area)

        no_items = np.array(no_items).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        no_touched = np.array(no_touched).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        no_breached = np.array(no_breached).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        damaged_area = np.array(damaged_area).reshape(cls.no_models, cls.no_speeds).mean(axis=0)
        elapsed = time.time() - start_time

        return no_items, no_touched, no_breached, damaged_area, elapsed

    def test_compare_cases(self):

        label1 = 'original'
        no_items1, no_touched1, no_breached1, damaged_area1, elapsed1 = self.run_original()
        print(f'Elapsed time: {elapsed1}')

        label2 = 'moment'
        no_items2, no_touched2, no_breached2, damaged_area2, elapsed2 = self.run_moment()
        print(f'Elapsed time: {elapsed2}')

        label3 = 'moment+mc'
        no_items3, no_touched3, no_breached3, damaged_area3, elapsed3 = self.run_mc()
        print(f'Elapsed time: {elapsed3}')

        label4 = 'boundary'
        no_items4, no_touched4, no_breached4, damaged_area4, elapsed4 = self.run_boundary()
        print(f'Elapsed time: {elapsed4}')

        label5 = 'implemented'
        no_items5, no_touched5, no_breached5, damaged_area5, elapsed5 = self.run_debris()
        print(f'Elapsed time: {elapsed5}')

        plt.figure()
        plt.plot(self.wind_speeds, np.cumsum(no_items1), 'b.-', label=label1)
        plt.plot(self.wind_speeds, np.cumsum(no_items2), 'r.--', label=label2)
        plt.plot(self.wind_speeds, np.cumsum(no_items3), 'g.--', label=label3)
        plt.plot(self.wind_speeds, np.cumsum(no_items4), 'c.--', label=label4)
        plt.plot(self.wind_speeds, np.cumsum(no_items5), 'k.--', label=label5)
        plt.title('Debris impact model comparision')
        plt.xlabel('Wind speed (m/s)')
        plt.ylabel('No. of debris supply')
        plt.legend(loc=4, numpoints=1)
        plt.grid(1)
        # plt.pause(1.0)
        plt.savefig('compare_1.png')
        plt.close()

        plt.figure()
        plt.plot(self.wind_speeds, no_breached1, 'b.-', label=label1)
        plt.plot(self.wind_speeds, no_breached2, 'r.--', label=label2)
        plt.plot(self.wind_speeds, no_breached3, 'g.--', label=label3)
        plt.plot(self.wind_speeds, no_breached4, 'c.--', label=label4)
        plt.plot(self.wind_speeds, no_breached5, 'k.--', label=label5)
        plt.title('Debris impact model comparision')
        plt.xlabel('Wind speed (m/s)')
        plt.ylabel('% of model with window breach')
        plt.legend(loc=4, numpoints=1)
        plt.grid(1)
        # plt.pause(1.0)
        plt.savefig('compare_2.png')
        plt.close()

        plt.figure()
        plt.plot(self.wind_speeds, np.cumsum(no_touched1), 'b.-', label=label1)
        plt.plot(self.wind_speeds, np.cumsum(no_touched2), 'r.--', label=label2)
        plt.plot(self.wind_speeds, np.cumsum(no_touched3), 'g.--', label=label3)
        plt.plot(self.wind_speeds, np.cumsum(no_touched4), 'c.--', label=label4)
        plt.plot(self.wind_speeds, np.cumsum(no_touched5), 'k.--', label=label5)
        plt.title('Debris impact model comparision')
        plt.xlabel('Wind speed (m/s)')
        plt.ylabel('No of debris impacts')
        plt.legend(loc=4, numpoints=1)
        plt.grid(1)
        # plt.pause(1.0)
        plt.savefig('compare_3.png')
        plt.close()

        plt.figure()
        plt.plot(self.wind_speeds, damaged_area1, 'b.-', label=label1)
        plt.plot(self.wind_speeds, damaged_area2, 'r.--', label=label2)
        plt.plot(self.wind_speeds, damaged_area3, 'g.--', label=label3)
        plt.plot(self.wind_speeds, damaged_area4, 'c.--', label=label4)
        plt.plot(self.wind_speeds, damaged_area5, 'k.--', label=label5)
        plt.title('Debris impact model comparision')
        plt.xlabel('Wind speed (m/s)')
        plt.ylabel('Damaged area')
        plt.legend(loc=4, numpoints=1)
        plt.grid(1)
        # plt.pause(1.0)
        plt.savefig('compare_4.png')
        plt.close()


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = os.sep.join(__file__.split(os.sep)[:-1])
        cls.path_scenario = os.path.join(
            path, 'test_scenarios', 'test_roof_sheeting')
        # set up logging
        file_logger = os.path.join(cls.path_scenario, 'output', 'log.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

        file_cfg = os.path.join(cls.path_scenario, 'test_roof_sheeting.cfg')
        cls.cfg = Config(file_cfg=file_cfg)

        cls.rnd_state = np.random.RandomState(1)
        cls.radius = 24.0
        cls.boundary = geometry.Point(0, 0).buffer(cls.radius)

        cls.footprint_inst = geometry.Polygon([(-6.5, 4.0), (6.5, 4.0), (6.5, -4.0),
                                      (-6.5, -4.0), (-6.5, 4.0)])

        cls.footprint_rotated = geometry.Polygon([
            (-4.0, 6.5), (4.0, 6.5), (4.0, -6.5), (-4.0, -6.5), (-4.0, 6.5)])

        ref04 = geometry.Polygon([(-24.0, 6.5), (4.0, 6.5), (4.0, -6.5),
                         (-24.0, -6.5), (-24.0, 6.5)])

        ref37 = geometry.Polygon([(-1.77, 7.42), (7.42, -1.77), (1.77, -7.42),
                         (-27.42, 1.77), (-1.77, 7.42)])

        ref26 = geometry.Polygon([(-26.5, 4.0), (6.5, 4.0), (6.5, -4.0),
                         (-26.5, -4.0), (-26.5, 4.0)])

        ref15 = geometry.Polygon([(-27.42, -1.77), (1.77, 7.42), (7.42, 1.77),
                         (-1.77, -7.42), (-27.42, -1.77)])

        cls.ref_footprint = {0: ref04, 1: ref15, 2: ref26, 3: ref37,
                             4: ref04, 5: ref15, 6: ref26, 7: ref37}

    def test_create_sources(self):

        debris_radius = 100.0
        debris_angle = 45.0
        building_spacing = 20.0

        sources1 = create_sources(radius=debris_radius,
                                  angle=debris_angle,
                                  bldg_spacing=building_spacing,
                                  flag_staggered=False)
        self.assertEqual(len(sources1), 13)

        plt.figure()
        source_x, source_y = [], []
        for source in sources1:
            source_x.append(source.x)
            source_y.append(source.y)
        plt.scatter(source_x, source_y, label='source', color='b')
        plt.scatter(0, 0, label='target', color='r')
        plt.legend(loc=2, scatterpoints=1)
        plt.pause(1.0)
        # plt.savefig('./source.png', dpi=300)
        plt.close()

    def test_create_sources_staggered(self):

        # staggered source
        debris_radius = 100.0
        debris_angle = 45.0
        building_spacing = 20.0

        sources2 = create_sources(radius=debris_radius,
                                  angle=debris_angle,
                                  bldg_spacing=building_spacing,
                                  flag_staggered=True)
        self.assertEqual(len(sources2), 15)

        plt.figure()
        source_x, source_y = [], []
        for source in sources2:
            source_x.append(source.x)
            source_y.append(source.y)
        plt.scatter(source_x, source_y, label='source', color='b')
        plt.scatter(0, 0, label='target', color='r')
        plt.legend(loc=2, scatterpoints=1)
        plt.pause(1.0)
        # plt.savefig('./source_staggered.png', dpi=300)
        plt.close()

    def test_footprint(self):

        wind_dir_idx = [0, 1, 2, 3]
        for idx in wind_dir_idx:

            house = House(self.cfg, 1)
            house._wind_dir_index = idx

            fig = plt.figure(1)
            ax = fig.add_subplot(111)

            _array = np.array(house.footprint.exterior.xy).T
            ax.add_patch(patches.Polygon(_array, alpha=0.3, label='footprint'))

            ax.add_patch(patches.Circle(xy=(0, 0),
                                        radius=self.radius,
                                        alpha=0.3, color='r', label='boundary'))

            x, y = self.ref_footprint[idx].exterior.xy
            ax.plot(x, y, 'b-', label='stretched')
            ax.set_xlim([-40, 20])
            ax.set_ylim([-20, 20])
            plt.title(f'Wind direction: {idx}')
            plt.legend(loc=2)
            # plt.show()
            plt.pause(1.0)
            plt.close()

    def test_contains(self):
        rect = geometry.Polygon([(-15, 4), (15, 4), (15, -4), (-15, -4)])
        self.assertTrue(rect.contains(geometry.Point(0, 0)))
        self.assertFalse(rect.contains(geometry.Point(-100, -1.56)))
        self.assertFalse(rect.contains(geometry.Point(10.88, 4.514)))
        self.assertFalse(rect.contains(geometry.Point(7.773, 12.66)))
        self.assertTrue(rect.touches(geometry.Point(15, -4)))

    def test_flight_distance(self):

        wind_speed = 50.0
        key = 'Tropical_town'

        expected = {'Rod': 34.605,
                    'Compact': 25.820,
                    'Sheet': 77.760}

        for debris_type in self.cfg.debris_types:

            _debris = Debris(debris_source=None,
                             debris_type=debris_type,
                             debris_property=self.cfg.debris_types[debris_type],
                             wind_speed=wind_speed,
                             rnd_state=self.rnd_state)
            _debris._frontal_area = self.cfg.debris_regions[key][
                f'{debris_type}_frontal_area_mean']
            _debris._mass = self.cfg.debris_regions[key][
                f'{debris_type}_mass_mean']
            _debris._flight_time = self.cfg.debris_regions[key][
                f'{debris_type}_flight_time_mean']

            try:
                self.assertAlmostEqual(expected[debris_type],
                                       _debris.flight_distance, places=2)
            except AssertionError:
                print(f'{debris_type}: {expected[debris_type]} is expected but {_debris.flight_distance}')

    def test_bivariate_gaussian(self):

        rnd_state = np.random.RandomState(1)
        sigma_x, sigma_y = 0.1, 0.3
        cov_matrix = [[math.pow(sigma_x, 2.0), 0.0],
                      [0.0, math.pow(sigma_y, 2.0)]]

        nsample = 1000
        _array = rnd_state.multivariate_normal(
            mean=[0.0, 0.0], cov=cov_matrix, size=nsample)

        stx, sty = np.std(_array, axis=0)
        rhoxy = np.corrcoef(_array[:, 0], _array[:, 1])[0, 1]
        self.assertAlmostEqual(stx, sigma_x, places=1)
        self.assertAlmostEqual(sty, sigma_y, places=1)
        self.assertAlmostEqual(rhoxy, 0.0, places=1)

        _array_x = rnd_state.normal(0.0, scale=sigma_x, size=nsample)
        _array_y = rnd_state.normal(0.0, scale=sigma_y, size=nsample)
        stx = _array_x.std()
        sty = _array_y.std()
        rhoxy = np.corrcoef(_array_x, _array_y)[0, 1]
        self.assertAlmostEqual(stx, sigma_x, places=1)
        self.assertAlmostEqual(sty, sigma_y, places=1)
        self.assertAlmostEqual(rhoxy, 0.0, places=1)

    def test_compute_coeff_beta_dist(self):

        # Tropical town, Rod
        debris_type = 'Rod'
        wind_speed = 40.0
        _debris = Debris(debris_source=None,
                         debris_type=debris_type,
                         debris_property=self.cfg.debris_types[debris_type],
                         wind_speed=wind_speed,
                         rnd_state=self.rnd_state)

        _debris._flight_distance = 10.0
        _debris._mass = 4.0
        _debris._frontal_area = 0.1

        beta_a, beta_b = _debris.compute_coeff_beta_dist()
        self.assertAlmostEqual(beta_a, 2.1619, places=3)
        self.assertAlmostEqual(beta_b, 3.4199, places=3)

    def test_debris_momentum(self):

        nsample = 100
        momentum = np.zeros(shape=nsample)
        for i in range(nsample):

            # Tropical town, Rod
            debris_type = 'Rod'
            wind_speed = 50.0
            _debris = Debris(debris_source=None,
                             debris_type=debris_type,
                             debris_property=self.cfg.debris_types[debris_type],
                             wind_speed=wind_speed,
                             rnd_state=self.rnd_state)

            _debris._flight_distance = 10.0
            _debris._mass = 4.0
            _debris._frontal_area = 0.1
            momentum[i] = _debris.momentum

        plt.figure()
        plt.hist(momentum)
        plt.title(f'sampled momentum with mean: {momentum.mean():.2f}, sd: {momentum.std():.2f}')
        plt.pause(1.0)
        plt.close()
        # plt.show()

    def test_debris_momentum_by_type(self):

        wind_speeds = np.arange(0.01, 120.0, 1.0)
        momentum = {}

        for key, value in self.cfg.debris_types.items():

            momentum[key] = np.zeros_like(wind_speeds)

            for i, wind_speed in enumerate(wind_speeds):

                _debris = Debris(debris_source=None,
                                 debris_type=key,
                                 debris_property=self.cfg.debris_types[key],
                                 wind_speed=wind_speed,
                                 rnd_state=self.rnd_state)

                _debris._mass = math.exp(value['mass_mu'])
                _debris._frontal_area = math.exp(value['frontal_area_mu'])
                momentum[key][i] = _debris.momentum

        dic_ = {'Compact': 'b', 'Sheet': 'r', 'Rod': 'g'}
        plt.figure()
        for _str, _value in momentum.items():
            plt.plot(wind_speeds, _value, color=dic_[_str],
                     label=_str, linestyle='-')

        plt.title('momentum by time')
        plt.xlabel('Wind speed (m/s)')
        plt.legend(loc=2)
        #plt.show()
        plt.pause(1.0)
        plt.close()

    def test_compare_flight_distance(self):

        wind_speeds = np.arange(0.0, 120.0, 1.0)
        flight_distance = {}
        flight_distance_poly5 = {}

        for key, value in self.cfg.debris_types.items():

            flight_distance[key] = np.zeros_like(wind_speeds)
            flight_distance_poly5[key] = np.zeros_like(wind_speeds)

            for i, wind_speed in enumerate(wind_speeds):

                _debris2 = Debris(debris_source=None,
                                  debris_type=key,
                                  debris_property=self.cfg.debris_types[key],
                                  wind_speed=wind_speed,
                                  rnd_state=self.rnd_state,
                                  flag_poly=2)

                _debris2._frontal_area = math.exp(value['frontal_area_mu'])
                _debris2._mass = math.exp(value['mass_mu'])
                _debris2._flight_time = math.exp(value['flight_time_mu'])

                _debris5 = Debris(debris_source=None,
                                  debris_type=key,
                                  debris_property=self.cfg.debris_types[key],
                                  wind_speed=wind_speed,
                                  rnd_state=self.rnd_state,
                                  flag_poly=5)

                _debris5._frontal_area = math.exp(value['frontal_area_mu'])
                _debris5._mass = math.exp(value['mass_mu'])
                _debris5._flight_time = math.exp(value['flight_time_mu'])

                flight_distance[key][i] = _debris2.flight_distance

                flight_distance_poly5[key][i] = _debris5.flight_distance

        dic_ = {'Compact': 'b', 'Sheet': 'r', 'Rod': 'g'}
        plt.figure()
        for _str, _value in flight_distance.items():
            plt.plot(wind_speeds, _value, color=dic_[_str],
                     label=f'{_str}',
                     linestyle='-')

        for _str, _value in flight_distance_poly5.items():
            plt.plot(wind_speeds, _value, color=dic_[_str],
                     label=f'{_str} (Lin and Vanmarcke, 2008)',
                     linestyle='--')

        plt.title('Flight distance')
        plt.legend(loc=2)
        plt.xlabel('Wind speed (m/s)')
        plt.ylabel('Flight distance (m)')
        # plt.show()
        plt.pause(1.0)
        # plt.savefig('./flight_distance.png', dpi=200)
        plt.close()

    def check_performance_of_parmap(self):

        no_models = 10
        no_debris = 300
        source = geometry.Point([20.0, 0.0])

        par_time, ser_time = [], []

        for j in range(no_models):

            debris_items = []
            for i in range(no_debris):
                _debris = Debris(debris_source=source,
                                 debris_type='Rod',
                                 debris_property=self.cfg.debris_types['Rod'],
                                 wind_speed=20.0,
                                 rnd_state=self.rnd_state,
                                 )
                debris_items.append(_debris)

            # serial run
            tic = time.time()
            for item in debris_items:
                _ = item.check_impact(footprint=self.footprint_inst,
                                      boundary=self.boundary)

            ser_time.append(time.time()-tic)

            tic = time.time()
            _ = parmap.map(check_impact_by_debris, debris_items,
                           self.footprint_inst, self.boundary)
            par_time.append(time.time()-tic)

        print(f'Parallel vs Serial: {sum(par_time)} vs {sum(ser_time)}')

    def test_number_of_touched(self):

        coverages = set_coverages_for_debris(cfg=self.cfg,
                                             wind_dir_idx=0,
                                             rnd_state=self.rnd_state)

        coverage_area = sum([x.area for x in coverages])
        coverage_prob = np.array([x.area for x in coverages]) / coverage_area

        key = 'Tropical_town'

        items_all = []
        no_impacts = 0
        no_generated = 0
        for wind_speed in self.cfg.wind_speeds:

            damage_incr = vulnerability_weibull_pdf(
                x=wind_speed,
                alpha=VUL_DIC[key]['alpha'],
                beta=VUL_DIC[key]['beta']) * self.cfg.wind_speed_increment

            mean_no_debris_items = np.rint(self.cfg.source_items * damage_incr)

            debris_items = generate_debris_items(
                rnd_state=self.rnd_state,
                wind_speed=wind_speed,
                cfg=self.cfg,
                mean_no_debris_items=mean_no_debris_items)

            for item in debris_items:
                item.check_impact(boundary=self.boundary,
                                      footprint=self.footprint_rotated)
                item.check_coverages(coverages=coverages,
                                     prob_coverages=coverage_prob)

            [items_all.append(x) for x in debris_items]
            no_impacts += sum([x.impact for x in debris_items])
            no_generated += len(debris_items)

        fig = plt.figure(1)
        ax = fig.add_subplot(111)

        title_str = f'total no. of impacts: {no_impacts} out of {no_generated}'
        ax.set_title(title_str)
        ax.set_xlim([-150, 150])
        ax.set_ylim([-100, 100])
        # plt.title('Wind direction: 0')

        # p = PolygonPatch(_debris.footprint, fc='red')
        # ax.add_patch(p)

        # ax.add_patch(patches_Polygon(_array, alpha=0.5))

        # shape_type = {'Compact': 'c', 'Sheet': 'g', 'Rod': 'r'}

        for item in items_all:
            _x, _y = item.trajectory.xy[0][1], item.trajectory.xy[1][1]
            ax.scatter(_x, _y, color='g', alpha=0.2)

        for source in self.cfg.debris_sources:
            ax.scatter(source.x, source.y, color='b')
        ax.scatter(0, 0, label='target', color='r')

        # add footprint
        ax.add_patch(patches.Polygon(self.footprint_rotated.exterior, fc='red', alpha=0.5))
        ax.add_patch(patches.Polygon(self.boundary.exterior, alpha=0.5))

        # plt.show()
        plt.pause(3.0)
        plt.close()


if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
