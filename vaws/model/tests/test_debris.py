from __future__ import division, print_function
import unittest
import os
import copy
import numpy as np
import pandas as pd
import matplotlib
import parmap
import matplotlib.pyplot as plt
import logging
import time
from scipy import stats

import numbers
import math
from shapely import geometry, affinity

from vaws.model.config import Config, WIND_DIR, DEBRIS_TYPES_KEYS
from vaws.model.debris import Debris, create_sources, determine_impact_by_debris
from vaws.model.coverage import Coverage
from vaws.model.curve import vulnerability_weibull_pdf
from vaws.model.house import House

VUL_DIC = {'Capital_city': {'alpha': 0.1586, 'beta': 3.8909},
           'Tropical_town': {'alpha': 0.1030, 'beta': 4.1825}}


class DebrisOriginal(object):
    """
    Original debris model
    : footprint: stretched
    : touching: within the footprint
    : estimated breached area assuming poisson distribution
    """

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

    def __init__(self, cfg, rnd_state, wind_dir_idx, coverages):

        assert isinstance(cfg, Config)
        assert isinstance(rnd_state, np.random.RandomState)
        assert wind_dir_idx in range(8)
        assert isinstance(coverages, dict)

        self.cfg = cfg
        self.rnd_state = rnd_state
        self.wind_dir_idx = wind_dir_idx
        self.coverages = coverages

        self._damage_incr = None

        self.debris_items = []  # container of items over wind steps
        self.debris_momentums = []  # container of momentums in a wind step

        # vary over wind speeds
        self.no_items = 0  # total number of debris items generated
        self.no_impacts = 0
        self.damaged_area = 0.0  # total damaged area by debris items

        self.coverage_idx = []
        _coverage_prob = []
        _area = 0.0
        for key, value in self.coverages.items():
            _area += value.area
            self.coverage_idx.append(key)
            _coverage_prob.append(_area)
        self.coverage_prob = np.array(_coverage_prob)/_area

        # vary over wind speeds
        self.sampled_impacts = 0

    @property
    def damage_incr(self):
        return self._damage_incr

    @damage_incr.setter
    def damage_incr(self, value):
        assert isinstance(value, numbers.Number)
        self._damage_incr = value

    @property
    def mean_no_items(self):
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
        """
        create house footprint by wind direction
        Note that debris source model is generated assuming wind blows from East.

        :param _tuple: (polygon_inst, wind_dir_index)

        :return:
            self.footprint, self.front_facing_walls
        """
        return affinity.rotate(
            self.cfg.footprint, self.__class__.angle_by_idx[self.wind_dir_idx])

    @property
    def front_facing_walls(self):
        return self.cfg.front_facing_walls[WIND_DIR[self.wind_dir_idx]]

    @property
    def area(self):
        return sum([x.area for x in self.coverages.itervalues()])

    @property
    def boundary(self):
        return geometry.Point(0, 0).buffer(self.cfg.boundary_radius)

    def run(self, wind_speed):
        """

        Args:
            wind_speed:

        Returns:

        """

        self.no_impacts = 0
        self.debris_momentums = []
        self.sampled_impacts = 0

        # sample a poisson for each source
        no_items_by_source = self.rnd_state.poisson(
            self.no_items_mean, size=len(self.cfg.debris_sources))

        self.no_items = no_items_by_source.sum()
        logging.debug('no_item_by_source at speed {:.3f}: {} sampled with {}'.format(
            wind_speed, self.no_items, self.no_items_mean))

        # loop through sources
        for no_item, source in zip(no_items_by_source, self.cfg.debris_sources):

            list_debris = self.rnd_state.choice(DEBRIS_TYPES_KEYS,
                                                size=no_item, replace=True,
                                                p=self.cfg.debris_types_ratio)

            for debris_type in list_debris:
                self.generate_debris_item(wind_speed, source, debris_type)

        if self.no_impacts:
            self.check_debris_impact()

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

        if self.footprint.contains(pt_debris):
            self.no_impacts += 1

        line_debris = geometry.LineString([source, pt_debris])
        self.debris_items.append((debris_type_str, line_debris))

        item_momentum = self.compute_debris_momentum(debris['cdav'],
                                                     frontal_area,
                                                     flight_distance,
                                                     mass,
                                                     wind_speed)

        self.debris_momentums.append(item_momentum)

    def check_debris_impact(self):
        """

        Returns:
            self.breached

        """

        for _id, _coverage in self.coverages.items():

            try:
                _capacity = self.rnd_state.lognormal(*_coverage.log_failure_momentum)
            except ValueError:  # when sigma = 0
                _capacity = math.exp(_coverage.log_failure_momentum[0])

            # Complementary CDF of impact momentum
            ccdf = 1.0*(self.debris_momentums > np.array(_capacity)).sum()/self.no_items
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

            assert flag_poly in self.__class__.flight_distance_power

            # dimensionless time
            t_star = self.__class__.g_const * flight_time / wind_speed

            # Tachikawa Number: rho*(V**2)/(2*g*h_m*rho_m)
            # assume h_m * rho_m == mass / frontal_area
            k_star = self.__class__.rho_air * math.pow(wind_speed, 2.0) / (
                2.0 * self.__class__.g_const * mass / frontal_area)
            kt_star = k_star * t_star

            kt_star_powered = np.array([math.pow(kt_star, i)
                for i in self.__class__.flight_distance_power[flag_poly]])
            coeff = np.array(self.__class__.flight_distance_coeff[
                              flag_poly][debris_type_str])
            less_dis = (coeff * kt_star_powered).sum()

            # dimensionless hor. displacement
            # k*x_star = k*g*x/V**2
            # x = (k*x_star)*(V**2)/(k*g)
            convert_to_dim = math.pow(wind_speed, 2.0) / (
                k_star * self.__class__.g_const)

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
        param_b = math.sqrt(self.__class__.rho_air * cdav * frontal_area / mass)
        _mean = 1.0 - math.exp(-param_b * math.sqrt(flight_distance))

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
        return mass * wind_speed * self.rnd_state.beta(beta_a, beta_b)


class DebrisCircle(DebrisOriginal):
    """
    debris model with
    : footprint: rotated
    : touching is intersect with footprint
    : estimated breached area assuming poisson distribution

    """

    def __init__(self, cfg, rnd_state, wind_dir_idx, coverages):

        super(DebrisTest, self).__init__(cfg=cfg,
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

        # reference point: target house
        pt_debris = geometry.Point(x + source.x - flight_distance, y + source.y)
        line_debris = geometry.LineString([source, pt_debris])

        if line_debris.intersects(self.footprint):
            self.no_impacts += 1

        line_debris = geometry.LineString([source, pt_debris])
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
    : footprint: stretched
    : touching: within the footprint
    : estimated breached area based on the MC
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
            flight_time = self.rnd_state.lognormal(self.cfg.flight_time_log_mu,
                                                   self.cfg.flight_time_log_std)
        except ValueError:
            flight_time = math.exp(self.cfg.flight_time_log_mu)

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

        # reference point: target house
        pt_debris = geometry.Point(x + source.x - flight_distance, y + source.y)
        line_debris = geometry.LineString([source, pt_debris])
        self.debris_items.append((debris_type_str, line_debris))

        if self.footprint.contains(pt_debris):

            self.no_impacts += 1

            item_momentum = self.compute_debris_momentum(debris['cdav'],
                                                         frontal_area,
                                                         flight_distance,
                                                         mass,
                                                         wind_speed)

            self.check_debris_impact_MC(item_momentum)

    def check_debris_impact_MC(self, item_momentum):
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
            _capacity = math.exp(_coverage.log_failure_momentum[0])

        if _capacity < item_momentum:
            # history of coverage is ignored
            if _coverage.description == 'window':
                _coverage.breached_area = _coverage.area
                _coverage.breached = 1
            else:
                _coverage.breached_area = min(
                    #frontal_area * self.__class__.amplification_factor,
                    1.0,
                    _coverage.area)


def area_stretched(stretch, a, b):
    return (stretch+a*2.0)*(b*2.0)


def area_circle(r, a, b):
    x_plus_a = math.sqrt(r**2 - b**2)
    theta = math.acos(x_plus_a/r)
    return theta * r**2 + x_plus_a*b + (a * 2*b)

'''
class CompareCase(unittest.TestCase):

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

        cfg_file = os.path.join(cls.path_scenario, 'test_roof_sheeting.cfg')
        cls.cfg = Config(cfg_file=cfg_file)

        cls.footprint_inst = Polygon([(-6.5, 4.0), (6.5, 4.0), (6.5, -4.0),
                                      (-6.5, -4.0), (-6.5, 4.0)])

        cls.footprint_stretched = Polygon([
            (-24.0, 6.5), (4.0, 6.5), (4.0, -6.5), (-24.0, -6.5), (-24.0, 6.5)])

        cls.footprint_rotated = Polygon([
            (-4.0, 6.5), (4.0, 6.5), (4.0, -6.5), (-4.0, -6.5), (-4.0, 6.5)])

        cls.radius = 24.0

    def test_run(self):

        # set up logging
        # file_logger = os.path.join(self.path_output, 'log_debris.txt')
        # logging.basicConfig(filename=file_logger,
        #                     filemode='w',
        #                     level=logging.DEBUG,
        #                     format='%(levelname)s %(message)s')

        wind_speeds = np.arange(40.0, 120.0, 1.0)

        #key = 'Capital_city'
        key = 'Tropical_town'

        alpha_ = VUL_DIC[key]['alpha']
        beta_ = VUL_DIC[key]['beta']

        incr_speed = wind_speeds[1] - wind_speeds[0]

        nmodels = 20

        self.cfg.building_spacing = 20.0
        self.cfg.debris_radius = 100.
        self.cfg.debris_angle = 45.0
        self.cfg.source_items = 100
        self.cfg.region_name = key

        self.cfg.process_config()

        start_time = time.time()

        no_items1 = []
        no_touched1 = []
        no_breached1 = []
        damaged_area1 = []

        # original debris model with stretched footprint
        for i in range(nmodels):

            coverages1 = copy.deepcopy(self.cfg.coverages)
            for _name, item in coverages1.iterrows():
                _coverage = Coverage(coverage_name=_name, **item)
                coverages1.loc[_name, 'coverage'] = _coverage

            debris1 = DebrisOriginal(self.cfg)
            debris1.rnd_state = np.random.RandomState(i)
            debris1.footprint = (self.footprint_stretched, 0)
            debris1.coverages = coverages1

            for wind_speed in wind_speeds:
                incr_damage = vulnerability_weibull_pdf(x=wind_speed,
                                                        alpha_=alpha_,
                                                        beta_=beta_) * incr_speed

                debris1.no_items_mean = incr_damage
                # print('{},,{}'.format(wind_speed, debris2.no_items_mean))
                debris1.run(wind_speed)
                no_items1.append(debris1.no_items)
                no_touched1.append(debris1.no_impacts)
                breached_area = sum(
                    [x.breached_area for x in debris1.coverages.itervalues()])
                damaged_area1.append(breached_area)

                # only check windows breach
                window_breach = np.array([
                    x.breached for x in debris1.coverages.itervalues()]).sum()
                if window_breach:
                    no_breached1.append(1)
                else:
                    no_breached1.append(0)

        no_items1 = np.array(no_items1).reshape(nmodels, len(wind_speeds)).mean(axis=0)
        no_touched1 = np.array(no_touched1).reshape(nmodels, len(wind_speeds)).mean(axis=0)
        no_breached1 = np.array(no_breached1).reshape(nmodels, len(wind_speeds)).mean(axis=0)
        damaged_area1 = np.array(damaged_area1).reshape(nmodels, len(wind_speeds)).mean(axis=0)
        print('Elapsed time: {}'.format(time.time()-start_time))

        start_time = time.time()
        no_items2 = []
        no_touched2 = []
        no_breached2 = []
        damaged_area2 = []

        # debris model with rotated footprint with circular boundary
        for i in range(nmodels):

            coverages2 = copy.deepcopy(self.cfg.coverages)
            for _name, item in coverages2.iterrows():
                _coverage = Coverage(coverage_name=_name, **item)
                coverages2.loc[_name, 'coverage'] = _coverage

            debris2 = Debris(self.cfg)
            debris2.rnd_state = np.random.RandomState(i)
            debris2.footprint = (self.footprint_inst, 0)
            debris2.boundary = self.radius
            debris2.coverages = coverages2

            for wind_speed in wind_speeds:
                incr_damage = vulnerability_weibull_pdf(x=wind_speed,
                                                        alpha_=alpha_,
                                                        beta_=beta_) * incr_speed

                debris2.no_items_mean = incr_damage
                # print('{},,{}'.format(wind_speed, debris2.no_items_mean))
                debris2.run(wind_speed)
                no_items2.append(debris2.no_items)
                no_touched2.append(debris2.no_impacts)
                breached_area = sum(
                    [x.breached_area for x in debris2.coverages.itervalues()])
                damaged_area2.append(breached_area)

                # only check windows breach
                window_breach = np.array([
                    x.breached for x in debris2.coverages.itervalues()]).sum()
                if window_breach:
                    no_breached2.append(1)
                else:
                    no_breached2.append(0)

        no_items2 = np.array(no_items2).reshape(nmodels, len(wind_speeds)).mean(axis=0)
        no_touched2 = np.array(no_touched2).reshape(nmodels, len(wind_speeds)).mean(axis=0)
        no_breached2 = np.array(no_breached2).reshape(nmodels, len(wind_speeds)).mean(axis=0)
        damaged_area2 = np.array(damaged_area2).reshape(nmodels, len(wind_speeds)).mean(axis=0)
        print('Elapsed time: {}'.format(time.time()-start_time))

        start_time = time.time()
        no_items3 = []
        no_touched3 = []
        no_breached3 = []
        damaged_area3 = []

        # debris model with stretched
        for i in range(nmodels):

            coverages3 = copy.deepcopy(self.cfg.coverages)
            for _name, item in coverages3.iterrows():
                _coverage = Coverage(coverage_name=_name, **item)
                coverages3.loc[_name, 'coverage'] = _coverage

            debris3 = DebrisMC(self.cfg)
            debris3.rnd_state = np.random.RandomState(i)
            debris3.footprint = (self.footprint_stretched, 0)
            debris3.coverages = coverages3

            for wind_speed in wind_speeds:
                incr_damage = vulnerability_weibull_pdf(x=wind_speed,
                                                        alpha_=alpha_,
                                                        beta_=beta_) * incr_speed

                debris3.no_items_mean = incr_damage
                # print('{},,{}'.format(wind_speed, debris2.no_items_mean))
                debris3.run(wind_speed)
                no_items3.append(debris3.no_items)
                no_touched3.append(debris3.no_impacts)
                breached_area = sum(
                    [x.breached_area for x in debris3.coverages.itervalues()])
                damaged_area3.append(breached_area)

                # only check windows breach
                window_breach = np.array([
                    x.breached for x in debris3.coverages.itervalues()]).sum()
                if window_breach:
                    no_breached3.append(1)
                else:
                    no_breached3.append(0)

        no_items3 = np.array(no_items3).reshape(nmodels, len(wind_speeds)).mean(axis=0)
        no_touched3 = np.array(no_touched3).reshape(nmodels, len(wind_speeds)).mean(axis=0)
        no_breached3 = np.array(no_breached3).reshape(nmodels, len(wind_speeds)).mean(axis=0)
        damaged_area3 = np.array(damaged_area3).reshape(nmodels, len(wind_speeds)).mean(axis=0)
        print('Elapsed time: {}'.format(time.time()-start_time))

        start_time = time.time()
        no_items4 = []
        no_touched4 = []
        no_breached4 = []
        damaged_area4 = []

        # debris model with rotated footprint with intersection
        for i in range(nmodels):

            coverages4 = copy.deepcopy(self.cfg.coverages)
            for _name, item in coverages4.iterrows():
                _coverage = Coverage(coverage_name=_name, **item)
                coverages4.loc[_name, 'coverage'] = _coverage

            debris4 = DebrisTest(self.cfg)
            debris4.rnd_state = np.random.RandomState(i)
            debris4.footprint = (self.footprint_rotated, 0)
            debris4.coverages = coverages4

            for wind_speed in wind_speeds:
                incr_damage = vulnerability_weibull_pdf(x=wind_speed,
                                                        alpha_=alpha_,
                                                        beta_=beta_) * incr_speed

                debris4.no_items_mean = incr_damage
                # print('{},,{}'.format(wind_speed, debris2.no_items_mean))
                debris4.run(wind_speed)
                no_items4.append(debris4.no_items)
                no_touched4.append(debris4.no_impacts)
                breached_area = sum(
                    [x.breached_area for x in debris4.coverages.itervalues()])
                damaged_area4.append(breached_area)

                # only check windows breach
                window_breach = np.array([
                    x.breached for x in debris4.coverages.itervalues()]).sum()
                if window_breach:
                    no_breached4.append(1)
                else:
                    no_breached4.append(0)

        no_items4 = np.array(no_items4).reshape(nmodels, len(wind_speeds)).mean(axis=0)
        no_touched4 = np.array(no_touched4).reshape(nmodels, len(wind_speeds)).mean(axis=0)
        no_breached4 = np.array(no_breached4).reshape(nmodels, len(wind_speeds)).mean(axis=0)
        damaged_area4 = np.array(damaged_area4).reshape(nmodels, len(wind_speeds)).mean(axis=0)
        print('Elapsed time: {}'.format(time.time()-start_time))


        plt.figure()
        # plt.plot(wind_speeds, no_items1, 'b.-',label='stretched')
        # plt.plot(wind_speeds, no_items2, 'r.--', label='alternative')
        # plt.plot(wind_speeds, no_items3, 'g.-', label='rotated')
        # plt.plot(wind_speeds, no_items4, 'c.-', label='rotated_inter')
        plt.plot(wind_speeds, np.cumsum(no_items1), 'b.-',label='cum_original')
        plt.plot(wind_speeds, np.cumsum(no_items2), 'r.--', label='cum_alternative')
        plt.plot(wind_speeds, np.cumsum(no_items3), 'g.--', label='cum_rotated')
        plt.plot(wind_speeds, np.cumsum(no_items4), 'c.--', label='cum_rotated_inter')
        plt.title('Debris impact model comparision')
        plt.xlabel('Wind speed (m/s)')
        plt.ylabel('No. of debris supply')
        plt.legend(loc=4, numpoints=1)
        plt.grid(1)
        plt.pause(1.0)
        # plt.show()
        # plt.savefig('compare_supply1.png')
        plt.close()

        plt.figure()
        plt.plot(wind_speeds, no_breached1, 'b.-',label='original')
        plt.plot(wind_speeds, no_breached2, 'r.--', label='alternative')
        plt.plot(wind_speeds, no_breached3, 'g.-', label='rotated')
        plt.plot(wind_speeds, no_breached4, 'c.-', label='rotated_inter')
        plt.title('Debris impact model comparision')
        plt.xlabel('Wind speed (m/s)')
        plt.ylabel('% of model with window breach')
        plt.legend(loc=4, numpoints=1)
        plt.grid(1)
        plt.pause(1.0)
        # plt.show()
        # plt.savefig('compare_breached1.png')
        plt.close()

        plt.figure()
        # plt.plot(wind_speeds, no_touched1, 'b.-', label='original')
        # plt.plot(wind_speeds, no_touched2, 'r.--', label='alternative')
        # plt.plot(wind_speeds, no_touched3, 'g.-', label='rotated')
        # plt.plot(wind_speeds, no_touched4, 'c.-', label='rotated_inter')
        plt.plot(wind_speeds, np.cumsum(no_touched1), 'b.-',label='cum_original')
        plt.plot(wind_speeds, np.cumsum(no_touched2), 'r.--', label='cum_alternative')
        plt.plot(wind_speeds, np.cumsum(no_touched3), 'g.-', label='cum_rotated')
        plt.plot(wind_speeds, np.cumsum(no_touched4), 'c.-', label='cum_rotated_inter')
        plt.title('Debris impact model comparision')
        plt.xlabel('Wind speed (m/s)')
        plt.ylabel('No of debris impacts')
        plt.legend(loc=4, numpoints=1)
        plt.grid(1)
        plt.pause(1.0)
        # plt.show()
        # plt.savefig('compare_impact1.png')
        plt.close()

        plt.figure()
        plt.plot(wind_speeds, damaged_area1, 'b.-',label='stretched')
        plt.plot(wind_speeds, damaged_area2, 'r.--', label='alternative')
        plt.plot(wind_speeds, damaged_area3, 'g.-', label='rotated')
        plt.plot(wind_speeds, damaged_area4, 'c.-', label='rotated_inter')
        plt.title('Debris impact model comparision')
        plt.xlabel('Wind speed (m/s)')
        plt.ylabel('Damaged area')
        plt.legend(loc=4, numpoints=1)
        plt.grid(1)
        plt.pause(1.0)
        # plt.show()
        # plt.savefig('compare_damaged_area.png')
        plt.close()
'''


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

        cfg_file = os.path.join(cls.path_scenario, 'test_roof_sheeting.cfg')
        cls.cfg = Config(cfg_file=cfg_file)

        cls.rnd_state = np.random.RandomState(1)
        cls.radius = 24.0
        cls.boundary = geometry.Point(0, 0).buffer(cls.radius)

        cls.footprint_inst = geometry.Polygon([(-6.5, 4.0), (6.5, 4.0), (6.5, -4.0),
                                      (-6.5, -4.0), (-6.5, 4.0)])

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
        self.assertEquals(len(sources1), 13)

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
        self.assertEquals(len(sources2), 15)

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
            ax.add_patch(matplotlib.patches.Polygon(_array, alpha=0.3, label='footprint'))

            ax.add_patch(matplotlib.patches.Circle(xy=(0, 0),
                                        radius=self.radius,
                                        alpha=0.3, color='r', label='boundary'))

            x, y = self.ref_footprint[idx].exterior.xy
            ax.plot(x, y, 'b-', label='stretched')
            ax.set_xlim([-40, 20])
            ax.set_ylim([-20, 20])
            plt.title('Wind direction: {}'.format(idx))
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
                '{}_frontal_area_mean'.format(debris_type)]
            _debris._mass = self.cfg.debris_regions[key][
                '{}_mass_mean'.format(debris_type)]
            _debris._flight_time = self.cfg.debris_regions[key][
                '{}_flight_time_mean'.format(debris_type)]

            try:
                self.assertAlmostEqual(expected[debris_type],
                                       _debris.flight_distance, places=2)
            except AssertionError:
                print('{}: {} is expected but {}'.format(
                    debris_type, expected[debris_type], _debris.flight_distance))

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
        plt.title('sampled momentum with mean: {:.2f}, sd: {:.2f}'.format(
            momentum.mean(), momentum.std()))
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
                     label='{}'.format(_str),
                     linestyle='-')

        for _str, _value in flight_distance_poly5.items():
            plt.plot(wind_speeds, _value, color=dic_[_str],
                     label='{} {}'.format(_str, '(Lin and Vanmarcke, 2008)'),
                     linestyle='--')

        plt.title('Flight distance')
        plt.legend(loc=2)
        plt.xlabel('Wind speed (m/s)')
        plt.ylabel('Flight distance (m)')
        # plt.show()
        plt.pause(1.0)
        # plt.savefig('./flight_distance.png', dpi=200)
        plt.close()

    @staticmethod
    def setup_coverages(df_coverages):

        coverages = df_coverages[['wall_name']].copy()
        coverages['breached_area'] = np.zeros_like(coverages.wall_name)

        for _name, item in df_coverages.iterrows():
            _coverage = Coverage(name=_name, **item)
            coverages.loc[_name, 'coverage'] = _coverage

        return coverages

    # @staticmethod
    # def setup_debris(mean_no_items, cfg, rnd_state, wind_speed):
    #
    #     debris_items = []
    #     no_items_by_source = stats.poisson.rvs(mu=mean_no_items,
    #                                            size=len(cfg.debris_sources),
    #                                            random_state=rnd_state)
    #
    #     for no_item, source in zip(no_items_by_source,
    #                                cfg.debris_sources):
    #         _debris_types = rnd_state.choice(DEBRIS_TYPES_KEYS,
    #                                          size=no_item,
    #                                          replace=True,
    #                                          p=cfg.debris_types_ratio)
    #
    #         for debris_type in _debris_types:
    #             _debris = Debris(debris_source=source,
    #                              debris_type=debris_type,
    #                              debris_property=cfg.debris_types[debris_type],
    #                              wind_speed=wind_speed,
    #                              rnd_state=rnd_state)
    #
    #             debris_items.append(_debris)
    #
    #     return debris_items

    def check_performance_of_parmap(self):

        no_debris = 300
        debris_items = []
        source = geometry.Point([20.0, 0.0])

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
            _ = determine_impact_by_debris(debris_item=item,
                                           footprint=self.footprint_inst,
                                           boundary=self.boundary)

        print('Elapsed time in serial: {}'.format(time.time()-tic))

        tic = time.time()
        _ = parmap.map(determine_impact_by_debris, debris_items,
                       self.footprint_inst, self.boundary)
        print('Elapsed time in parallel: {}'.format(time.time()-tic))

    def test_number_of_touched(self):

        house = House(self.cfg, 1)

        incr_speed = self.cfg.speeds[1] - self.cfg.speeds[0]

        key = 'Tropical_town'

        items_all = []
        no_impacts = []
        no_generated = 0
        for speed in self.cfg.speeds:

            damage_incr = vulnerability_weibull_pdf(
                x=speed,
                alpha_=VUL_DIC[key]['alpha'],
                beta_=VUL_DIC[key]['beta']) * incr_speed

            house.damage_incr = damage_incr

            house.set_debris(speed)

            house.debris_items = parmap.map(determine_impact_by_debris,
                                            house.debris_items, house.footprint,
                                            house.boundary)

            no_impacts.append(house.no_debris_impact)
            items_all.append(house.debris_items)
            no_generated += house.no_debris_items

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        # plt.title('Wind direction: 0')

        # p = PolygonPatch(_debris.footprint, fc='red')
        # ax.add_patch(p)

        # ax.add_patch(patches_Polygon(_array, alpha=0.5))

        # shape_type = {'Compact': 'c', 'Sheet': 'g', 'Rod': 'r'}

        for items in items_all:
            for item in items:
                _x, _y = item.trajectory.xy[0][1], item.trajectory.xy[1][1]
                ax.scatter(_x, _y, color='g', alpha=0.2)

        title_str = 'total no. of impacts: {} out of {}'.format(
            sum(no_impacts), no_generated)
        plt.title(title_str)
        ax.set_xlim([-150, 150])
        ax.set_ylim([-100, 100])

        source_x, source_y = [], []
        for source in self.cfg.debris_sources:
            source_x.append(source.x)
            source_y.append(source.y)
        ax.scatter(source_x, source_y, label='source', color='b')
        ax.scatter(0, 0, label='target', color='r')

        # add footprint
        ax.add_patch(matplotlib.patches.Polygon(house.footprint.exterior, fc='red', alpha=0.5))
        ax.add_patch(matplotlib.patches.Polygon(house.boundary.exterior, alpha=0.5))

        plt.show()
        # plt.pause(1.0)
        # plt.close()

if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)

