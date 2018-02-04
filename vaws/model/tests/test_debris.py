import unittest
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as patches_Polygon
import logging

from numbers import Number
import math
from shapely.geometry import Point, Polygon, LineString
from shapely.affinity import rotate

from vaws.model.config import Config
from vaws.model.debris import Debris
from vaws.model.coverage import Coverage
from vaws.model.curve import vulnerability_weibull_pdf


def get_damaged_area(coverages, debris, footprint_inst, wind_speeds, vul_params):

    debris.footprint = (footprint_inst, 0)
    rnd_state = np.random.RandomState(1)
    debris.rnd_state = rnd_state
    debris.coverages = coverages

    incr_speed = wind_speeds[1] - wind_speeds[0]

    alpha_ = vul_params['alpha']
    beta_ = vul_params['beta']

    damaged_area = []
    no_touched = []
    no_items = []
    sampled_impacts = []

    for wind_speed in wind_speeds:

        incr_damage = vulnerability_weibull_pdf(x=wind_speed,
                                                alpha_=alpha_,
                                                beta_=beta_) * incr_speed
        debris.no_items_mean = incr_damage

        debris.run(wind_speed)

        breached_area = np.sum(
            [x.breached_area for x in debris.coverages.itervalues()])

        damaged_area.append(breached_area)
        no_touched.append(debris.no_touched)
        no_items.append(debris.no_items)
        sampled_impacts.append(debris.sampled_impacts)

    return damaged_area, no_touched, no_items, sampled_impacts


class DebrisAlt(object):

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
        self.debris_momentums = []

        # vary over wind speeds
        self.no_items = 0
        self.sampled_impacts = 0
        self.no_touched = None
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
            value (float) :

        Returns:
            set no_items_mean
        """
        assert isinstance(value, Number)

        self._no_items_mean = np.rint(self.cfg.source_items * value)

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

        self.coverage_prob = np.array(self.coverage_prob)/self.area

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

        assert isinstance(value, np.random.RandomState)
        self._rnd_state = value

    def run(self, wind_speed):
        """

        Args:
            wind_speed:

        Returns:

        """

        self.no_touched = 0
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

            list_debris = self.rnd_state.choice(self.cfg.debris_types_keys,
                                                size=no_item, replace=True,
                                                p=self.cfg.debris_types_ratio)

            # logging.debug('source: {}, list_debris: {}'.format(source, list_debris))

            for debris_type in list_debris:
                self.generate_debris_item(wind_speed, source, debris_type)

        if self.no_touched:
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
        cov_matrix = [[math.pow(sigma_x, 2.0), 0.0], [0.0, math.pow(sigma_y, 2.0)]]

        x, y = self.rnd_state.multivariate_normal(mean=[0.0, 0.0],
                                                  cov=cov_matrix)
        # reference point: target house
        pt_debris = Point(x + source.x - flight_distance, y + source.y)
        line_debris = LineString([source, pt_debris])

        self.debris_items.append((debris_type_str, line_debris))

        item_momentum = self.compute_debris_momentum(debris['cdav'],
                                                     frontal_area,
                                                     flight_distance,
                                                     mass,
                                                     wind_speed)

        self.debris_momentums.append(item_momentum)

        if line_debris.intersects(self.footprint):
            self.no_touched += 1

        # logging.debug('debris impact from {}, {}'.format(
        #     pt_debris.x, pt_debris.y))

    def check_debris_impact(self):
        """

        Returns:
            self.breached

        """

        for _id, _coverage in self.coverages.iteritems():

            try:
                _capacity = self.rnd_state.lognormal(*_coverage.log_failure_momentum)
            except ValueError:  # when sigma = 0
                _capacity = math.exp(_coverage.log_failure_momentum[0])

            # Complementary CDF of impact momentum
            ccdf = 1.0*(self.debris_momentums > np.array(_capacity)).sum()/self.no_items
            poisson_rate = self.no_touched * _coverage.area / self.area * ccdf

            if _coverage.description == 'window':
                prob_damage = 1.0 - math.exp(-1.0*poisson_rate)
                rv = self.rnd_state.rand()
                if rv < prob_damage:
                    _coverage.breached_area = _coverage.area
                    _coverage.breached = 1

                self.sampled_impacts += 1
                # print('1')
            else:
                # assume area: no_impacts * size(1) * amplification_factor(1)
                sampled_impacts = self.rnd_state.poisson(poisson_rate)
                self.sampled_impacts += sampled_impacts
                _coverage.breached_area = min(sampled_impacts, _coverage.area)
                # print('{}:{}'.format(sampled_impacts, self.no_touched))

            # logging.debug('coverage {} breached by debris b/c {:.3f} < {:.3f} -> area: {:.3f}'.format(
            #     _coverage.name, _capacity, item_momentum, _coverage.breached_area))

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
            convert_to_dim = pow(wind_speed, 2.0) / (
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

'''
    def run_alt(self, wind_speed):
        """ Returns several results as data members
        """

        # self.no_items_mean = self.compute_number_of_debris_items(incr_damage)

        self.damaged_area = 0.0
        self.no_touched = 0

        # sample a poisson for each source
        no_items_by_source = self.rnd_state.poisson(
            self.no_items_mean, size=len(self.cfg.debris_sources))

        self.no_items = sum(no_items_by_source)
        logging.debug('no_item_by_source at speed {:.3f}: {}'.format(
            wind_speed, self.no_items))

        # loop through sources
        for no_item, source in zip(no_items_by_source, self.cfg.debris_sources):

            list_debris = self.rnd_state.choice(self.cfg.debris_types.keys(),
                                                size=no_item, replace=True)

            # logging.debug('source: {}, list_debris: {}'.format(source, list_debris))

            for _debris_type in list_debris:
                self.generate_debris_item_alt(wind_speed, source, _debris_type)

def generate_debris_item_alt(debris, wind_speed, source, rnd_state):
    """

    Args:
        wind_speed:
        source:
        debris_type_str:

    Returns:

    """


        mass = rnd_state.lognormal(debris['mass_mu'], debris['mass_std'])

        frontal_area = rnd_state.lognormal(debris['frontal_area_mu'],
                                                debris['frontal_area_std'])

        flight_time = rnd_state.lognormal(self.cfg.flight_time_log_mu,
                                               self.cfg.flight_time_log_std)

        flight_distance = compute_flight_distance(debris_type_str,
                                                   flight_time,
                                                   frontal_area,
                                                   mass,
                                                   wind_speed)

        # determine landing location for a debris item
        # sigma_x, y are taken from Wehner et al. (2010)
        sigma_x = flight_distance / 3.0
        sigma_y = flight_distance / 12.0
        cov_matrix = [[pow(sigma_x, 2.0), 0.0], [0.0, pow(sigma_y, 2.0)]]

        x, y = rnd_state.multivariate_normal(mean=[0.0, 0.0],
                                                  cov=cov_matrix)
        # reference point: target house
        pt_debris = Point(x + source.x - flight_distance, y + source.y)
        line_debris = LineString([source, pt_debris])

        if (footprint.contains(pt_debris) or footprint.touches(pt_debris)):

            no_touched += 1

            item_momentum = self.compute_debris_mementum(debris['cdav'],
                                                     frontal_area,
                                                     flight_distance,
                                                     mass,
                                                     wind_speed,
                                                     rnd_state)

            # determine coverage type
            _rv = self.rnd_state.uniform()
            _id = self.coverages[
                self.coverages['cum_prop_area'] > _rv].index[0]
            _coverage = self.coverages.loc[_id]

            # check whether it impacts or not using failure momentum
            _capacity = self.rnd_state.lognormal(
                *_coverage['log_failure_momentum'])

            if _capacity < item_momentum:

                logging.debug(
                    'coverage type:{}, capacity:{}, demand:{}'.format(
                        _coverage['description'], _capacity, item_momentum))

                if _coverage['description'] == 'window':
                    self.breached = True
                    self.damaged_area += _coverage['area']

                else:
                    self.damaged_area += min(frontal_area,
                                             _coverage['area'])
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

        # staggered source
        sources2 = Debris.create_sources(self.cfg.debris_radius,
                                         self.cfg.debris_angle,
                                         self.cfg.building_spacing,
                                         True)

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

    def test_footprint04(self):

        _debris = Debris(cfg=self.cfg)
        _debris.footprint = (self.footprint_inst, 0)

        fig = plt.figure(1)
        ax = fig.add_subplot(111)

        _array = np.array(_debris.footprint.exterior.xy).T

        ax.add_patch(patches_Polygon(_array, alpha=0.3))
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

        _array = np.array(_debris.footprint.exterior.xy).T

        ax.add_patch(patches_Polygon(_array, alpha=0.3))

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

        _array = np.array(_debris.footprint.exterior.xy).T

        ax.add_patch(patches_Polygon(_array, alpha=0.3))

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

        _array = np.array(_debris.footprint.exterior.xy).T

        ax.add_patch(patches_Polygon(_array, alpha=0.3))

        x, y = self.ref_footprint[3].exterior.xy
        ax.plot(x, y, 'b-')
        ax.set_xlim([-40, 20])
        ax.set_ylim([-20, 20])
        plt.title('Wind direction: 3')
        # plt.show()
        plt.pause(1.0)
        plt.close()

    def test_footprint_non_rect(self):
        # Not working yet 

        footprint_inst = Polygon(
            [(-6.5, 4.0), (6.5, 4.0), (6.5, 0.0), (0.0, 0.0),
             (0.0, -4.0), (-6.5, -4.0), (-6.5, 4.0)])

        _debris = Debris(cfg=self.cfg)
        _debris.footprint = (footprint_inst, 0)

        fig = plt.figure(1)
        ax = fig.add_subplot(111)

        _array = np.array(_debris.footprint.exterior.xy).T

        ax.add_patch(patches_Polygon(_array, alpha=0.3))

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
    
    def test_compute_debris_momentum(self):

        debris = Debris(cfg=self.cfg)

        wind_speeds = np.arange(0.01, 120.0, 1.0)
        # flight_time = math.exp(self.cfg.flight_time_log_mu)
        momentum = {}
        debris.rnd_state = np.random.RandomState(1)

        for key, value in self.cfg.debris_types.iteritems():

            momentum[key] = np.zeros_like(wind_speeds)

            frontal_area = math.exp(value['frontal_area_mu'])
            mass = math.exp(value['mass_mu'])
            flight_time = math.exp(self.cfg.flight_time_log_mu)
            cdav = value['cdav']

            for i, wind_speed in enumerate(wind_speeds):

                flight_distance = debris.compute_flight_distance(
                    key, flight_time, frontal_area, mass, wind_speed)

                momentum[key][i] = debris.compute_debris_momentum(
                    cdav, frontal_area, flight_distance, mass, wind_speed)

        dic_ = {'Compact': 'b', 'Sheet': 'r', 'Rod': 'g'}
        plt.figure()
        for _str, _value in momentum.iteritems():
            plt.plot(wind_speeds, _value, color=dic_[_str],
                     label=_str, linestyle='-')

        plt.title('momentum')
        plt.legend(loc=2)
        # plt.show()
        plt.pause(1.0)
        plt.close()

    def test_compute_flight_distance(self):

        debris = Debris(cfg=self.cfg)

        wind_speeds = np.arange(0.0, 120.0, 1.0)
        flight_time = math.exp(self.cfg.flight_time_log_mu)
        flight_distance = {}
        flight_distance_poly5 = {}

        for key, value in self.cfg.debris_types.iteritems():

            frontal_area = math.exp(value['frontal_area_mu'])
            mass = math.exp(value['mass_mu'])

            flight_distance[key] = np.zeros_like(wind_speeds)
            flight_distance_poly5[key] = np.zeros_like(wind_speeds)

            for i, wind_speed in enumerate(wind_speeds):
                flight_distance[key][i] = \
                    debris.compute_flight_distance(
                        key, flight_time, frontal_area, mass, wind_speed)

                flight_distance_poly5[key][i] = \
                    debris.compute_flight_distance(
                        key, flight_time, frontal_area, mass, wind_speed, flag_poly=5)

        dic_ = {'Compact': 'b', 'Sheet': 'r', 'Rod': 'g'}
        plt.figure()
        for _str, _value in flight_distance.iteritems():
            plt.plot(wind_speeds, _value, color=dic_[_str],
                     label='{}'.format(_str),
                     linestyle='-')

        for _str, _value in flight_distance_poly5.iteritems():
            plt.plot(wind_speeds, _value, color=dic_[_str],
                     label='{} {}'.format(_str, '(Lin and Vanmarcke, 2008)'),
                     linestyle='--')

        plt.title('Flight distance')
        plt.legend(loc=2)
        plt.xlabel('Wind speed (m/s)')
        plt.xlabel('Flight distance (m)')
        # plt.show()
        plt.pause(1.0)
        # plt.savefig('./flight_distance.png', dpi=200)
        plt.close()

    def test_run(self):

        # set up logging
        # file_logger = os.path.join(self.path_output, 'log_debris.txt')
        # logging.basicConfig(filename=file_logger,
        #                     filemode='w',
        #                     level=logging.DEBUG,
        #                     format='%(levelname)s %(message)s')

        wind_speeds = np.arange(0.0, 120.0, 1.0)
        vul_dic = {'Capital_city': {'alpha': 0.1586, 'beta': 3.8909},
                   'Tropical_town': {'alpha': 0.1030, 'beta': 4.1825}}

        key = 'Capital_city'
        #key = 'Tropical_town'

        coverages1 = copy.deepcopy(self.cfg.coverages)
        # change all types to window
        # coverages1['description'].replace('door', 'window')
        # coverages1['description'].replace('weatherboard', 'window')
        for _name, item in coverages1.iterrows():
            _coverage = Coverage(coverage_name=_name, **item)
            coverages1.loc[_name, 'coverage'] = _coverage

        debris1 = Debris(self.cfg)
        #debris1.cfg.source_items = 100
        damaged_area1, no_touched1, no_items1, sampled_impacts1 = get_damaged_area(coverages=coverages1,
                                         debris=debris1,
                                         footprint_inst=self.footprint_inst,
                                         wind_speeds=wind_speeds,
                                         vul_params=vul_dic[key])

        coverages2 = copy.deepcopy(self.cfg.coverages)
        # coverages2['description'].replace('door', 'window')
        # coverages2['description'].replace('weatherboard', 'window')
        for _name, item in coverages2.iterrows():
            _coverage = Coverage(coverage_name=_name, **item)
            coverages2.loc[_name, 'coverage'] = _coverage
        debris2 = DebrisAlt(self.cfg)
        #debris1.cfg.source_items = 100

        damaged_area2, no_touched2, no_items2, sampled_impacts2 = get_damaged_area(coverages=coverages2,
                                         debris=debris2,
                                         footprint_inst=self.footprint_inst,
                                         wind_speeds=wind_speeds,
                                         vul_params=vul_dic[key])

        assert debris1.area == debris2.area

        plt.figure()
        plt.plot(wind_speeds, np.array(damaged_area2) / debris2.area * 100.0, 'b.-',label='original')
        plt.plot(wind_speeds, np.array(damaged_area1) / debris1.area * 100.0, 'r.--', label='alternative')
        plt.title('Debris impact model comparision')
        plt.xlabel('Wind speed (m/s)')
        plt.ylabel('Damaged area (%)')
        plt.legend(loc=2, numpoints=1)
        # plt.pause(1.0)
        # plt.show()
        plt.savefig('compare.png')
        plt.close()

        plt.figure()
        plt.plot(wind_speeds, no_items2, 'b.-',label='original')
        plt.plot(wind_speeds, no_items1, 'r.--', label='alternative')
        plt.title('Debris impact model comparision')
        plt.xlabel('Wind speed (m/s)')
        plt.ylabel('No. of debris supply')
        plt.legend(loc=2, numpoints=1)
        # plt.pause(1.0)
        # plt.show()
        plt.savefig('compare_supply.png')
        plt.close()

        plt.figure()
        plt.plot(wind_speeds, sampled_impacts2, 'b.-', label='original_sampled')
        plt.plot(wind_speeds, no_touched1, 'r.--', label='alternative')
        plt.plot(wind_speeds, no_touched2, 'g.-', label='original')
        plt.title('Debris impact model comparision')
        plt.xlabel('Wind speed (m/s)')
        plt.ylabel('No of debris impacts')
        plt.legend(loc=2, numpoints=1)
        # plt.pause(1.0)
        # plt.show()
        plt.savefig('compare_impact.png')
        plt.close()

    '''
    def test_check_debris_impact_option1(self):

        _coverages = self.cfg.coverages.copy()

        for _name, item in _coverages.iterrows():

            _coverage = Coverage(coverage_name=_name, **item)
            _coverages.loc[_name, 'coverage'] = _coverage

        _debris = Debris(cfg=self.cfg)
        _debris.footprint = (self.footprint_inst, 0)
        rnd_state = np.random.RandomState(1)
        _debris.rnd_state = rnd_state
        _debris.coverages = _coverages

        # for each source
        source = self.cfg.debris_sources[0]
        wind_speed = 60.0
        no_item = 20

        list_debris = rnd_state.choice(self.cfg.debris_types_keys,
                                       size=no_item, replace=True,
                                       p=self.cfg.debris_types_ratio)

        footprint = _debris.footprint
        coverage = _debris.coverages[1]
        debris_momentum = []
        list_breached = []
        for debris_type in list_debris:
            item_momentum, breached = generate_debris_item_option1(
                self.cfg, rnd_state, wind_speed, source, debris_type, footprint,
                coverage)
            debris_momentum.append(item_momentum)
            list_breached.append(breached)

        print('{}'.format(debris_momentum))
        print('{}'.format(list_breached))

        # option2
        # Complementary CDF of impact momentum
        no_touched = (list_breached >= np.array([0])).sum()
        try:
            _capacity = rnd_state.lognormal(*coverage.log_failure_momentum)
        except ValueError:  # when sigma = 0
            _capacity = math.exp(coverage.log_failure_momentum[0])

        ccdf = 1.0 * (debris_momentum > np.array(_capacity)).sum() / len(
            debris_momentum)
        poisson_rate = no_touched * coverage.area / _debris.area * ccdf

        # if _coverage.description == 'window':
        prob_damage = 1.0 - math.exp(-1.0 * poisson_rate)
        print('{}'.format(prob_damage))
    '''
    '''
    def test_number_of_touched_org(self):

        no_items = []
        no_items_mean = []
        no_touched = []

        _debris = Debris(cfg=self.cfg)
        _stretched_poly = Polygon([(-24.0, 6.5), (4.0, 6.5), (4.0, -6.5),
                                   (-24.0, -6.5), (-24.0, 6.5)])

        _debris.footprint = (_stretched_poly, 2)  # no rotation

        rnd_state = np.random.RandomState(1)
        _debris.rnd_state = rnd_state

        incr_speed = self.cfg.speeds[1] - self.cfg.speeds[0]

        for speed in self.cfg.speeds:

            incr_damage = vulnerability_weibull(x=speed,
                                                alpha_=0.10304,
                                                beta_=4.18252,
                                                flag='pdf') * incr_speed

            _debris.no_items_mean = incr_damage

            _debris.run_alt(speed)
            no_items.append(_debris.no_items)
            no_touched.append(_debris.no_touched)
            no_items_mean.append(_debris.no_items_mean)

        fig = plt.figure(1)
        ax = fig.add_subplot(111)

        ax.set_xlim([-150, 150])
        ax.set_ylim([-100, 100])
        plt.title('Wind direction: 0')

        for _target in _debris.debris_items:
            x, y = _target.xy
            ax.plot(x, y, linestyle='-', color='c', alpha=0.1)

        p = PolygonPatch(_debris.footprint, fc='red')
        ax.add_patch(p)
        x, y = _debris.footprint.exterior.xy
        ax.plot(x, y, 'k-')

        for item in self.cfg.debris_sources:
            ax.plot(item.x, item.y, 'ko')

        title_str = 'org: no_items_mean:{}, no_items:{}, no_touched:{}'.format(
            sum(no_items_mean), sum(no_items), sum(no_touched))
        plt.title(title_str)
        # plt.show()
        plt.pause(1.0)
        plt.close()
    '''

    def test_number_of_touched_revised(self):

        no_items = []
        no_items_mean = []
        no_touched = []

        self.cfg.source_items = 100

        _coverages = self.cfg.coverages.copy()

        for _name, item in _coverages.iterrows():

            _coverage = Coverage(coverage_name=_name, **item)
            _coverages.loc[_name, 'coverage'] = _coverage

        _debris = Debris(cfg=self.cfg)
        _debris.footprint = (self.footprint_inst, 0)

        rnd_state = np.random.RandomState(1)
        _debris.rnd_state = rnd_state
        _debris.coverages = _coverages

        incr_speed = self.cfg.speeds[1] - self.cfg.speeds[0]

        for speed in self.cfg.speeds:

            incr_damage = vulnerability_weibull_pdf(x=speed,
                                                    alpha_=0.10304,
                                                    beta_=4.18252) * incr_speed

            _debris.no_items_mean = incr_damage

            _debris.run(speed)
            no_items.append(_debris.no_items)
            no_touched.append(_debris.no_touched)
            no_items_mean.append(_debris.no_items_mean)

        fig = plt.figure(1)
        ax = fig.add_subplot(111)

        ax.set_xlim([-150, 150])
        ax.set_ylim([-100, 100])
        plt.title('Wind direction: 0')

        for _, _target in _debris.debris_items:
            x, y = _target.xy
            ax.plot(x, y, linestyle='-', color='c', alpha=0.1)

        _array = np.array(_debris.footprint.exterior.xy).T

        ax.add_patch(patches_Polygon(_array, alpha=0.3))

        x, y = _debris.footprint.exterior.xy
        ax.plot(x, y, 'k-')

        for item in self.cfg.debris_sources:
            ax.plot(item.x, item.y, 'ko')

        title_str = 'no_items_mean:{}, no_items:{}, no_touched:{}'.format(
            sum(no_items_mean), sum(no_items), sum(no_touched))
        plt.title(title_str)

        # plt.show()
        plt.pause(1.0)
        plt.close()

if __name__ == '__main__':
    unittest.main()

