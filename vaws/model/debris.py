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
from scipy import stats

from vaws.model.config import G_CONST, RHO_AIR
from vaws.model.stats import sample_lognormal

FLIGHT_DISTANCE_COEFF = {2: {'Compact': [0.011, 0.2060],
                             'Sheet': [0.3456, 0.072],
                             'Rod': [0.2376, 0.0723]},
                         5: {'Compact': [0.405, -0.036, -0.052, 0.008],
                             'Sheet': [0.456, -0.148, 0.024, -0.0014],
                             'Rod': [0.4005, -0.16, 0.036, -0.0032]}}

FLIGHT_DISTANCE_POWER = {2: [1, 2],
                         5: [2, 3, 4, 5]}


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
            x = stats.norm.rvs(loc=0,
                               scale=self.flight_distance / 3.0,
                               random_state=self.rnd_state)
            y = stats.norm.rvs(loc=0,
                               scale=self.flight_distance / 12.0,
                               random_state=self.rnd_state)

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
                k_star = RHO_AIR * math.pow(self.wind_speed, 2.0) / (
                    2.0 * G_CONST * self.mass / self.frontal_area)
                kt_star = k_star * t_star

                kt_star_powered = np.array(
                    [math.pow(kt_star, i) for i in
                     FLIGHT_DISTANCE_POWER[self.flag_poly]])
                coeff = np.array(FLIGHT_DISTANCE_COEFF[self.flag_poly][self.type])
                less_dis = (coeff * kt_star_powered).sum()

                # dimensionless hor. displacement
                # k*x_star = k*g*x/V**2
                # x = (k*x_star)*(V**2)/(k*g)
                convert_to_dim = math.pow(self.wind_speed, 2.0) / (
                    k_star * G_CONST)
                self._flight_distance = convert_to_dim * less_dis

        return self._flight_distance

    def determine_impact(self, footprint, boundary):
        land_within_footprint = footprint.contains(self.landing)
        intersect_within_boundary = (self.trajectory.intersects(footprint)
                                     and boundary.contains(self.landing))
        if land_within_footprint or intersect_within_boundary:
            self.impact = 1
        else:
            self.impact = 0

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


def determine_impact_by_debris(debris_item, footprint, boundary):
    """

    :param debris_item:
    :param footprint:
    :param boundary:
    :return:
    """

    debris_item.determine_impact(footprint=footprint, boundary=boundary)

    return debris_item


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

'''
class Debris(object):

    def __init__(self, cfg, seed, wind_dir_idx, coverages):

        assert isinstance(cfg, Config)
        assert isinstance(seed, int)
        assert wind_dir_idx in range(8)
        assert isinstance(coverages, dict)

        self.cfg = cfg
        self.seed = seed
        self.rnd_state = np.random.RandomState(seed)
        self.wind_dir_idx = wind_dir_idx
        self.coverages = coverages

        self._damage_incr = None

        self.items = None  # container of items over wind steps
        self.momentums = None  # container of momentums in a wind step

        # vary over wind speeds
        self.no_items = 0  # total number of generated debris items
        self.no_impacts = 0  # total number of impacted debris items
        self.damaged_area = 0.0  # total damaged area by debris items




    @property
    def area(self):
        return sum([x.area for _, x in self.coverages.items()])

    def run(self, wind_speed):
        """

        Args:
            wind_speed:

        Returns:

        """

        # sample a poisson for each source
        no_items_by_source = stats.poisson.rvs(mu=self.mean_no_items,
                                               size=len(self.cfg.debris_sources),
                                               random_state=self.seed)

        self.no_items = no_items_by_source.sum()
        logging.debug('no_item_by_source at speed {:.3f}: {} sampled with {}'.format(
            wind_speed, self.no_items, self.mean_no_items))

        debris_type_per_source = self.generate_debris_type_per_source(
            no_items_by_source)

        out_tuples = parmap.map(generate_debris_item_per_source,
                                debris_type_per_source, self.cfg.debris_types,
                                self.footprint, self.boundary, wind_speed,
                                self.seed)

        _df = pd.DataFrame.from_records(out_tuples,
                                        columns=['debris_type', 'trajectory',
                                                 'flag_impact', 'momentum'])

        self.no_impacts = _df['flag_impact'].sum()
        self.momentums = _df['momentum'].values
        self.items = _df[['debris_type', 'trajectory']].values.tolist()

        if self.no_impacts:
            self.check_debris_impact()

    def generate_debris_type_per_source(self, no_items_by_source):
        debris_type_per_source = []
        for no_item, source in zip(no_items_by_source, self.cfg.debris_sources):
            _debris_types = self.rnd_state.choice(self.cfg.debris_types_keys,
                                                  size=no_item, replace=True,
                                                  p=self.cfg.debris_types_ratio)
            for _type in _debris_types:
                debris_type_per_source.append((source, _type))

        return debris_type_per_source

    def check_debris_impact(self):
        """

        Args:
            frontal_area:
            item_momentum:

        Returns:
            self.breached

        """

        for _id, _coverage in self.coverages.items():

            _capacity = sample_lognormal(*_coverage.log_failure_momentum,
                                         rnd_state=self.seed)

            # Complementary CDF of impact momentum
            ccdf = (self.momentums > _capacity).sum()/self.no_items
            poisson_rate = self.no_impacts * _coverage.area / self.area * ccdf

            if _coverage.description == 'window':
                prob_damage = 1.0 - math.exp(-1.0*poisson_rate)
                if self.rnd_state.rand() < prob_damage:
                    _coverage.breached_area = _coverage.area
                    _coverage.breached = 1
            else:
                # assume area: no_impacts * size(1) * amplification_factor(1)
                sampled_impacts = self.rnd_state.poisson(poisson_rate)
                _coverage.breached_area = min(sampled_impacts, _coverage.area)

            # logging.debug('coverage {} breached by debris b/c {:.3f} < {:.3f} -> area: {:.3f}'.format(
            #     _coverage.name, _capacity, item_momentum, _coverage.breached_area))

'''




'''
def generate_debris_item(debris_item, footprint, boundary, wind_speed, rnd_state):
    """

    :param source_and_debris_type:
    :param debris_types:
    :param footprint:
    :param boundary:
    :param wind_speed:
    :param rnd_state: None, integer or np.random.RandomState
    :return:
    """

    assert isinstance(source_and_debris_type, tuple)
    source, debris_type = source_and_debris_type

    debris_property = debris_types[debris_type].copy()

    for item in ['mass', 'frontal_area', 'flight_time']:

        debris_property[item] = sample_lognormal(
            mu_lnx=debris_property['{}_mu'.format(item)],
            std_lnx=debris_property['{}_std'.format(item)],
            rnd_state=rnd_state)

    flight_distance = compute_flight_distance(debris_type=debris_type,
                                              debris_property=debris_property,
                                              wind_speed=wind_speed)
    debris_property['flight_distance']

    # logging.debug('debris type:{}, area:{}, time:{}, distance:{}'.format(
    #    debris_type_str, frontal_area, flight_time, flight_distance))

    # determine landing location for a debris item
    # sigma_x, y are taken from Wehner et al. (2010)
    x = stats.norm.rvs(loc=0, scale=flight_distance / 3.0, random_state=rnd_state)
    y = stats.norm.rvs(loc=0, scale=flight_distance / 12.0, random_state=rnd_state)

    # cov_matrix = [[pow(sigma_x, 2.0), 0.0], [0.0, pow(sigma_y, 2.0)]]
    # x, y = self.rnd_state.multivariate_normal(mean=[0.0, 0.0],
    #                                           cov=cov_matrix)
    # reference point: target house
    landing = geometry.Point(x + source.x - flight_distance,
                             y + source.y)
    trajectory = geometry.LineString([source, landing])

    debris_momentum = compute_debris_momentum(debris_property=debris_property,
                                              wind_speed=wind_speed,
                                              rnd_state=rnd_state)

    flag_impact = 0
    if footprint.contains(landing) or (trajectory.intersects(footprint)
                                       and boundary.contains(landing)):
        flag_impact = 1

    return debris_type, trajectory, flag_impact, debris_momentum
'''


"""
mean_no_items = 0.5

no_items_by_source = stats.poisson.rvs(mu=mean_no_items,
                                       size=len(cfg.debris_sources),
                                       random_state=rnd_state)

debris_items = []
for no_item, source in zip(no_items_by_source, cfg.debris_sources):
    _debris_types = rnd_state.choice(DEBRIS_TYPES_KEYS,
                                          size=no_item,
                                          replace=True,
                                          p=cfg.debris_types_ratio)

    for debris_type in _debris_types:
        _debris = Debris(debris_source=source,
                         debris_type=debris_type,
                         debris_property=cfg.debris_types[debris_type],
                         wind_speed=wind_speed,
                         rnd_state=rnd_state)

        debris_items.append(_debris)


class Dummy(object):

    def __init__(self, x):
        self.x = x

    @property
    def y(self):
        return self.x + 1

    @property
    def y2(self, value=4):
        return self.y + value
"""