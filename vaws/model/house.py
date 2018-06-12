import copy
import logging
import numpy as np
import numbers
import collections
from shapely import geometry, affinity
from scipy import stats
import pandas as pd
import parmap

from vaws.model.config import Config, ROTATION_BY_WIND_IDX, DEBRIS_TYPES_KEYS, WIND_DIR
from vaws.model.connection import ConnectionTypeGroup
from vaws.model.zone import Zone
from vaws.model.debris import Debris, determine_impact_by_debris
from vaws.model.coverage import Coverage
from vaws.model.damage_costing import compute_water_ingress_given_damage


class House(object):

    def __init__(self, cfg, seed):

        assert isinstance(cfg, Config)
        assert isinstance(seed, int)

        self.cfg = cfg
        self.seed = seed
        self.rnd_state = np.random.RandomState(self.seed)

        # attributes assigned by self.read_house_data
        self.name = None
        self.width = None
        self.height = None
        self.length = None
        self.replace_cost = None
        self.cpe_cv = None
        self.cpe_str_cv = None
        self.cpe_k = None
        self.cpe_str_k = None
        self.big_a = None
        self.big_b = None
        self.big_a_str = None
        self.big_b_str = None

        # debris related
        self.debris = None
        self._damage_incr = None
        self._windward_coverages = None
        self._windward_coverages_area = None

        # random variables
        self._wind_dir_index = None  # 0 to 7
        self._construction_level = None
        self._profile_index = None
        self._terrain_height_multiplier = None
        self._shielding_multiplier = None

        self.groups = collections.OrderedDict()  # list of conn type groups
        self.connections = {}  # dict of connections with name
        self.zones = {}  # dict of zones with id
        self.coverages = None  # pd.dataframe of coverages
        self.debris_items = None

        # vary over wind speeds
        self.qz = None
        self.cpi = 0.0
        self.collapse = False
        self.repair_cost = 0.0
        self.water_ingress_cost = 0.0
        self.di = None
        self.di_except_water = None

        self.bucket = {}
        self.init_bucket()

        # init house
        self.set_house_data()

        # house is consisting of connections, coverages, and zones
        self.set_coverages()
        self.set_zones()
        self.set_connections()

    @property
    def damage_incr(self):
        return self._damage_incr

    @damage_incr.setter
    def damage_incr(self, value):
        assert isinstance(value, numbers.Number)
        try:
            del self._mean_no_items
        except AttributeError:
            pass
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
        """
        try:
            return self._mean_no_items
        except AttributeError:
            mean_no_items = np.rint(self.cfg.source_items * self._damage_incr)
            self._mean_no_items = mean_no_items
            return mean_no_items

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
            self.cfg.footprint, ROTATION_BY_WIND_IDX[self.wind_dir_index])

    @property
    def front_facing_walls(self):
        return self.cfg.front_facing_walls[WIND_DIR[self.wind_dir_index]]

    @property
    def boundary(self):
        return geometry.Point(0, 0).buffer(self.cfg.boundary_radius)

    @property
    def combination_factor(self):
        """
        AS/NZS 1170.2 action combination factor, Kc
        reduction when wind pressures from more than one building sufrace, e.g.,
        walls and roof
        """
        return 1.0 if abs(self.cpi) < 0.2 else 0.9

    @property
    def mean_factor(self):
        return self.cfg.construction_levels[
            self.construction_level]['mean_factor']

    @property
    def cv_factor(self):
        return self.cfg.construction_levels[
                self.construction_level]['cv_factor']

    @property
    def terrain_height_multiplier(self):
        if self._terrain_height_multiplier is None:
            self._terrain_height_multiplier = np.interp(
                self.height, self.cfg.profile_heights,
                self.cfg.wind_profiles[self.profile_index])
        return self._terrain_height_multiplier

    @property
    def shielding_multiplier(self):
        """
        AS4055 (Wind loads for housing) defines the shielding multiplier
        for full, partial and no shielding as 0.85, 0.95 and 1.0, respectively.

        Based on the JDH report, the following percentages are recommended for
        the shielding of houses well inside Australian urban areas:

            Full shielding: 63%,
            Partial shielding: 15%,
            No shielding: 22%.

        Note that regional shielding factor is less or equal to 0.85, then
        it model buildings are considered to be in Australian urban area.

        """
        if self._shielding_multiplier is None:
            if self.cfg.regional_shielding_factor <= 0.85:  # urban area
                idx = (self.cfg.shielding_multiplier_thresholds <=
                       self.rnd_state.random_integers(0, 100)).sum()
                self._shielding_multiplier = self.cfg.shielding_multipliers[idx][1]
            else:  #
                self._shielding_multiplier = 1.0
        return self._shielding_multiplier

    @property
    def no_debris_items(self):
        """total number of generated debris items"""
        return len(self.debris_items)

    @property
    def no_debris_impact(self):
        """total number of impacted debris items"""
        return sum([x.impact for x in self.debris_items])

    @property
    def debris_momentums(self):
        """list of momentums of generated debris items"""
        return np.array([x.momentum for x in self.debris_items])

    @property
    def windward_coverages(self):
        if self._windward_coverages is None:
            self._windward_coverages = self.coverages.loc[
                self.coverages.direction == 'windward']
        return self._windward_coverages

    @property
    def windward_coverages_area(self):
        if self._windward_coverages_area is None:
            self._windward_coverages_area = sum(
                [x.coverage.area for _, x in self.windward_coverages.iterrows()])
        return self._windward_coverages_area

    @property
    def window_breached_by_debris(self):
        return sum([x.coverage.breached for _, x in self.coverages.iterrows()
                    if x.coverage.description == 'window']) > 0

    @property
    def wind_dir_index(self):
        if self._wind_dir_index is None:
            if self.cfg.wind_dir_index == 8:
                self._wind_dir_index = self.rnd_state.random_integers(0, 7)
            else:
                self._wind_dir_index = self.cfg.wind_dir_index
        return self._wind_dir_index

    @property
    def profile_index(self):
        if self._profile_index is None:
            self._profile_index = self.rnd_state.random_integers(
                1, len(self.cfg.wind_profiles))
        return self._profile_index

    @property
    def construction_level(self):
        """

        Returns: construction_level

        """
        if self._construction_level is None:
            rv = self.rnd_state.random_integers(0, 100)
            key, value, cum_prob = None, None, 0.0
            for key, value in self.cfg.construction_levels.items():
                cum_prob += value['probability'] * 100.0
                if rv <= cum_prob:
                    break
            self._construction_level = key
        return self._construction_level

    def run_simulation(self, wind_speed):

        if not self.collapse:

            logging.info('wind speed {:.3f}'.format(wind_speed))

            # compute load by zone
            self.compute_qz(wind_speed)

            for _, _zone in self.zones.items():
                _zone.calc_zone_pressure(self.cpi, self.qz,
                                         self.combination_factor)

            if self.coverages is not None:
                for _, _ps in self.coverages.iterrows():
                    _ps['coverage'].check_damage(self.qz, self.cpi,
                                                 self.combination_factor,
                                                 wind_speed)

            # check damage by connection type group
            for _, _group in self.groups.items():
                _group.check_damage(wind_speed)

            # change influence / influence patch
            for _, _group in self.groups.items():
                if _group.damage_dist:
                    _group.update_influence(self)

            self.check_house_collapse(wind_speed)

            # cpi is computed here for the next step
            self.check_internal_pressurisation(wind_speed)

            self.compute_damage_index(wind_speed)

            self.fill_bucket()

        return copy.deepcopy(self.bucket)

    def init_bucket(self):

        # house
        for item in ['house', 'debris']:
            self.bucket[item] = {}
            for att in getattr(self.cfg, '{}_bucket'.format(item)):
                self.bucket[item][att] = None

        # components
        for comp in self.cfg.list_components:
            self.bucket[comp] = {}
            for att in getattr(self.cfg, '{}_bucket'.format(comp)):
                self.bucket[comp][att] = {}
                try:
                    for item in getattr(self.cfg, 'list_{}s'.format(comp)):
                        self.bucket[comp][att][item] = None
                except TypeError:
                    pass

    def fill_bucket(self):

        # house
        for att in self.cfg.house_bucket:
            self.bucket['house'][att] = getattr(self, att)

        if self.cfg.flags['debris']:
            for att in self.cfg.debris_bucket:
                self.bucket['debris'][att] = getattr(self.debris, att)

        # components
        for comp in self.cfg.list_components:
            if comp == 'coverage':
                try:
                    for item, value in self.coverages['coverage'].iteritems():
                        for att in self.cfg.coverage_bucket:
                            self.bucket[comp][att][item] = getattr(value, att)
                except TypeError:
                    pass
            else:
                _dic = getattr(self, '{}s'.format(comp))
                for att in getattr(self.cfg, '{}_bucket'.format(comp)):
                    for item, value in _dic.items():
                        self.bucket[comp][att][item] = getattr(value, att)

    def compute_qz(self, wind_speed):
        """
        calculate qz, velocity pressure given wind velocity
        qz = 1/2*rho_air*(V*Mz,cat*Ms)**2 * 1.0e-3  (kN)
        Args:
            wind_speed: wind velocity (m/s)

        Returns:
            qz

        """

        self.qz = 0.5 * self.cfg.rho_air * (
            wind_speed *
            self.terrain_height_multiplier *
            self.shielding_multiplier) ** 2 * 1.0E-3

    def determine_breach(self, row):

        # Complementary CDF of impact momentum
        ccdf = (self.debris_momentums >
                row.coverage.momentum_capacity).sum() / self.no_debris_items
        poisson_rate = (self.no_debris_impact * row.coverage.area /
                        self.windward_coverages_area * ccdf)

        if row.coverage.description == 'window':
            prob_damage = 1.0 - np.exp(-1.0 * poisson_rate)
            rv = self.rnd_state.rand()
            if rv < prob_damage:
                row.coverage.breached_area = row.coverage.area
                row.coverage.breached = 1
        else:
            # assume area: no_impacts * size(1) * amplification_factor(1)
            sampled_impacts = self.rnd_state.poisson(poisson_rate)
            row.coverage.breached_area = min(sampled_impacts, row.coverage.area)

    def check_internal_pressurisation(self, wind_speed):
        """

        Args:
            wind_speed:

        Returns:
            self.cpi
            self.cpi_wind_speed

        """

        if self.cfg.flags['debris'] and wind_speed:

            self.set_debris(wind_speed)

            self.debris_items = parmap.map(determine_impact_by_debris,
                                           self.debris_items, self.footprint,
                                           self.boundary)

            self.windward_coverages.apply(self.determine_breach, axis=1)

                # logging.debug('coverage {} breached by debris b/c {:.3f} < {:.3f} -> area: {:.3f}'.format(
                #     _coverage.name, _capacity, item_momentum, _coverage.breached_area))

        # area of breached coverages
        if self.coverages is not None:
            self.cpi = self.compute_damaged_area_and_assign_cpi()

    def check_house_collapse(self, wind_speed):
        """

        Args:
            wind_speed:

        Returns: collapse of house

        """
        if not self.collapse:

            for _, _group in self.groups.items():

                if 0 < _group.trigger_collapse_at <= _group.prop_damaged:

                    self.collapse = True

                    for _connection in self.connections.itervalues():

                        if _connection.damaged == 0:
                            _connection.damaged = 1
                            _connection.capacity = wind_speed

    def compute_damage_index(self, wind_speed):
        """

        Args:
            wind_speed:

        Returns:
            damage_index: repair cost / replacement cost

        Note:

            1. compute sum of damaged area by group 
            2. revised damage area by group by applying damage factoring
            3. calculate sum of revised damaged area by damage scenario
            4. apply costing modules

        """

        # sum of damaged area by group
        area_by_group, total_area_by_group = self.compute_area_by_group()

        # apply damage factoring
        revised_area_by_group = self.apply_damage_factoring(area_by_group)

        # sum of area by scenario
        prop_area_by_scenario = self.compute_area_by_scenario(
            revised_area_by_group, total_area_by_group)

        _list = []
        for key, value in prop_area_by_scenario.items():
            try:
                tmp = self.cfg.costings[key].compute_cost(value)
            except AssertionError:
                logging.error('{} of {} is invalid'.format(value, key))
            else:
                _list.append(tmp)
        self.repair_cost = np.array(_list).sum()

        # calculate initial envelope repair cost before water ingress is added
        self.di_except_water = min(self.repair_cost / self.replace_cost,
                                   1.0)

        if self.di_except_water < 1.0 and self.cfg.flags['water_ingress']:

            water_ingress_perc = 100.0 * compute_water_ingress_given_damage(
                self.di_except_water, wind_speed,
                self.cfg.water_ingress)

            damage_name = self.determine_scenario_for_water_ingress_costing(
                prop_area_by_scenario)

            self.compute_water_ingress_cost(damage_name, water_ingress_perc)

            _di = (self.repair_cost + self.water_ingress_cost) / self.replace_cost
            self.di = min(_di, 1.0)

        else:
            self.di = self.di_except_water

        logging.info('At {}, repair_cost: {:.3f}, cost by water: {:.3f}, '
                     'di except water:{:.3f}, di: {:.3f}'.format(
            wind_speed, self.repair_cost, self.water_ingress_cost,
            self.di_except_water, self.di))

    def compute_water_ingress_cost(self, damage_name, water_ingress_perc):
        # compute water ingress

        # finding index close to water ingress threshold
        _df = self.cfg.water_ingress_costings[damage_name]
        idx = np.abs(_df.index - water_ingress_perc).argsort()[0]

        self.water_ingress_cost = \
            _df['costing'].values[idx].compute_cost(self.di_except_water)

    def determine_scenario_for_water_ingress_costing(self,
                                                     prop_area_by_scenario):
        # determine damage scenario
        damage_name = 'WI only'  # default
        for _name in self.cfg.damage_order_by_water_ingress:
            try:
                prop_area_by_scenario[_name]
            except KeyError:
                logging.warning(
                    '{} is not defined in the costing'.format(_name))
            else:
                if prop_area_by_scenario[_name]:
                    damage_name = _name
                    break

        return damage_name

    def apply_damage_factoring(self, area_by_group):
        revised = copy.deepcopy(area_by_group)
        for _source, target_list in self.cfg.damage_factorings.items():
            for _target in target_list:
                try:
                    revised[_source] -= area_by_group[_target]
                except KeyError:
                    msg = 'either {} or {} is not found in damage factorings'.format(
                        _source, _target)
                    logging.error(msg)

        return revised

    def compute_area_by_scenario(self, revised_area, total_area_by_group):
        area_by_scenario = collections.defaultdict(int)
        total_area_by_scenario = collections.defaultdict(int)
        for scenario, _list in self.cfg.costing_to_group.items():
            for _group in _list:
                if _group in revised_area:
                    area_by_scenario[scenario] += \
                        max(revised_area[_group], 0.0)
                    total_area_by_scenario[scenario] += total_area_by_group[
                        _group]

        # prop_area_by_scenario
        prop_area_by_scenario = {key: value / total_area_by_scenario[key]
                                 for key, value in area_by_scenario.items()}
        return prop_area_by_scenario

    def compute_area_by_group(self):

        area_by_group = collections.defaultdict(int)
        total_area_by_group = collections.defaultdict(int)

        for _group in self.groups.itervalues():
            area_by_group[_group.name] += _group.damaged_area
            total_area_by_group[_group.name] += _group.costing_area

        # remove group with zero costing area
        for key, value in total_area_by_group.items():
            if value == 0:
                total_area_by_group.pop(key, None)
                area_by_group.pop(key, None)

        # include DEBRIS when debris is ON
        if self.cfg.coverages_area:
            area_by_group['debris'] = sum([x.coverage.breached_area
                                           for _, x in self.coverages.iterrows()])
            total_area_by_group['debris'] = self.cfg.coverages_area

        return area_by_group, total_area_by_group

    def set_house_data(self):

        for key, value in self.cfg.house.items():
            setattr(self, key, value)

    def set_zones(self):

        for _name, item in self.cfg.zones.items():

            new_item = item.copy()
            new_item.update(
                {'wind_dir_index': self.wind_dir_index,
                 'shielding_multiplier': self.shielding_multiplier,
                 'building_spacing': self.cfg.building_spacing,
                 'flag_differential_shielding': self.cfg.flags['differential_shielding'],
                 'cpe_cv': self.cpe_cv,
                 'cpe_k': self.cpe_k,
                 'big_a': self.big_a,
                 'big_b': self.big_b,
                 'cpe_str_cv': self.cpe_str_cv,
                 'cpe_str_k': self.cpe_str_k,
                 'big_a_str': self.big_a_str,
                 'big_b_str': self.big_b_str,
                 'rnd_state': self.rnd_state
                 })

            self.zones[_name] = Zone(name=_name, **new_item)
            # self.zone_by_grid[_zone.grid] = _zone

    def set_connections(self):

        for (_, sub_group_name), connections_by_sub_group in \
                self.cfg.connections.groupby(by=['group_idx', 'sub_group']):

            # sub_group
            group_name = connections_by_sub_group['group_name'].values[0]
            dic_group = copy.deepcopy(self.cfg.groups[group_name])
            dic_group['sub_group'] = sub_group_name

            _group = ConnectionTypeGroup(name=group_name,
                                         **dic_group)
            self.groups[sub_group_name] = _group

            _group.damage_grid = self.cfg.damage_grid_by_sub_group[sub_group_name]
            _group.costing = self.assign_costing(dic_group['damage_scenario'])

            added = pd.DataFrame({'mean_factor': self.mean_factor,
                                  'cv_factor': self.cv_factor,
                                  'rnd_state': self.rnd_state},
                                 index=connections_by_sub_group.index)

            df_connections = connections_by_sub_group.join(added)
            _group.connections = df_connections.to_dict('index')

            # linking with connections
            for connection_name, _connection in _group.connections.items():

                self.connections[connection_name] = _connection

                _connection.influences = self.cfg.influences[_connection.name]

                # influence_patches
                if _connection.name in self.cfg.influence_patches:
                    _connection.influence_patch = \
                        self.cfg.influence_patches[_connection.name]
                else:
                    _connection.influence_patch = {}

                try:
                    _group.damage_grid[_connection.grid] = 0  # intact
                except IndexError:
                    logging.warning(
                        'conn grid {} does not exist within group grid {}'.format(
                            _connection.grid, _group.damage_grid))
                except TypeError:
                    logging.warning(
                        'conn grid does not exist for group {}'.format(_group.name))

                _group.connection_by_grid = _connection.grid, _connection

                self.link_connection_to_influence(_connection)

    def link_connection_to_influence(self, _connection):
        msg = 'unable to associate {conn} with {name} wrt influence'
        # linking connections either zones or connections
        for _, _inf in _connection.influences.items():
            try:
                _inf.source = self.zones[_inf.name]
            except KeyError:
                try:
                    _inf.source = self.connections[_inf.name]
                except KeyError:
                    logging.warning(msg.format(conn=_connection.name,
                                               name=_inf.name))

    def assign_costing(self, damage_scenario):
        """

        Args:
            damage_scenario:

        Returns:

        """
        try:
            return self.cfg.costings[damage_scenario]
        except KeyError:
            pass

    def set_debris(self, wind_speed):

        self.debris_items = []

        no_items_by_source = self.rnd_state.poisson(
            self.mean_no_items, size=len(self.cfg.debris_sources))

        for no_item, source in zip(no_items_by_source, self.cfg.debris_sources):
            _debris_types = self.rnd_state.choice(DEBRIS_TYPES_KEYS,
                                                  size=no_item,
                                                  replace=True,
                                                  p=self.cfg.debris_types_ratio)

            for debris_type in _debris_types:

                _debris = Debris(debris_source=source,
                                 debris_type=debris_type,
                                 debris_property=self.cfg.debris_types[debris_type],
                                 wind_speed=wind_speed,
                                 rnd_state=self.rnd_state)

                self.debris_items.append(_debris)

    def set_coverages(self):

        if self.cfg.coverages is not None:

            df_coverages = self.cfg.coverages.copy()

            df_coverages['direction'] = df_coverages['wall_name'].apply(
                self.assign_windward)

            self.coverages = df_coverages[['direction', 'wall_name']].copy()
            #self.coverages['breached_area'] = np.zeros_like(self.coverages.direction)

            new_item = {
                'wind_dir_index': self.wind_dir_index,
                'cpe_cv': self.cpe_cv,
                'cpe_k': self.cpe_k,
                'big_a': self.big_a,
                'big_b': self.big_b,
                'cpe_str_cv': self.cpe_str_cv,
                'cpe_str_k': self.cpe_str_k,
                'big_a_str': self.big_a_str,
                'big_b_str': self.big_b_str,
                'rnd_state': self.rnd_state
            }

            for _name, item in df_coverages.iterrows():

                for key, value in new_item.items():
                    item[key] = value

                _coverage = Coverage(name=_name, **item)

                self.coverages.loc[_name, 'coverage'] = _coverage

    def assign_windward(self, wall_name):

        windward_dir = WIND_DIR[self.wind_dir_index]
        windward = self.cfg.front_facing_walls[windward_dir]

        leeward = self.cfg.front_facing_walls[WIND_DIR[
            (self.wind_dir_index + 4) % 8]]

        side1, side2 = None, None
        if len(windward_dir) == 1:
            side1 = self.cfg.front_facing_walls[WIND_DIR[
                (self.wind_dir_index + 2) % 8]]
            side2 = self.cfg.front_facing_walls[WIND_DIR[
                (self.wind_dir_index + 6) % 8]]

        # assign windward, leeward, side
        if wall_name in windward:
            return 'windward'
        elif wall_name in leeward:
            return 'leeward'
        elif wall_name in side1:
            return 'side1'
        elif wall_name in side2:
            return 'side2'

    def compute_damaged_area_and_assign_cpi(self):

        # self.coverages['breached_area'] = \
        #     self.coverages['coverage'].apply(lambda x: x.breached_area)

        # self.debris.damaged_area = self.coverages['breached_area'].sum()

        # breached_area_by_wall = \
        #     self.coverages.groupby('direction').apply(
        #         lambda x: x.coverage.breached_area, axis=1).sum()

        breached_area_by_wall = self.coverages.groupby('direction').apply(
            lambda x: x.coverage).apply(lambda x: x.breached_area).sum(
            level='direction')

        # check if opening is dominant or non-dominant
        max_breached = breached_area_by_wall[breached_area_by_wall ==
                                             breached_area_by_wall.max()]

        # cpi
        if len(max_breached) == 1:  # dominant opening

            area_else = breached_area_by_wall.sum() - max_breached.iloc[0]
            if area_else:
                ratio = max_breached.iloc[0] / area_else
                row = (self.cfg.dominant_opening_ratio_thresholds <= ratio).sum()
            else:
                row = 4

            direction = max_breached.index[0]

            cpi = self.cfg.cpi_table_for_dominant_opening[row][direction]

            if row > 1:  # factor with Cpe

                # cpe where the largest opening
                breached_area = self.coverages.loc[
                    self.coverages['direction'] == direction].apply(
                    lambda x: x.coverage.breached_area, axis=1)

                max_area = breached_area[breached_area == breached_area.max()]
                cpe_array = np.array([self.coverages.loc[i, 'coverage'].cpe
                                     for i in max_area.keys()])
                max_cpe = max(cpe_array.min(), cpe_array.max(), key=abs)
                cpi *= max_cpe

            logging.debug('cpi for bldg with dominant opening: {}'.format(cpi))

        elif len(max_breached) == len(breached_area_by_wall):  # all equal openings
            if max_breached.iloc[0]:
                cpi = -0.3
            else:  # in case no opening
                cpi = 0.0

            logging.debug('cpi for bldg without dominant opening: {}'.format(cpi))

        else:
            if breached_area_by_wall['windward']:
                cpi = 0.2
            else:
                cpi = -0.3

            logging.debug('cpi for bldg without dominant opening: {}'.format(cpi))

        return cpi
