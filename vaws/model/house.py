import copy
import logging
import numpy as np
import collections

from vaws.model.config import Config
from vaws.model.connection import ConnectionTypeGroup
from vaws.model.zone import Zone
from vaws.model.debris import Debris
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
        self.cpe_cov = None
        self.cpe_str_cov = None
        self.cpe_k = None
        self.cpe_str_k = None
        self.big_a = None
        self.big_b = None
        self.big_a_str = None
        self.big_b_str = None

        # debris related
        self.debris = None

        # random variables
        self.wind_dir_index = None  # 0 to 7
        self.construction_level = None
        self.profile_index = None
        self.terrain_height_multiplier = None
        self.shielding_multiplier = None

        self.groups = collections.OrderedDict()  # list of conn type groups
        self.connections = {}  # dict of connections with name
        self.zones = {}  # dict of zones with id
        self.coverages = None  # pd.dataframe of coverages

        # vary over wind speeds
        self.qz = None
        self.cpi = 0.0
        self.collapse = False
        self.window_breached_by_debris = False
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
        self.set_debris()

    def run_simulation(self, wind_speed):

        if not self.collapse:

            logging.info('wind speed {:.3f}'.format(wind_speed))

            # compute load by zone
            self.compute_qz(wind_speed)

            # load = qz * (Cpe + Cpi) * A + dead_load
            for _zone in self.zones.itervalues():
                _zone.calc_zone_pressure(self.cpi, self.qz)

            if self.coverages is not None:
                for _, _ps in self.coverages.iterrows():
                    _ps['coverage'].check_damage(self.qz, self.cpi, wind_speed)

            for _, _connection in self.connections.items():
                _connection.compute_load()

            # check damage by connection type group
            for _, _group in self.groups.items():
                _group.check_damage(wind_speed)
                _group.compute_damaged_area()

                # change influence / influence patch
                if _group.flag_damaged and self.cfg.flags.get('damage_distribute_{}'.format(
                        _group.name)):
                    _group.update_influence(self)

            self.check_house_collapse(wind_speed)

            # cpi is computed here for the next step
            self.check_internal_pressurisation(wind_speed)

            if self.coverages is not None:
                self.coverages['breached_area'] = \
                    self.coverages['coverage'].apply(
                        lambda x: x.breached_area)

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

    def check_internal_pressurisation(self, wind_speed):
        """

        Args:
            wind_speed:

        Returns:
            self.cpi
            self.cpi_wind_speed

        """

        if self.cfg.flags['debris']:
            self.debris.run(wind_speed)

        # logging.debug('no_items_mean: {}, no_items:{}'.format(
        #     self.house.debris.no_items_mean,
        #     self.house.debris.no_items))

        # area of breached coverages
        if self.coverages is not None:
            self.cpi = self.compute_damaged_area_and_assign_cpi()

            if self.cfg.flags['debris']:
                window_breach = np.array([x.breached for x in self.debris.coverages.itervalues()]).sum()
                if window_breach:
                    self.window_breached_by_debris = True

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

        # print('{}'.format(area_by_scenario))
        # print('{}'.format(total_area_by_scenario))

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
            area_by_group['debris'] = self.coverages['breached_area'].sum()
            total_area_by_group['debris'] = self.cfg.coverages_area

        return area_by_group, total_area_by_group

    @property
    def mean_factor(self):
        return self.cfg.construction_levels[
            self.construction_level]['mean_factor']

    @property
    def cov_factor(self):
        return self.cfg.construction_levels[
                self.construction_level]['cov_factor']

    def set_terrain_height_multiplier(self):
        self.terrain_height_multiplier = np.interp(
            self.height, self.cfg.profile_heights,
            self.cfg.wind_profiles[self.profile_index])

    def set_shielding_multiplier(self):
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
        if self.cfg.regional_shielding_factor <= 0.85:  # urban area
            idx = (self.cfg.shielding_multiplier_thresholds <=
                   self.rnd_state.random_integers(0, 100)).sum()
            self.shielding_multiplier = self.cfg.shielding_multipliers[idx][1]
        else:  #
            self.shielding_multiplier = 1.0

    def set_wind_dir_index(self):
        if self.cfg.wind_dir_index == 8:
            self.wind_dir_index = self.rnd_state.random_integers(0, 7)
        else:
            self.wind_dir_index = self.cfg.wind_dir_index

    def set_profile_index(self):
        self.profile_index = self.rnd_state.random_integers(
            1, len(self.cfg.wind_profiles))

    def set_construction_level(self):
        """

        Returns: construction_level, mean_factor, cov_factor

        """
        rv = self.rnd_state.random_integers(0, 100)
        key, value, cum_prob = None, None, 0.0
        for key, value in self.cfg.construction_levels.items():
            cum_prob += value['probability'] * 100.0
            if rv <= cum_prob:
                break
        self.construction_level = key

    def set_house_data(self):

        for key, value in self.cfg.house.items():
            setattr(self, key, value)

        self.set_wind_dir_index()
        self.set_construction_level()
        self.set_profile_index()
        self.set_terrain_height_multiplier()
        self.set_shielding_multiplier()

    def set_zones(self):

        for _name, item in self.cfg.zones.items():

            item.update(
                {'wind_dir_index': self.wind_dir_index,
                 'shielding_multiplier': self.shielding_multiplier,
                 'building_spacing': self.cfg.building_spacing,
                 'flag_differential_shielding': self.cfg.flags['differential_shielding']})

            _zone = Zone(name=_name, **item)

            _zone.sample_cpe(
                cpe_cov=self.cpe_cov,
                cpe_k=self.cpe_k,
                big_a=self.big_a,
                big_b=self.big_b,
                cpe_str_cov=self.cpe_str_cov,
                cpe_str_k=self.cpe_str_k,
                big_a_str=self.big_a_str,
                big_b_str=self.big_b_str,
                rnd_state=self.rnd_state)

            self.zones[_name] = _zone
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
            _group.connections = connections_by_sub_group.to_dict('index')

            # linking with connections
            for connection_name, _connection in _group.connections.items():

                self.connections[connection_name] = _connection
                self.set_connection_property(_connection)

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
        # linking connections either zones or connections
        for _, _inf in _connection.influences.items():
            try:
                _inf.source = self.zones[_inf.name]
            except KeyError:
                try:
                    _inf.source = self.connections[_inf.name]
                except KeyError:
                    logging.warning('unable to associate {} with {} wrt influence'.format(
                        _connection.name, _inf.name))

    def set_connection_property(self, _connection):
        """

        Args:
            _connection: instance of Connection class

        Returns:

        """
        _connection.sample_strength(mean_factor=self.mean_factor,
                                    cov_factor=self.cov_factor,
                                    rnd_state=self.rnd_state)
        _connection.sample_dead_load(rnd_state=self.rnd_state)

        _connection.influences = self.cfg.influences[_connection.name]

        # influence_patches
        if _connection.name in self.cfg.influence_patches:
            _connection.influence_patch = \
                self.cfg.influence_patches[_connection.name]
        else:
            _connection.influence_patch = {}

    def assign_costing(self, key):
        """

        Args:
            key:

        Returns:

        """

        if key in self.cfg.costings:
            return self.cfg.costings[key]
        else:
            logging.warning('group {} undefined in cfg.costings'.format(key))

    def set_debris(self):

        coverages_debris = {}
        if self.coverages is not None and not self.coverages.empty:
            coverages_debris = self.coverages.loc[
                self.coverages.direction == 'windward', 'coverage'].to_dict()

        self.debris = Debris(cfg=self.cfg,
                             wind_dir_idx=self.wind_dir_index,
                             rnd_state=self.rnd_state,
                             coverages=coverages_debris)

    def set_coverages(self):

        if self.cfg.coverages is not None:

            df_coverages = self.cfg.coverages.copy()

            df_coverages['direction'] = df_coverages['wall_name'].apply(
                self.assign_windward)

            self.coverages = df_coverages[['direction', 'wall_name']].copy()
            self.coverages['breached_area'] = np.zeros_like(self.coverages.direction)

            for _name, item in df_coverages.iterrows():

                item['wind_dir_index'] = self.wind_dir_index

                _coverage = Coverage(name=_name, **item)

                _coverage.sample_cpe(
                    cpe_cov=self.cpe_cov,
                    cpe_k=self.cpe_k,
                    big_a=self.big_a,
                    big_b=self.big_b,
                    cpe_str_cov=self.cpe_str_cov,
                    cpe_str_k=self.cpe_str_k,
                    big_a_str=self.big_a_str,
                    big_b_str=self.big_b_str,
                    rnd_state=self.rnd_state)

                _coverage.sample_strength(rnd_state=self.rnd_state)

                self.coverages.loc[_name, 'coverage'] = _coverage

    def assign_windward(self, wall_name):

        windward_dir = self.cfg.wind_dir[self.wind_dir_index]
        windward = self.cfg.front_facing_walls[windward_dir]

        leeward = self.cfg.front_facing_walls[self.cfg.wind_dir[
            (self.wind_dir_index + 4) % 8]]

        side1, side2 = None, None
        if len(windward_dir) == 1:
            side1 = self.cfg.front_facing_walls[self.cfg.wind_dir[
                (self.wind_dir_index + 2) % 8]]
            side2 = self.cfg.front_facing_walls[self.cfg.wind_dir[
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

        breached_area_by_wall = \
            self.coverages.groupby('direction')['breached_area'].sum()

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
                    self.coverages['direction'] == direction, 'breached_area']
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