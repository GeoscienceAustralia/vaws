import copy
import logging
import numpy as np
# import numbers
import collections
from shapely import geometry, affinity
import pandas as pd
from distributed.worker import logger

from vaws.model.constants import (WIND_DIR, ROTATION_BY_WIND_IDX,
                                  SHIELDING_MULTIPLIERS_KEYS,
                                  SHIELDING_MULTIPLIERS_PROB,
                                  SHIELDING_MULTIPLIERS,
                                  DOMINANT_OPENING_RATIO_THRESHOLDS,
                                  CPI_TABLE_FOR_DOMINANT_OPENING, RHO_AIR)
from vaws.model.config import Config
from vaws.model.connection import ConnectionTypeGroup
from vaws.model.zone import Zone
from vaws.model.debris import generate_debris_items
from vaws.model.coverage import Coverage
from vaws.model.damage_costing import compute_water_ingress_given_damage


def run_simulation(house, wind_speed, ispeed, damage_increment, prop_water_ingress, cfg):

    logger.debug(f'wind speed {wind_speed:.3f}')

    house.damage_increment = damage_increment

    house.prop_water_ingress = prop_water_ingress

    if not house.collapse:


        # compute load by zone
        house.compute_qz(wind_speed)

        for _, zone in house.zones.items():
            zone.calc_zone_pressure(cpi=house.cpi,
                                    qz=house.qz,
                                    combination_factor=house.combination_factor)

        if house.coverages is not None:
            for _, coverage in house.coverages['coverage'].iteritems():
                coverage.check_damage(qz=house.qz,
                                      cpi=house.cpi,
                                      combination_factor=house.combination_factor,
                                      wind_speed=wind_speed)

        # check damage by connection type group
        for _, group in house.groups.items():
            group.check_damage(wind_speed=wind_speed)

        # update connection damage status
        for _, connection in house.connections.items():
            connection.damaged_previous = connection.damaged

        # change influence / influence patch
        for _, group in house.groups.items():
            if group.damage_dist:
                group.update_influence(house_inst=house)

        house.check_house_collapse(wind_speed=wind_speed)

        # cpi is computed here for the next step
        house.run_debris_and_update_cpi(wind_speed, cfg)

        house.compute_damage_index(wind_speed, ispeed, cfg)

    else:
        # still run debris
        house.run_debris_and_update_cpi(wind_speed)

    # save results
    house.fill_bucket(cfg)

    return copy.deepcopy(house.bucket)






class House(object):

    def __init__(self, cfg, seed):

        assert isinstance(cfg, Config)
        assert isinstance(seed, int)

        #logger = logger or logging.getLogger(__name__)
        #self.seed = seed
        self.rnd_state = np.random.RandomState(seed)

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
        # self._cv_factor = None
        # self._mean_factor = None
        self._total_area_by_group = None

        # debris related
        #self._footprint = None
        #self._damage_increment = None  # compute mean_no_debris_items
        self._debris_coverages = None
        self._debris_coverages_area = None
        self._debris_coverages_area_ratio = None
        self._front_facing_walls = None

        # random variables

        self.groups = collections.OrderedDict()  # list of conn type groups
        self.connections = {}  # dict of connections with name
        self.zones = {}  # dict of zones with id
        self.coverages = None  # pd.dataframe of coverages
        self.debris_items = None

        # vary over wind speeds
        self.qz = None
        self.collapse = False
        self.repair_cost = 0.0
        self.water_ingress_cost = 0.0
        self.water_ingress_perc = 0.0
        self.di = None
        self.di_except_water = None
        self._prop_water_ingress = None  # compute mean_no_debris_items

        self.bucket = {}

        # init house
        self.init_bucket(cfg)
        self.set_wind_dir_index(cfg)
        self.set_walls(cfg)
        self.set_terrain_height_multiplier(cfg)
        self.set_shielding_multiplier(cfg)
        self.set_house_data(cfg)
        self.set_coverages(cfg)
        self.set_zones(cfg)
        self.set_connections(cfg)
        self.set_total_area_by_group(cfg)
        self.set_footprint(cfg)


    def set_total_area_by_group(self, cfg):

        self.total_area_by_group = {}
        for key, value in cfg.groups.items():
            if value['costing_area'] > 0:
                self.total_area_by_group[key] = value['costing_area']

        if cfg.coverages_area:
            self.total_area_by_group['debris'] = cfg.coverages_area

        if cfg.wall_collapse:
            self.total_area_by_group['wall'] = cfg.coverages_area

    def set_walls(self, cfg):
        self.windward_walls = cfg.front_facing_walls[WIND_DIR[self.wind_dir_index]]

        self.leeward_walls = cfg.front_facing_walls[WIND_DIR[(self.wind_dir_index + 4) % 8]]

        if len(WIND_DIR[self.wind_dir_index]) == 1:
            self.side1_walls = cfg.front_facing_walls[WIND_DIR[(self.wind_dir_index + 2) % 8]]
            self.side2_walls = cfg.front_facing_walls[WIND_DIR[(self.wind_dir_index + 6) % 8]]
        else:
            self.side1_walls = None
            self.side2_walls = None

    def set_mean_no_debris_items(self, cfg):
        """
        dN = f * dD
        where
            dN: incr. number of debris items,
            dD: incr in vulnerability (or damage index)
            f : a constant factor

        if we use dD/dV (pdf of vulnerability), then dN = f * (dD/dV) * dV

        :return:
        """
        try:
            mean_no_debris_items = np.rint(cfg.source_items *
                                           self.damage_increment)
        except TypeError:
            pass
        else:
            self.mean_no_debris_items = mean_no_debris_items

    def set_footprint(self, cfg):
        """
        create house footprint by wind direction
        Note that debris source model is generated assuming wind blows from East.

        :return:
            self.footprint
        """
        self.footprint = affinity.rotate(cfg.footprint, ROTATION_BY_WIND_IDX[self.wind_dir_index])

    def set_front_facing_walls(self, cfg):
        self.front_facing_walls = cfg.front_facing_walls[WIND_DIR[self.wind_dir_index]]

    @property
    def combination_factor(self):
        """
        AS/NZS 1170.2 action combination factor, Kc
        reduction when wind pressures from more than one building surface, e.g.,
        walls and roof
        """
        return 1.0 if abs(self.cpi) < 0.2 else 0.9

    def set_terrain_height_multiplier(self, cfg):
        self.set_profile_index(cfg)
        self.terrain_height_multiplier = np.interp(
                cfg.house['height'], cfg.profile_heights,
                cfg.wind_profiles[self.profile_index])

    def set_shielding_multiplier(self, cfg):
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
        if cfg.regional_shielding_factor <= 0.85:  # urban area
            key = self.rnd_state.choice(SHIELDING_MULTIPLIERS_KEYS,
                                        p=SHIELDING_MULTIPLIERS_PROB)
            self.shielding_multiplier = SHIELDING_MULTIPLIERS[key]
        else:  #
            self.shielding_multiplier = 1.0

    @property
    def no_debris_items(self):
        """total number of generated debris items"""
        try:
            return len(self.debris_items)
        except TypeError:
            return None

    @property
    def no_debris_impacts(self):
        """total number of impacted debris items"""
        try:
            return sum([x.impact for x in self.debris_items])
        except TypeError:
            return None

    @property
    def debris_momentums(self):
        """list of momentums of generated debris items"""
        try:
            return np.array([x.momentum for x in self.debris_items])
        except TypeError:
            return None

    @property
    def debris_coverages(self):
        if self.coverages is not None and self._debris_coverages is None:
            self._debris_coverages = self.coverages.loc[
                self.coverages.direction == 'windward', 'coverage'].tolist()
        return self._debris_coverages

    @property
    def debris_coverages_area(self):
        if self.debris_coverages and self._debris_coverages_area is None:
            self._debris_coverages_area = sum([x.area for x in self.debris_coverages])
        return self._debris_coverages_area

    @property
    def debris_coverages_area_ratio(self):
        if self.debris_coverages and self._debris_coverages_area_ratio is None:
            self._debris_coverages_area_ratio = [
                x.area/self.debris_coverages_area for x in self.debris_coverages]
        return self._debris_coverages_area_ratio

    @property
    def window_breached(self):
        if self.coverages is not None:
            return sum([x.coverage.breached for _, x in self.coverages.iterrows()
                        if x.coverage.description == 'window']) > 0
        else:
            return None

    @property
    def breached_area(self):
        if self.coverages is not None:
            return sum([x.coverage.breached_area for _, x in
                        self.coverages.iterrows()])
        else:
            return None

    def set_wind_dir_index(self, cfg):
        if cfg.wind_dir_index == 8:
            self.wind_dir_index = self.rnd_state.randint(0, 7 + 1)
        else:
            self.wind_dir_index = cfg.wind_dir_index

    def set_profile_index(self, cfg):
        self.profile_index = self.rnd_state.randint(1, len(cfg.wind_profiles) + 1)

    @property
    def cpi(self):
        try:
            return self._cpi
        except AttributeError:

            cpi = 0.0

            if self.coverages is not None:

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
                        row = (DOMINANT_OPENING_RATIO_THRESHOLDS <= ratio).sum()
                    else:
                        row = 4

                    direction = max_breached.index[0]
                    cpi = CPI_TABLE_FOR_DOMINANT_OPENING[row][direction]

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

                    logger.debug(f'cpi for bldg with dominant opening: {cpi}')

                elif len(max_breached) == len(breached_area_by_wall):  # all equal openings
                    if max_breached.iloc[0]:
                        cpi = -0.3
                    else:  # in case no opening
                        cpi = 0.0

                    logger.debug(f'cpi for bldg without dominant opening: {cpi}')

                else:
                    if breached_area_by_wall['windward']:
                        cpi = 0.2
                    else:
                        cpi = -0.3

                    logger.debug(f'cpi for bldg without dominant opening: {cpi}')

            self._cpi = cpi

            return cpi

    @property
    def prop_water_ingress(self):
        return self._prop_water_ingress

    @prop_water_ingress.setter
    def prop_water_ingress(self, value):
        # assert isinstance(value, numbers.Number)
        self._prop_water_ingress = value

    def init_bucket(self, cfg):

        # house
        self.bucket['house'] = {}
        for att, _ in cfg.house_bucket:
            self.bucket['house'][att] = None

        # components
        for comp in cfg.list_components:
            self.bucket[comp] = {}
            if comp == 'debris':
                for att, _ in getattr(cfg, f'{comp}_bucket'):
                    self.bucket[comp][att] = None
            else:
                for att, _ in getattr(cfg, f'{comp}_bucket'):
                    self.bucket[comp][att] = {}
                    try:
                        for item in getattr(cfg, f'list_{comp}s'):
                            self.bucket[comp][att][item] = None
                    except TypeError:
                        pass

    def fill_bucket(self, cfg):

        # house
        for att, _ in cfg.house_bucket:
            self.bucket['house'][att] = getattr(self, att)

        # components
        for comp in cfg.list_components:
            if comp == 'coverage':  # pd.DataFrame
                try:
                    for item, value in self.coverages['coverage'].iteritems():
                        for att, _ in cfg.coverage_bucket:
                            self.bucket[comp][att][item] = getattr(value, att)
                except TypeError:
                    pass
            elif comp == 'group':
                pass
            elif comp == 'debris':
                if cfg.flags['debris']:
                    for att, _ in cfg.debris_bucket:
                        self.bucket[comp][att] = [getattr(x, att)
                                                  for x in self.debris_items]
            else:  # dictionary
                dic = getattr(self, f'{comp}s')
                for att, _ in getattr(cfg, f'{comp}_bucket'):
                    for item, value in dic.items():
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

        self.qz = 0.5 * RHO_AIR * (
            wind_speed *
            self.terrain_height_multiplier *
            self.shielding_multiplier) ** 2 * 1.0E-3

    def run_debris_and_update_cpi(self, wind_speed, cfg):
        """

        Args:
            wind_speed:

        Returns:
            self.cpi
            self.cpi_wind_speed

        """
        # update cpi
        try:
            del self._cpi
        except AttributeError:
            pass

        if cfg.flags['debris'] and wind_speed:

            self.set_mean_no_debris_items(cfg)
            self.debris_items = generate_debris_items(cfg=cfg,
                                                      mean_no_debris_items=self.mean_no_debris_items,
                                                      wind_speed=wind_speed,
                                                      rnd_state=self.rnd_state)

            logger.debug(f'no debris items: {len(self.debris_items)}')
            for item in self.debris_items:
                item.check_impact(footprint=self.footprint,
                                  boundary=cfg.impact_boundary)
                item.check_coverages(coverages=self.debris_coverages,
                                     prob_coverages=self.debris_coverages_area_ratio)

    def check_house_collapse(self, wind_speed):
        """

        Args:
            wind_speed:

        Returns: collapse of house

        """
        if not self.collapse:

            for _, group in self.groups.items():

                if 0 < group.trigger_collapse_at <= group.prop_damaged:

                    self.collapse = True

                    for _, connection in self.connections.items():

                        if connection.damaged == 0:
                            connection.damaged = 1
                            connection.capacity = wind_speed

    def compute_damage_index(self, wind_speed, ispeed, cfg):
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

        msg = 'At {speed}, cost: {cost:.3f}, cost by water: {water:.3f}, ' \
              'di except water:{di_except_water:.3f}, di: {di:.3f}'

        # sum of damaged area by group
        area_by_group, prop_by_group = self.compute_area_by_group(cfg)

        # apply damage factoring
        revised_area_by_group = self.apply_damage_factoring(area_by_group, cfg)

        # assign damaged area by group
        for key, value in revised_area_by_group.items():
            self.bucket['group']['damaged_area'][key] = value
            if key not in ['debris', 'wall']:
                self.bucket['group']['prop_damaged'][key] = prop_by_group[key] / cfg.groups[key]['no_connections']

        # sum of area by scenario
        prop_area_by_scenario = self.compute_area_by_scenario(revised_area_by_group, cfg)

        self.repair_cost_by_scenario = {}
        for key, value in prop_area_by_scenario.items():
            try:
                cost = cfg.costings[key].compute_cost(value)
            except AssertionError:
                logger.error(f'{value} of {key} is invalid')
            else:
                self.repair_cost_by_scenario[key] = cost
        self.repair_cost = sum(self.repair_cost_by_scenario.values())

        # calculate initial envelope repair cost before water ingress is added
        self.di_except_water = min(self.repair_cost / self.replace_cost,
                                   1.0)

        if self.di_except_water < 1.0 and cfg.flags['water_ingress']:

            # applying prob_water_ingress
            target_prob = cfg.water_ingress_ref[ispeed] - self.prop_water_ingress
            if (self.water_ingress_perc > 0) or (self.rnd_state.uniform() < target_prob) or (
                self.di_except_water >= cfg.water_ingress_di_threshold_wi):
                self.water_ingress_perc = 100.0 * compute_water_ingress_given_damage(
                     self.di_except_water, wind_speed, cfg.water_ingress)
            else:
                self.water_ingress_perc = 0.0

            damage_name = self.determine_scenario_for_water_ingress_costing(
                prop_area_by_scenario, cfg)

            self.compute_water_ingress_cost(damage_name, cfg)

            _di = (self.repair_cost + self.water_ingress_cost) / self.replace_cost
            self.di = min(_di, 1.0)

        else:
            self.water_ingress_perc = 100.0
            self.di = self.di_except_water

        logger.debug(msg.format(speed=wind_speed,
                                     cost=self.repair_cost,
                                     water=self.water_ingress_cost,
                                     di_except_water=self.di_except_water,
                                     di=self.di))

    def compute_water_ingress_cost(self, damage_name, cfg):
        # compute water ingress

        df = cfg.water_ingress_costings[damage_name].copy()
        # before: finding index close to water ingress threshold
        # idx = np.abs(df.index - water_ingress_perc).argsort()[0]
        #self.water_ingress_cost = \
        #    df['costing'].values[idx].compute_cost(water_ingress_perc/100)

        # after: interpolation
        # compute cost
        df['cost'] = df.apply(lambda x: x['costing'].compute_cost(self.water_ingress_perc/100), axis=1)
        # interpolation
        self.water_ingress_cost = np.interp(self.water_ingress_perc, df.index, df['cost'])

    def determine_scenario_for_water_ingress_costing(self,
                                                     prop_area_by_scenario, cfg):
        # determine damage scenario
        damage_name = 'WI only'  # default
        for name in cfg.damage_order_by_water_ingress:
            try:
                prop_area_by_scenario[name]
            except KeyError:
                pass
                #logger.warning(
                #    f'{name} is not defined in the costing')
            else:
                if prop_area_by_scenario[name]:
                    damage_name = name
                    break

        return damage_name

    def apply_damage_factoring(self, area_by_group, cfg):
        msg = 'either {source} or {target} is not found in damage factorings'
        revised = copy.deepcopy(area_by_group)
        for source, target_list in cfg.damage_factorings.items():
            for target in target_list:
                try:
                    revised[source] -= area_by_group[target]
                except KeyError:
                    logger.error(msg.format(source=source, target=target))

        return revised

    def compute_area_by_scenario(self, revised_area, cfg):
        area_by_scenario = collections.defaultdict(int)
        total_area_by_scenario = collections.defaultdict(int)
        for scenario, _list in cfg.costing_to_group.items():
            for group in _list:
                if group in revised_area:
                    area_by_scenario[scenario] += max(revised_area[group], 0.0)
                    total_area_by_scenario[scenario] += self.total_area_by_group[group]

        # prop_area_by_scenario
        prop_area_by_scenario = {key: value / total_area_by_scenario[key]
                                 for key, value in area_by_scenario.items()}
        return prop_area_by_scenario

    def compute_area_by_group(self, cfg):

        area_by_group = collections.defaultdict(int)
        prop_by_group = collections.defaultdict(int)

        for _, group in self.groups.items():
            prop_by_group[group.name] += group.prop_damaged * group.no_connections
            area_by_group[group.name] += group.damaged_area

        # include DEBRIS when debris is ON
        if cfg.coverages_area:
            area_by_group['debris'] = self.breached_area

        if cfg.flags['wall_collapse']:
            # first compute % of roof to wall connections
            no_damaged = sum([self.connections[k].damaged
                for k in cfg.wall_collapse['connections']])
            roof_loss = no_damaged / cfg.wall_collapse['no'] * 100
            wall_loss = np.interp(roof_loss, cfg.wall_collapse['roof_damage'],
                                  cfg.wall_collapse['wall_damage']) / 100
            wall_loss *= self.total_area_by_group['wall']
            area_by_group['wall'] = max(wall_loss - self.breached_area, 0)

            logger.debug(f'roof_loss: {roof_loss:.1f}%, wall_loss: {wall_loss:.3f}, breached_area: {self.breached_area}, damaged_wall_area: {area_by_group["wall"]:.3f}')

        return area_by_group, prop_by_group

    def set_house_data(self, cfg):

        for key, value in cfg.house.items():
            setattr(self, key, value)

    def set_zones(self, cfg):

        for name, item in cfg.zones.items():

            new_item = item.copy()
            new_item.update(
                {'wind_dir_index': self.wind_dir_index,
                 'shielding_multiplier': self.shielding_multiplier,
                 'building_spacing': cfg.building_spacing,
                 'flag_differential_shielding': cfg.flags['differential_shielding'],
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

            self.zones[name] = Zone(name=name, **new_item)
            # self.zone_by_grid[_zone.grid] = _zone

    def set_connections(self, cfg):

        for (_, sub_group_name), connections_by_sub_group in \
                cfg.connections.groupby(by=['group_idx', 'sub_group']):

            # sub_group
            group_name = connections_by_sub_group['group_name'].values[0]
            dic_group = copy.deepcopy(cfg.groups[group_name])
            dic_group['sub_group'] = sub_group_name

            group = ConnectionTypeGroup(name=group_name, **dic_group)
            self.groups[sub_group_name] = group

            group.damage_grid = cfg.damage_grid_by_sub_group[sub_group_name]
            group.costing = self.assign_costing(dic_group['damage_scenario'], cfg)

            added = pd.DataFrame({'rnd_state': self.rnd_state},
                                 index=connections_by_sub_group.index)
            #
            # added = pd.DataFrame({'mean_factor': self.mean_factor,
            #                       'cv_factor': self.cv_factor,
            #                       'rnd_state': self.rnd_state},
            #                      index=connections_by_sub_group.index)

            df_connections = connections_by_sub_group.join(added)
            group.connections = df_connections.to_dict('index')

            # linking with connections
            for connection_name, connection in group.connections.items():

                self.connections[connection_name] = connection

                connection.influences = cfg.influences[connection.name]

                # influence_patches
                if connection.name in cfg.influence_patches:
                    connection.influence_patch = cfg.influence_patches[connection.name]
                else:
                    connection.influence_patch = {}

                try:
                    group.damage_grid[connection.grid] = 0  # intact
                except IndexError:
                    logger.warning(
                        f'conn grid {connection.grid} does not exist within group grid {group.damage_grid}')
                except TypeError:
                    logger.warning(
                        f'conn grid does not exist for group {group.name}')

                group.connection_by_grid = connection.grid, connection

                self.link_connection_to_influence(connection)

    def link_connection_to_influence(self, connection):
        # linking connections either zones or connections
        for _, inf in connection.influences.items():
            try:
                inf.source = self.zones[inf.name]
            except KeyError:
                try:
                    inf.source = self.connections[inf.name]
                except KeyError:
                    msg = f'unable to associate {connection.name} with {inf.name} wrt influence'
                    logger.warning(msg)

    def assign_costing(self, damage_scenario, cfg):
        """

        Args:
            damage_scenario:

        Returns:

        """
        try:
            return cfg.costings[damage_scenario]
        except KeyError:
            pass

    def set_coverages(self, cfg):

        if cfg.coverages is not None:

            df_coverages = cfg.coverages.copy()

            df_coverages['direction'] = df_coverages['wall_name'].apply(
                self.assign_windward)

            self.coverages = df_coverages[['direction', 'wall_name']].copy()

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

            for name, item in df_coverages.iterrows():

                for key, value in new_item.items():
                    item[key] = value

                coverage = Coverage(name=name, **item)

                self.coverages.loc[name, 'coverage'] = coverage

    def assign_windward(self, wall_name):
        """

        :param wall_name:
        :return:
        """

        # assign windward, leeward, side
        if wall_name in self.windward_walls:
            return 'windward'
        elif wall_name in self.leeward_walls:
            return 'leeward'
        elif wall_name in self.side1_walls:
            return 'side1'
        elif wall_name in self.side2_walls:
            return 'side2'
