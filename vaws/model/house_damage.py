import copy
import logging
from numpy import array, abs, argsort
from numpy.random import RandomState
from collections import defaultdict

from vaws.model.house import House
from vaws.model.config import Config
from vaws.model.damage_costing import compute_water_ingress_given_damage


class HouseDamage(object):

    def __init__(self, cfg, seed):

        assert isinstance(cfg, Config)
        assert isinstance(seed, int)

        self.cfg = cfg
        self.seed = seed
        self.rnd_state = RandomState(self.seed)

        self.house = House(cfg, self.rnd_state)

        # vary over wind speeds
        self.qz = None
        self.ms = None
        self.cpi = 0.0
        self.collapse = False
        self.breached = False
        self.repair_cost = 0.0
        self.water_ingress_cost = 0.0
        self.di = None
        self.di_except_water = None

        self.bucket = {}
        self.init_bucket()

    def run_simulation(self, wind_speed):

        if not self.collapse:

            logging.info('wind speed {:.3f}'.format(wind_speed))

            # compute load by zone
            self.compute_qz_ms(wind_speed)

            # load = qz * (Cpe + Cpi) * A + dead_load
            for _zone in self.house.zones.itervalues():
                _zone.calc_zone_pressure(self.house.wind_orientation,
                                         self.cpi,
                                         self.qz,
                                         self.ms,
                                         self.cfg.building_spacing,
                                         self.cfg.flags['diff_shielding'])

            if self.house.coverages is not None:
                for _, _ps in self.house.coverages.iterrows():
                    _ps['coverage'].check_damage(self.qz, self.cpi, wind_speed)

            for _connection in self.house.connections.itervalues():
                _connection.compute_load()

            # check damage by connection type group
            for _group in self.house.groups.itervalues():

                _group.check_damage(wind_speed)
                _group.compute_damaged_area()

                # change influence / influence patch
                if _group.damaged and self.cfg.flags.get('damage_distribute_{}'.format(
                        _group.name)):
                    _group.update_influence(self.house)

            self.check_house_collapse(wind_speed)
            self.compute_damage_index(wind_speed)

            # cpi is computed here for the next step
            self.check_internal_pressurisation(wind_speed)

            self.fill_bucket()

        return copy.deepcopy(self.bucket)

    def init_bucket(self):

        # house
        for item in ['house', 'house_damage', 'debris']:
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
            self.bucket['house'][att] = getattr(self.house, att)

        for att in self.cfg.house_damage_bucket:
            self.bucket['house_damage'][att] = getattr(self, att)

        if self.cfg.flags['debris']:
            for att in self.cfg.debris_bucket:
                self.bucket['debris'][att] = getattr(self.house.debris, att)

        # components
        for comp in self.cfg.list_components:
            if comp == 'coverage':
                try:
                    for item, value in self.house.coverages['coverage'].iteritems():
                        for att in self.cfg.coverage_bucket:
                            self.bucket[comp][att][item] = getattr(value, att)
                except TypeError:
                    pass
            else:
                _dic = getattr(self.house, '{}s'.format(comp))
                for att in getattr(self.cfg, '{}_bucket'.format(comp)):
                    for item, value in _dic.iteritems():
                        self.bucket[comp][att][item] = getattr(value, att)

    def compute_qz_ms(self, wind_speed):
        """
        calculate qz, velocity pressure given wind velocity
        qz = 0.6*10-3*(Mz,cat*V)**2
        Args:
            wind_speed: wind velocity (m/s)

        Returns:
            qz
            update ms

        """
        if self.cfg.regional_shielding_factor <= 0.85:
            thresholds = array([63, 63 + 15])
            ms_dic = {0: 1.0, 1: 0.85, 2: 0.95}
            idx = (thresholds <= self.rnd_state.random_integers(0, 100)).sum()
            self.ms = ms_dic[idx]
            wind_speed *= self.ms / self.cfg.regional_shielding_factor
        else:
            self.ms = 1.0

        self.qz = 0.6 * 1.0e-3 * (wind_speed * self.house.mzcat) ** 2

    def check_internal_pressurisation(self, wind_speed):
        """

        Args:
            wind_speed:

        Returns:
            self.cpi
            self.cpi_wind_speed

        """

        if self.cfg.flags['debris']:
            self.house.debris.run(wind_speed)

        # logging.debug('no_items_mean: {}, no_items:{}'.format(
        #     self.house.debris.no_items_mean,
        #     self.house.debris.no_items))

        # area of breached coverages
        if self.house.coverages is not None:
            self.cpi = self.house.assign_cpi()

            if self.house.debris.damaged_area:
                self.breached = True

    def check_house_collapse(self, wind_speed):
        """

        Args:
            wind_speed:

        Returns: collapse of house

        """
        if not self.collapse:

            for _group in self.house.groups.itervalues():

                if 0 < _group.trigger_collapse_at <= _group.prop_damaged:

                    self.collapse = True

                    for _connection in self.house.connections.itervalues():

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
        for key, value in prop_area_by_scenario.iteritems():
            try:
                tmp = self.cfg.costings[key].compute_cost(value)
            except AssertionError:
                logging.error('{} of {} is invalid'.format(value, key))
            else:
                _list.append(tmp)
        self.repair_cost = array(_list).sum()

        # calculate initial envelope repair cost before water ingress is added
        self.di_except_water = min(self.repair_cost / self.house.replace_cost,
                                   1.0)

        if self.di_except_water < 1.0 and self.cfg.flags['water_ingress']:

            water_ingress_perc = 100.0 * compute_water_ingress_given_damage(
                self.di_except_water, wind_speed,
                self.cfg.water_ingress)

            damage_name = self.determine_scenario_for_water_ingress_costing(
                prop_area_by_scenario)

            self.compute_water_ingress_cost(damage_name, water_ingress_perc)

            _di = (self.repair_cost + self.water_ingress_cost) / self.house.replace_cost
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
        idx = argsort(abs(_df.index - water_ingress_perc))[0]

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
        for _source, target_list in self.cfg.damage_factorings.iteritems():
            for _target in target_list:
                try:
                    revised[_source] -= area_by_group[_target]
                except KeyError:
                    msg = 'either {} or {} is not found in damage factorings'.format(
                        _source, _target)
                    logging.error(msg)

        return revised

    def compute_area_by_scenario(self, revised_area, total_area_by_group):
        area_by_scenario = defaultdict(int)
        total_area_by_scenario = defaultdict(int)
        for scenario, _list in self.cfg.costing_to_group.iteritems():
            for _group in _list:
                if _group in revised_area:
                    area_by_scenario[scenario] += \
                        max(revised_area[_group], 0.0)
                    total_area_by_scenario[scenario] += total_area_by_group[
                        _group]

        # prop_area_by_scenario
        prop_area_by_scenario = {key: value / total_area_by_scenario[key]
                                 for key, value in area_by_scenario.iteritems()}
        return prop_area_by_scenario

    def compute_area_by_group(self):

        area_by_group = defaultdict(int)
        total_area_by_group = defaultdict(int)

        for _group in self.house.groups.itervalues():
            area_by_group[_group.name] += _group.damaged_area
            total_area_by_group[_group.name] += _group.costing_area

        # remove group with zero costing area
        for key, value in total_area_by_group.items():
            if value == 0:
                total_area_by_group.pop(key, None)
                area_by_group.pop(key, None)

        # include DEBRIS when debris is ON
        if self.cfg.coverages_area:
            area_by_group['debris'] = self.house.debris.damaged_area
            total_area_by_group['debris'] = self.cfg.coverages_area

        return area_by_group, total_area_by_group
