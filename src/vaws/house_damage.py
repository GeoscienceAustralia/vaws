import copy
import logging
import pandas as pd
import numpy as np
from collections import defaultdict

from vaws.house import House
from vaws.config import Config
from vaws.damage_costing import cal_water_ingress_given_damage


class HouseDamage(object):

    def __init__(self, cfg, seed):

        assert isinstance(cfg, Config)
        assert isinstance(seed, int)

        self.cfg = cfg
        self.rnd_state = np.random.RandomState(seed)

        self.house = House(cfg, self.rnd_state)

        # vary over wind speeds
        self.qz = None
        self.ms = None
        self.cpi = 0.0
        self.cpi_wind_speed = 0.0
        self.collapse = False
        self.repair_cost = 0.0
        self.water_ingress_cost = 0.0
        self.di = None
        self.di_except_water = None

        self.bucket = dict()
        self.init_bucket()

    def run_simulation(self, wind_speed):

        logging.info('wind speed {:.3f}'.format(wind_speed))

        # only check if debris is ON
        # cpi is computed here

        if self.cfg.flags['debris']:
            self.check_internal_pressurisation(wind_speed)

        # compute load by zone
        self.calculate_qz_ms(wind_speed)

        # load = qz * (Cpe + Cpi) * A + Dead
        for _zone in self.house.zones.itervalues():

            _zone.calc_zone_pressures(self.house.wind_orientation,
                                      self.cpi,
                                      self.qz,
                                      self.ms,
                                      self.cfg.building_spacing,
                                      self.cfg.flags['diff_shielding'])

        # check damage by connection type group
        for _group in self.house.groups.itervalues():

            _group.check_damage(wind_speed)
            _group.cal_damaged_area()

            if _group.damaged and self.cfg.flags.get('dmg_distribute_{}'.format(
                    _group.name)):
                _group.distribute_damage()

        self.check_house_collapse(wind_speed)
        self.cal_damage_index(wind_speed)

        self.fill_bucket()

        return copy.deepcopy(self.bucket)

    def init_bucket(self):

        # house
        for item in self.cfg.list_house_bucket + self.cfg.list_debris_bucket + \
                self.cfg.list_house_damage_bucket:
            self.bucket.setdefault('house', {})[item] = None

        # components
        for item in self.cfg.list_components:
            _index = getattr(self.cfg, 'list_{}s'.format(item))
            _columns = getattr(self.cfg, 'list_{}_bucket'.format(item))
            self.bucket[item] = pd.DataFrame(index=_index, columns=_columns)

    def fill_bucket(self):

        # house
        for item in self.cfg.list_house_damage_bucket:
            self.bucket['house'][item] = getattr(self, item)

        for item in self.cfg.list_house_bucket:
            self.bucket['house'][item] = getattr(self.house, item)

        if self.cfg.flags['debris']:
            for item in self.cfg.list_debris_bucket:
                self.bucket['house'][item] = getattr(self.house.debris, item)

        # components
        for item in self.cfg.list_components:
            for att in getattr(self.cfg, 'list_{}_bucket'.format(item)):
                _dic = getattr(self.house, '{}s'.format(item))
                for key, value in _dic.iteritems():
                    self.bucket[item].at[key, att] = getattr(value, att)

    def calculate_qz_ms(self, wind_speed):
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
            thresholds = np.array([63, 63 + 15])
            ms_dic = {0: 1.0, 1: 0.85, 2: 0.95}
            idx = sum(thresholds <= self.rnd_state.random_integers(0, 100))
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

        self.house.debris.run(wind_speed)
        logging.debug('no_items_mean: {}, no_items:{}'.format(
            self.house.debris.no_items_mean,
            self.house.debris.no_items))

        if self.cpi < 0.7 and self.house.debris.breached:
            self.cpi = 0.7
            self.cpi_wind_speed = wind_speed

    def check_house_collapse(self, wind_speed):
        """

        Args:
            wind_speed:

        Returns: collapse of house

        """
        if not self.collapse:

            for _group in self.house.groups.itervalues():

                if 0 < _group.trigger_collapse_at <= _group.prop_damaged_group:

                    self.collapse = True

                    # FIXME!! Don't understand WHY?
                    for _connection in self.house.connections.itervalues():
                        _connection.set_damage(wind_speed)
                    #
                    # for _zone in self.house.zones:
                    #     _zone.effective_area = 0.0

    def cal_damage_index(self, wind_speed):
        """

        Args:
            wind_speed:

        Returns:
            damage_index: repair cost / replacement cost

        Note:

            1. sum of damaged area by each connection group
            2. revised damage area by group by applying damage factoring
            3. calculate sum of revised damaged area by damage scenario
            4. apply costing modules

        """

        # sum of damaged area by group
        area_by_group = defaultdict(int)
        total_area_by_group = defaultdict(int)
        for _group in self.house.groups.itervalues():
            area_by_group[_group.name] += _group.damaged_area
            total_area_by_group[_group.name] += _group.costing_area

        # prop_area_by_group
        prop_area_by_group = {key: value / total_area_by_group[key]
                              for key, value in area_by_group.iteritems()}

        # apply damage factoring
        revised_prop = copy.deepcopy(prop_area_by_group)
        for _source, target_list in self.cfg.dic_damage_factorings.iteritems():
            for _target in target_list:
                revised_prop[_source] -= prop_area_by_group[_target]

        # sum of area by scenario
        area_by_scenario = defaultdict(int)
        total_area_by_scenario = defaultdict(int)
        for scenario, _list in self.cfg.dic_costing_to_group.iteritems():
            for _group in _list:
                area_by_scenario[scenario] += \
                    max(revised_prop[_group], 0.0) * total_area_by_group[_group]
                total_area_by_scenario[scenario] += total_area_by_group[_group]

        # prop_area_by_scenario
        prop_area_by_scenario = {key: value / total_area_by_scenario[key]
                                 for key, value in area_by_scenario.iteritems()}

        _list = [self.cfg.dic_costings[key].calculate_cost(value)
                 for key, value in prop_area_by_scenario.iteritems()]
        self.repair_cost = np.array(_list).sum()

        # calculate initial envelope repair cost before water ingress is added
        self.di_except_water = min(self.repair_cost / self.house.replace_cost,
                                   1.0)

        if self.di_except_water < 1.0 and self.cfg.flags['water_ingress']:

            # compute water ingress
            water_ingress_perc = 100.0 * cal_water_ingress_given_damage(
                self.di_except_water, wind_speed,
                self.cfg.water_ingress_given_di)

            # determine damage scenario
            damage_name = 'WI only'  # default
            for _name in self.cfg.damage_order_by_water_ingress:
                if prop_area_by_scenario[_name]:
                    damage_name = _name
                    break

            # finding index close to water ingress threshold
            _df = self.cfg.dic_water_ingress_costings[damage_name]
            idx = np.argsort(np.abs(_df.index - water_ingress_perc))[0]

            self.water_ingress_cost = \
                _df.at[idx, 'costing'].calculate_cost(self.di_except_water)
            _di = (self.repair_cost +
                   self.water_ingress_cost) / self.house.replace_cost
            self.di = min(_di, 1.0)

        else:
            self.di = self.di_except_water

        logging.info('repair_cost:{:.3f}, water:{:.3f}, '
                     'di_except_water:{:.3f}, di:{:.3f}'.format(
            self.repair_cost, self.water_ingress_cost, self.di_except_water, self.di))
