import copy
import logging
import numpy as np
from collections import defaultdict

from vaws.house import House
from vaws.config import Config
from vaws.damage_costing import compute_water_ingress_given_damage


class HouseDamage(object):

    def __init__(self, cfg, seed):

        assert isinstance(cfg, Config)
        assert isinstance(seed, int)

        self.cfg = cfg
        self.seed = seed
        self.rnd_state = np.random.RandomState(self.seed)

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

        self.bucket = {}
        self.init_bucket()

    def run_simulation(self, wind_speed):

        if not self.collapse:

            logging.info('wind speed {:.3f}'.format(wind_speed))

            # cpi is computed here
            self.check_internal_pressurisation(wind_speed)

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
                if _group.damaged and self.cfg.flags.get('dmg_distribute_{}'.format(
                        _group.name)):
                    _group.update_influence(self.house)

            self.check_house_collapse(wind_speed)
            self.compute_damage_index(wind_speed)

            self.fill_bucket()

        return copy.deepcopy(self.bucket)

    def init_bucket(self):

        # house
        for item in ['house', 'house_damage', 'debris']:
            for att in getattr(self.cfg, '{}_bucket'.format(item)):
                self.bucket.setdefault(item, {})[att] = None

        # components
        for item in self.cfg.list_components:
            self.bucket[item] = {}
            for _conn in getattr(self.cfg, 'list_{}s'.format(item)):
                self.bucket[item][_conn] = {}
                for att in getattr(self.cfg, '{}_bucket'.format(item)):
                    self.bucket[item][_conn][att] = None

    def fill_bucket(self):

        # house
        for item in self.cfg.house_bucket:
            self.bucket['house'][item] = getattr(self.house, item)

        for item in self.cfg.house_damage_bucket:
            self.bucket['house_damage'][item] = getattr(self, item)

        if self.cfg.flags['debris']:
            for item in self.cfg.debris_bucket:
                self.bucket['debris'][item] = getattr(self.house.debris, item)

        # components
        for item in self.cfg.list_components:
            for att in getattr(self.cfg, '{}_bucket'.format(item)):
                _dic = getattr(self.house, '{}s'.format(item))
                for _conn, value in _dic.iteritems():
                    self.bucket[item][_conn][att] = getattr(value, att)

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

        if self.cfg.flags['debris']:
            self.house.debris.run(wind_speed)

        # logging.debug('no_items_mean: {}, no_items:{}'.format(
        #     self.house.debris.no_items_mean,
        #     self.house.debris.no_items))

        # area of breached coverages
        if self.house.coverages is not None:
            self.house.assign_cpi()

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

        prop_area_by_group = {key: value / total_area_by_group[key]
                              for key, value in area_by_group.iteritems()}

        # TODO
        # include DEBRIS when debris is ON

        # apply damage factoring
        revised_prop = copy.deepcopy(prop_area_by_group)
        for _source, target_list in self.cfg.damage_factorings.iteritems():
            for _target in target_list:
                revised_prop[_source] -= prop_area_by_group[_target]

        # sum of area by scenario
        area_by_scenario = defaultdict(int)
        total_area_by_scenario = defaultdict(int)
        for scenario, _list in self.cfg.costing_to_group.iteritems():
            for _group in _list:
                area_by_scenario[scenario] += \
                    max(revised_prop[_group], 0.0) * total_area_by_group[_group]
                total_area_by_scenario[scenario] += total_area_by_group[_group]

        # prop_area_by_scenario
        prop_area_by_scenario = {key: value / total_area_by_scenario[key]
                                 for key, value in area_by_scenario.iteritems()}

        _list = [self.cfg.costings[key].compute_cost(value)
                 for key, value in prop_area_by_scenario.iteritems()]
        self.repair_cost = np.array(_list).sum()

        # calculate initial envelope repair cost before water ingress is added
        self.di_except_water = min(self.repair_cost / self.house.replace_cost,
                                   1.0)

        if self.di_except_water < 1.0 and self.cfg.flags['water_ingress']:

            # compute water ingress
            water_ingress_perc = 100.0 * compute_water_ingress_given_damage(
                self.di_except_water, wind_speed,
                self.cfg.water_ingress_given_di)

            # determine damage scenario
            damage_name = 'WI only'  # default
            for _name in self.cfg.damage_order_by_water_ingress:
                if prop_area_by_scenario[_name]:
                    damage_name = _name
                    break

            # finding index close to water ingress threshold
            _df = self.cfg.water_ingress_costings[damage_name]
            idx = np.argsort(np.abs(_df.index - water_ingress_perc))[0]

            self.water_ingress_cost = \
                _df.at[idx, 'costing'].compute_cost(self.di_except_water)
            _di = (self.repair_cost +
                   self.water_ingress_cost) / self.house.replace_cost
            self.di = min(_di, 1.0)

        else:
            self.di = self.di_except_water

        logging.info('At {}, repair_cost: {:.3f}, cost by water: {:.3f}, '
                     'di except water:{:.3f}, di: {:.3f}'.format(
            wind_speed, self.repair_cost, self.water_ingress_cost,
            self.di_except_water, self.di))

    # @staticmethod
    # def get_cpi_for_dominant_opening(ratio, ):
