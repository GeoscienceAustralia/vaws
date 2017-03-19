import logging
import copy
import pandas as pd
import numpy as np

from house import House
from scenario import Scenario


class HouseDamage(object):

    def __init__(self, cfg, seed):

        assert isinstance(cfg, Scenario)
        assert isinstance(seed, int)

        self.cfg = cfg
        self.rnd_state = np.random.RandomState(seed)

        self.house = House(cfg, self.rnd_state)

        # vary over wind speeds
        self.qz = None
        self.Ms = None
        self.cpi = 0.0
        self.cpi_wind_speed = None
        self.collapse = False
        self.di = None
        self.di_except_water = None

        self.bucket = dict()
        self.init_bucket()

    def run_simulation(self, wind_speed):

        logging.info('wind speed {:.3f}'.format(wind_speed))

        # only check if debris is ON
        # cpi is computed here
        self.check_internal_pressurisation(wind_speed)

        # compute load by zone
        self.calculate_qz_Ms(wind_speed)

        # load = qz * (Cpe + Cpi) * A + Dead
        for _zone in self.house.zones.itervalues():

            _zone.calc_zone_pressures(self.house.wind_orientation,
                                      self.cpi,
                                      self.qz,
                                      self.Ms,
                                      self.cfg.building_spacing,
                                      self.cfg.flags['diff_shielding'])

        # check damage by connection type group
        for _group in self.house.groups.itervalues():

            if self.cfg.flags.get('conn_type_group_{}'.format(_group.name)):
                _group.check_damage(wind_speed)
                _group.cal_prop_damaged()

                if _group.damaged and self.cfg.flags.get('dmg_distribute_{}'.format(
                        _group.name)):
                    _group.distribute_damage()

        self.check_house_collapse(wind_speed)
        self.cal_damage_index()

        self.fill_bucket()

        return copy.deepcopy(self.bucket)

    def init_bucket(self):

        # house
        for item in self.cfg.list_house_bucket + self.cfg.list_debris_bucket + \
                self.cfg.list_house_damage_bucket:
            self.bucket.setdefault('house', {})[item] = None

        # components
        for item in self.cfg.list_compnents:
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
        for item in self.cfg.list_compnents:
            for att in getattr(self.cfg, 'list_{}_bucket'.format(item)):
                _dic = getattr(self.house, '{}s'.format(item))
                for key, value in _dic.iteritems():
                    self.bucket[item].loc[key, att] = getattr(value, att)

    def calculate_qz_Ms(self, wind_speed):
        """
        calculate qz, velocity pressure given wind velocity
        qz = 0.6*10-3*(Mz,cat*V)**2
        Args:
            wind_speed: wind velocity (m/s)

        Returns:
            qz
            update Ms

        """
        if self.cfg.regional_shielding_factor <= 0.85:
            thresholds = np.array([63, 63 + 15])
            ms_dic = {0: 1.0, 1: 0.85, 2: 0.95}
            idx = sum(thresholds <= self.rnd_state.random_integers(0, 100))
            self.Ms = ms_dic[idx]
            wind_speed *= self.Ms / self.cfg.regional_shielding_factor
        else:
            self.Ms = 1.0

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

    def cal_damage_index(self):
        """

        Args:
            wind_speed:

        Returns:
            damage_index: repair cost / replacement cost

        """

        # factor costing
        damaged_area_final = dict()
        for source_id, target_list in self.cfg.dic_damage_factorings.iteritems():
            damaged_area_final[source_id] = self.house.groups[source_id].prop_damaged_area
            for target_id in target_list:
                damaged_area_final[source_id] -= self.house.groups[target_id].prop_damaged_area

        # assign value
        for source_id in self.cfg.dic_damage_factorings:
            self.house.groups[source_id].prop_damaged_area = damaged_area_final[source_id]

        # calculate repair cost
        repair_cost = 0.0
        for group in self.house.groups.itervalues():
            group.cal_repair_cost(group.prop_damaged_area)
            repair_cost += group.repair_cost

        # calculate initial envelope repair cost before water ingress is added
        self.di_except_water = min(repair_cost / self.house.replace_cost, 1.0)

        if self.di_except_water < 1.0 and self.cfg.flags['water_ingress']:

            print 'Not implemented'
            # (self.water_ratio, self.water_damage_name, self.water_ingress_cost,
            #  self.water_costing) = \
            #     wateringress.get_costing_for_envelope_damage_at_v(
            #         self.di_except_water, wind_speed, self.house.water_groups)
            #
            # repair_cost += self.water_ingress_cost
            #
            # # combined internal + envelope damage costing can now be calculated
            # self.di = min(repair_cost / self.house.replace_cost, 1.0)
        else:
            self.di = self.di_except_water

        logging.info('total repair_cost:{:.3f}, di: {:.3f}'.format(repair_cost,
                                                                   self.di))
