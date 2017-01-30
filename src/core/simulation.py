import sys
import os
import time
import copy
import parmap
# import itertools
import pandas as pd
import numpy as np
from optparse import OptionParser
# # from collections import OrderedDict

from house import House
from scenario import Scenario
# import logger
import logging
import database
from version import VERSION_DESC


def simulate_wind_damage_to_houses(cfg):

    # simulator main_loop
    tic = time.time()

    if cfg.parallel:
        logging.info('Starting simulation in parallel')
        list_results = parmap.map(run_simulation_per_house,
                                  range(cfg.no_sims), cfg)
    else:
        logging.info('Starting simulation in serial')
        list_results = []
        for id_sim in range(cfg.no_sims):
            result = run_simulation_per_house(id_sim, cfg)
            list_results.append(result)

    print('{}'.format(time.time()-tic))

    # save to files
    # record randomly generated parameters by simulation model
    hdf = pd.HDFStore(cfg.file_model, mode='w')
    for item in ['wind_orientation', 'profile', 'construction_level', 'mzcat']:
        ps_ = pd.Series([x[item] for x in list_results])
        hdf.append(item, ps_, format='t')

    for item in ['qz', 'Ms', 'cpi', 'cpiAt', 'collapse', 'di',
                 'di_except_water']:
        ps_ = pd.DataFrame([x[item] for x in list_results])
        hdf.append(item, ps_, format='t')
    hdf.close()

    # by connection for each model
    hdf = pd.HDFStore(cfg.file_conn, mode='w')
    for item in ['dead_load', 'strength']:
        ps_ = pd.DataFrame([x[item] for x in list_results])
        hdf.append(item, ps_, format='t')

    for item in ['damaged', 'failure_v_raw', 'load']:
        dic_ = {key: x[item] for key, x in enumerate(list_results)}
        hdf[item] = pd.Panel(dic_)
        # hdf.put(item, pd.Panel(dic_), format='t')
        # pd.Panel(dic_).to_hdf(hdf, key=item, format='f')
    hdf.close()

    # results by type for each model
    hdf = pd.HDFStore(cfg.file_type, mode='w')
    for item in ['damage_capacity', 'prop_damaged_type']:
        dic_ = {key: x[item] for key, x in enumerate(list_results)}
        hdf[item] = pd.Panel(dic_)
    hdf.close()

    # results by group for each model
    hdf = pd.HDFStore(cfg.file_group, mode='w')
    for item in ['prop_damaged_group', 'prop_damaged_area', 'repair_cost']:
        dic_ = {key: x[item] for key, x in enumerate(list_results)}
        hdf[item] = pd.Panel(dic_)
    hdf.close()

    # by zone for each model
    hdf = pd.HDFStore(cfg.file_zone, mode='w')
    for item in ['cpe_eave']:
        ps_ = pd.DataFrame([x[item] for x in list_results])
        hdf.append(item, ps_, format='t')

    for item in ['pz', 'pz_str', 'area', 'cpe', 'cpe_str']:
        dic_ = {key: x[item] for key, x in enumerate(list_results)}
        hdf[item] = pd.Panel(dic_)
    hdf.close()

    return list_results


def run_simulation_per_house(id_sim, cfg):
    """

    Args:
        id_sim:
        cfg:

    Returns:

    """

    seed = cfg.flags['random_seed'] + id_sim
    house_damage = HouseDamage(cfg, seed)

    for item in ['profile', 'wind_orientation', 'construction_level', 'mzcat']:
        house_damage.bucket[item] = getattr(house_damage.house, item)

    # iteration over wind speed list
    for id_speed, wind_speed in enumerate(cfg.speeds):

        # simulate sampled house
        logging.info('model #{} at speed {:.3f}'.format(id_sim, wind_speed))

        house_damage.run_simulation(wind_speed)

    return copy.deepcopy(house_damage.bucket)


class HouseDamage(object):

    def __init__(self, cfg, seed):

        assert isinstance(cfg, Scenario)
        assert isinstance(seed, int)

        self.cfg = cfg
        self.rnd_state = np.random.RandomState(seed)

        self.house = House(cfg, self.rnd_state)

        self.qz = None
        self.Ms = None
        self.cpi = None
        self.cpiAt = None
        self.collapse = False
        self.di = None
        self.di_prev = None
        self.di_except_water = None

        self.bucket = None
        self.init_bucket()

    def run_simulation(self, wind_speed):

        # only check if debris is ON
        # cpi is computed here
        self.check_pressurized_failure(wind_speed)

        # compute load by zone
        # load = qz * (Cpe + Cpi) * A + Dead
        self.calculate_qz_Ms(wind_speed)

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
                self.distribute_damage(_group)

        self.check_house_collapse(wind_speed)
        self.cal_damage_index()

        self.fill_bucket(wind_speed)

    def init_bucket(self):

        self.bucket = dict()

        list_groups = [x.name for x in self.house.groups.itervalues()]
        list_types = [x.name for x in self.house.types.itervalues()]
        list_conns = [x.name for x in self.house.connections.itervalues()]
        list_zones = [x.name for x in self.house.zones.itervalues()]

        # by wind speed
        for item in ['qz', 'Ms', 'cpi', 'cpiAt', 'collapse', 'di',
                     'di_except_water']:
            self.bucket[item] = pd.Series(index=self.cfg.idx_speeds,
                                          dtype=float)

        # by group
        for item in ['prop_damaged_group', 'prop_damaged_area', 'repair_cost']:
            self.bucket[item] = pd.DataFrame(
                dtype=float, columns=list_groups, index=self.cfg.idx_speeds)

        # by type
        for item in ['damage_capacity', 'prop_damaged_type']:
            self.bucket[item] = pd.DataFrame(
                dtype=float, columns=list_types, index=self.cfg.idx_speeds)

        # by connection
        for item in ['damaged', 'failure_v_raw', 'load']:
            self.bucket[item] = pd.DataFrame(
                dtype=float, columns=list_conns, index=self.cfg.idx_speeds)

        for item in list_conns:
            for _att in ['strength', 'dead_load']:
                self.bucket.setdefault(_att, {})[item] = None

        # by zone
        for item in ['pz', 'pz_str', 'area', 'cpe', 'cpe_str']:
            self.bucket[item] = pd.DataFrame(
                None, columns=list_zones, index=self.cfg.idx_speeds)

        for item in list_zones:
            for _att in ['cpe_eave']:
                self.bucket.setdefault(_att, {})[item] = None

    def fill_bucket(self, wind_speed):

        ispeed = np.abs(self.cfg.speeds-wind_speed).argmin()

        for item in ['qz', 'Ms', 'cpi', 'cpiAt', 'collapse', 'di',
                     'di_except_water']:
            self.bucket[item][ispeed] = getattr(self, item)

        # by group
        for item in ['prop_damaged_group', 'prop_damaged_area', 'repair_cost']:
            for _value in self.house.groups.itervalues():
                self.bucket[item][_value.name][ispeed] = getattr(_value, item)

        # by type
        for item in ['damage_capacity', 'prop_damaged_type']:
            for _value in self.house.types.itervalues():
                self.bucket[item][_value.name][ispeed] = getattr(_value, item)

        # by connection
        for _value in self.house.connections.itervalues():

            for item in ['damaged', 'failure_v_raw', 'load']:
                self.bucket[item][_value.name][ispeed] = getattr(_value, item)

            for item in ['strength', 'dead_load']:
                self.bucket[item][_value.name] = getattr(_value, item)

        # by zone
        for _value in self.house.zones.itervalues():

            for item in ['pz', 'pz_str', 'area', 'cpe', 'cpe_str']:
                self.bucket[item][_value.name][ispeed] = getattr(_value, item)

            for item in ['cpe_eave']:
                self.bucket[item][_value.name] = getattr(_value, item)

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

    def check_pressurized_failure(self, wind_speed):
        if self.cfg.flags['debris']:
            print 'Not implemented'
            # self.debris_manager.run(v)
            # if self.cpi == 0 and self.debris_manager.get_breached():
            #     self.cpi = 0.7
            #     self.cpiAt = v
        else:
            self.cpi = 0.0
            self.cpiAt = 0.0

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
                    for _conn in self.house.connections:
                        _conn.set_damage(wind_speed)
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
        for source_id, target_list in self.house.factors_costing.iteritems():
            for target_id in target_list:
                self.house.groups[source_id].prop_damaged_area -= self.house.groups[target_id].prop_damaged_area

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

        logging.debug('total repair_cost:{:.3f}, di: {:.3f}'.format(repair_cost,
                                                              self.di))

        self.di_prev = self.di

    def distribute_damage(self, group):

        if group.name == 'batten':

            assert group.dist_dir == 'row'
            # distributed over the same number

            # row: index of chr, col: index of number e.g, 0,0 : A1
            for row, col in zip(*np.where(group.damage_grid)):

                source_conn = group.conn_by_grid[row, col]

                intact = np.where(~group.damage_grid[:, col])[0]

                intact_left = intact[np.where(row > intact)[0]]
                intact_right = intact[np.where(row < intact)[0]]

                logging.debug('cols of intact conns:{}'.format(intact))

                if intact_left.size * intact_right.size > 0:
                    infl_coeff = 0.5  # can be prop. to distance later
                else:
                    infl_coeff = 1.0

                # logging.debug('conn {} at {} is distributed'.format(
                #     source_conn.name, source_conn.grid))
                # source_conn.distributed = True

                try:
                    group.update_influence_target_conn(intact_right[0],
                                                       col,
                                                       source_conn,
                                                       infl_coeff)

                except IndexError:
                    #logging.debug('no update from conn {}'.format(
                    #    source_conn.name))
                    pass

                try:
                    group.update_influence_target_conn(intact[-1],
                                                       col,
                                                       source_conn,
                                                       infl_coeff)
                except IndexError:
                    #logging.debug('no update from conn {}'.format(
                    #    source_conn.name))
                    pass

        elif group.name == 'sheeting':

            assert group.dist_dir == 'col'
            # distributed over the same char.

            # row: index of chr, col: index of number e.g, 0,0 : A1
            for row, col in zip(*np.where(group.damage_grid)):

                source_zone = self.house.zone_by_grid[row, col]

                intact = np.where(~group.damage_grid[row, :])[0]

                intact_left = intact[np.where(col > intact)[0]]
                intact_right = intact[np.where(col < intact)[0]]

                logging.debug('rows of intact zones:{}'.format(intact))

                if intact_left.size * intact_right.size > 0:
                    infl_coeff = 0.5
                else:
                    infl_coeff = 1.0

                # if group.set_zone_to_zero:
                #     logging.debug('zone {} at {} distributed'.format(
                #         source_zone.name, source_zone.grid))
                #     source_zone.distributed = True

                try:
                    group.update_influence_target_conn(row,
                                                       intact_right[0],
                                                       source_zone,
                                                       infl_coeff)
                except IndexError:
                    pass
                    #logging.debug('no update from zone {}'.format(
                    #    source_zone.name))

                try:
                    group.update_influence_target_conn(row,
                                                       intact_left[-1],
                                                       source_zone,
                                                       infl_coeff)
                except IndexError:
                    # logging.debug('no update from zone {}'.format(
                    #     source_zone.name))
                    pass

        else:

            try:
                assert group.dist_dir == 'col'
                # distributed over the same char.
            except AssertionError:
                logging.debug('group {} is skipped in distribution'.format(
                    group.name))
            else:
                # row: index of chr, col: index of number e.g, 0,0 : A1
                for row, col in zip(*np.where(group.damage_grid)):

                    source_conn = group.conn_by_grid[row, col]

                    intact = np.where(~group.damage_grid[row, :])[0]

                    intact_left = intact[np.where(col > intact)[0]]
                    intact_right = intact[np.where(col < intact)[0]]

                    logging.debug('rows of intact conns:{}'.format(intact))

                    if intact_left.size * intact_right.size > 0:
                        infl_coeff = 0.5
                    else:
                        infl_coeff = 1.0

                    # if group.set_zone_to_zero:
                    #     logging.debug('zone {} at {} distributed'.format(
                    #         source_zone.name, source_zone.grid))
                    #     source_zone.distributed = True

                    try:
                        group.update_influence_target_conn(row,
                                                           intact_right[0],
                                                           source_conn,
                                                           infl_coeff)
                    except IndexError:
                        pass
                        # logging.debug('no update from zone {}'.format(
                        #    source_zone.name))

                    try:
                        group.update_influence_target_conn(row,
                                                           intact_left[-1],
                                                           source_conn,
                                                           infl_coeff)
                    except IndexError:
                        # logging.debug('no update from zone {}'.format(
                        #     source_zone.name))
                        pass


def main():
    USAGE = ('%prog -s <scenario_file> [-m <model database file>] '
             '[-o <output_folder>]')
    parser = OptionParser(usage=USAGE, version=VERSION_DESC)
    parser.add_option("-s", "--scenario",
                      dest="scenario_filename",
                      help="read scenario description from FILE",
                      metavar="FILE")
    parser.add_option("-m", "--model",
                      dest="model_database",
                      help="Use Model Database from FILE",
                      metavar="FILE")
    parser.add_option("-o", "--output",
                      dest="output_folder",
                      help="folder name to store simulation results",
                      metavar="FOLDER")
    # parser.add_option("-v", "--verbose",
    #                   action="store_true",
    #                   dest="verbose",
    #                   default=False,
    #                   help="show verbose simulator output")
    parser.add_option("-i", "--import",
                      dest="data_folder",
                      help="data folder to import into model.db",
                      metavar="FOLDER")
    parser.add_option("--plot_vuln",
                      action="store_true",
                      dest="plot_vuln",
                      default=False,
                      help="show vulnerability plot")
    parser.add_option("--plot_frag",
                      action="store_true",
                      dest="plot_frag",
                      default=False,
                      help="show fragility plot")

    (options, args) = parser.parse_args()

    path_, _ = os.path.split(sys.argv[0])

    if options.output_folder is None:
        options.output_folder = os.path.abspath(os.path.join(path_,
                                                             '../output'))
    else:
        options.output_folder = os.path.abspath(
            os.path.join(os.getcwd(), options.output_folder))

    # set up logging
    file_logger = os.path.join(options.output_folder, 'log.txt')
    logging.basicConfig(filename=file_logger,
                        filemode='w',
                        level=logging.DEBUG,
                        format='%(levelname)s %(message)s')

    if options.data_folder:
        db = database.DatabaseManager(options.model_database)
        import dbimport
        dbimport.import_model(options.data_folder, db)
        db.close()
    else:
        if options.scenario_filename:
            conf = Scenario(cfg_file=options.scenario_filename,
                            output_path=options.output_folder)
            _ = simulate_wind_damage_to_houses(conf)
        else:
            print '\nERROR: Must provide a scenario file to run simulator...\n'
            parser.print_help()

    logging.info('Program finished')

if __name__ == '__main__':
    main()
