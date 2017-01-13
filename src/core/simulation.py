# adjust python path so we may import things from peer packages
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

# import terrain
# import house
from house import House
# import connection
# from connection import calc_connection_loads
# import zone
# from zone import calc_zone_pressures
from scenario import Scenario
# import curve
# import curve_log
import logger
import database
# from debris import DebrisManager
# import debris
# import wateringress
# import engine
# import output
from version import VERSION_DESC


def simulate_wind_damage_to_houses(cfg):

    # simulator main_loop
    tic = time.time()

    if cfg.parallel:
        list_results = parmap.map(run_simulation_per_house,
                                  range(cfg.no_sims), cfg)
    else:
        list_results = []
        for id_sim in range(cfg.no_sims):
            result = run_simulation_per_house(id_sim, cfg)
            list_results.append(result)

    print('{}'.format(time.time()-tic))

    # write to files
    pd.DataFrame([x['dead_load'] for x in list_results]).to_csv(
        cfg.file_dead_load_by_conn)

    # list_results
    # file_dead_load_by_conn
    # 'file_strength_by_conn',
    #
    #
    # 'file_dmg_by_conn',
    #
    # 'file_dmg_area_by_conn_grp',
    # 'file_dmg_dist_by_conn',
    # 'file_dmg_freq_by_conn_type',
    # 'file_dmg_idx',
    # 'file_dmg_map_by_conn_type',
    # 'file_dmg_pct_by_conn_type',
    # 'file_eff_area_by_zone',
    # 'file_frag',
    # 'file_house_cpi',
    # 'file_repair_cost_by_conn_grp',
    # 'file_rnd_parameters',
    # 'file_water',
    # 'file_wind_debris',

    return list_results


def run_simulation_per_house(id_sim, cfg):
    """

    Args:
        id_sim:
        cfg:

    Returns:

    """

    rnd_state = np.random.RandomState(cfg.flags['random_seed'] + id_sim)
    house_damage = HouseDamage(cfg, rnd_state)

    # iteration over wind speed list
    for id_speed, wind_speed in enumerate(cfg.speeds):

        # simulate sampled house
        house_damage.run_simulation(wind_speed)

    return copy.deepcopy(house_damage.bucket)


class HouseDamage(object):

    def __init__(self, cfg, rnd_state):

        assert isinstance(cfg, Scenario)
        assert isinstance(rnd_state, np.random.RandomState)

        self.cfg = cfg
        self.rnd_state = rnd_state

        self.house = House(cfg, rnd_state)

        self.qz = None
        self.Ms = None
        self.cpi = None
        self.cpiAt = None
        self.collapse = False
        self.di = None
        self.di_except_water = None

        self.bucket = dict()

        self.list_groups = [x.name for x in self.house.groups.itervalues()]
        self.list_types = [x.name for x in self.house.types.itervalues()]
        self.list_conns = [x.name for x in self.house.connections.itervalues()]

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
            _group.check_damage(wind_speed)

        # redistribute by
        # for _group in self.house.groups.itervalues():
        #     self.redistribute_damage(_group)

        self.check_house_collapse(wind_speed)
        self.cal_damage_index()
        self.fill_bucket(wind_speed)

    def init_bucket(self):

        # by wind speed
        for item in ['qz', 'Ms', 'cpi', 'cpiAt', 'collapse', 'di',
                     'di_except_water']:
            self.bucket[item] = pd.Series(None, index=self.cfg.idx_speeds)

        # by group
        for item in ['prop_damaged', 'prop_damaged_area', 'repair_cost']:
            self.bucket[item] = pd.DataFrame(
                None, columns=self.list_groups, index=self.cfg.idx_speeds)

        # by type
        for item in ['damage_capacity', 'prop_damaged']:
            self.bucket[item] = pd.DataFrame(
                None, columns=self.list_types, index=self.cfg.idx_speeds)

        # by connection
        for item in ['damaged', 'damaged_by_dist', 'failure_v_raw', 'load']:
            self.bucket[item] = pd.DataFrame(
                None, columns=self.list_conns, index=self.cfg.idx_speeds)

        for item in self.list_conns:
            self.bucket.setdefault('strength', {})[item] = None
            self.bucket.setdefault('dead_load', {})[item] = None

        # by zone
        # for item in ['result_effective_area']:
        #     result_buckets[item] = pd.DataFrame(
        #         None, columns=cfg.list_zone, index=cfg.idx_speeds)

    def fill_bucket(self, wind_speed):

        ispeed = np.abs(self.cfg.speeds-wind_speed).argmin()

        for item in ['qz', 'Ms', 'cpi', 'cpiAt', 'collapse', 'di',
                     'di_except_water']:
            self.bucket[item][ispeed] = getattr(self, item)

        # by group
        for item in ['prop_damaged', 'prop_damaged_area', 'repair_cost']:
            for _value in self.house.groups.itervalues():
                self.bucket[item][_value.name][ispeed] = getattr(_value, item)

        # by type
        for item in ['damage_capacity', 'prop_damaged']:
            for _value in self.house.types.itervalues():
                self.bucket[item][_value.name][ispeed] = getattr(_value, item)

        # by connection
        for _value in self.house.connections.itervalues():

            for item in ['damaged', 'damaged_by_dist', 'failure_v_raw', 'load']:
                self.bucket[item][_value.name][ispeed] = getattr(_value, item)

            for item in ['strength', 'dead_load']:
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

    def check_pressurized_failure(self, v):
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

                if 0 < _group.trigger_collapse_at <= _group.prop_damaged:

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

        # calculate repair cost
        repair_cost = 0.0
        for _group in self.house.groups.itervalues():

            if _group.name == 'debris':
                print 'Not implemented yet'
                # if self.debris_manager:
                #     conn_type_group.result_percent_damaged = \
                #         self.debris_manager.result_dmgperc
                # else:
                #     conn_type_group.result_percent_damaged = 0
            else:
                # compute prop_damaged_area, prop_damaged, and repair_cost
                _group.cal_prop_damaged()

                prop_damaged_area = _group.prop_damaged_area

                try:
                    _group_id_uppers = self.house.factors_costing[_group.id]
                except KeyError:
                    pass
                else:
                    for _id in _group_id_uppers:
                        prop_damaged_area -= self.house.groups[_id].prop_damaged_area

                _group.cal_repair_cost(prop_damaged_area)

                repair_cost += _group.repair_cost

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

        self.prev_di = self.di

    """
    def redistribute_damage(self, conn_type_group):
        # setup for distribution
        if conn_type_group.distribution_order <= 0:
            return

        if conn_type_group.distribution_direction == 'col':
            distByCol = True
            primaryDir = self.house.cols
            secondaryDir = self.house.rows
        else:
            distByCol = False
            primaryDir = self.house.rows
            secondaryDir = self.house.cols

        # walk the zone grid for current group
        # (only one conn of each group per zone)
        for i in primaryDir:
            for j in secondaryDir:
                # determine zoneLocation and then zone
                # cols + rows
                if distByCol:
                    zoneLoc = i
                    zoneLoc += str(j)
                else:
                    zoneLoc = j
                    zoneLoc += str(i)

                # not all grid locations have a zone
                # not all zones have area or connections remaining
                # not all zones have connections of all types
                # grab appropriate connection from zone
                z = zoneByLocationMap[zoneLoc]
                if (zoneLoc in house.zoneByLocationMap and
                        z.result_effective_area and
                        z.located_conns and
                            conn_type_group.group_name in connByZoneTypeMap[
                            z.zone_name] and
                        self.cfg.flags.get('dmg_distribute_{}'.format(
                            conn_type_group.group_name))):
                    conn = connByZoneTypeMap[z.zone_name][
                        conn_type_group.group_name]
                else:
                    continue

                # if that connection is (newly) damaged then redistribute
                # load/infl/area
                if conn.result_damaged and not conn.result_damage_distributed:
                    # print 'Connection: {} newly damaged'.format(conn_type_group.group_name)

                    if conn_type_group.patch_distribution == 1:
                        patchList = \
                            self.db.qryConnectionPatchesFromDamagedConn(conn.id)

                        for patch in patchList:
                            patch_zone = zoneByIDMap[patch[1]]
                            patch_conn = connByIDMap[patch[0]]
                            curr_infl = inflZonesByConn[patch_conn].get(
                                patch_zone, None)
                            if curr_infl is None:
                                # print 'discarding patch as no existing patch present: %s-->%s = %f' % (patch_conn, patch_zone, patch[2])
                                continue
                            # print 'patching: %s-->%s = %f' % (patch_conn, patch_zone, patch[2])
                            inflZonesByConn[patch_conn][patch_zone] = patch[2]
                    else:
                        gridCol, gridRow = zone.getGridFromZoneLoc(z.zone_name)
                        if conn.edge != 3:
                            if distByCol:
                                if conn.edge == 0:
                                    k = 0.5
                                    # search for next intact connection in + dir
                                    value1 = self.redistribute_to_nearest_zone(
                                        z, range(gridRow + 1,
                                                 self.house.roof_rows),
                                        k, conn_type_group, gridCol,
                                        gridRow, distByCol)
                                    if not value1:
                                        k = 1.0
                                    value2 = self.redistribute_to_nearest_zone(
                                        z, reversed(range(0, gridRow)), k,
                                        conn_type_group, gridCol, gridRow,
                                        distByCol)
                                    if not value2:
                                        self.redistribute_to_nearest_zone(
                                            z, range(gridRow + 1,
                                                     self.house.roof_rows),
                                            k, conn_type_group, gridCol,
                                            gridRow, distByCol)
                                elif conn.edge == 2:
                                    k = 1.0
                                    self.redistribute_to_nearest_zone(
                                        z, range(gridRow + 1,
                                                 self.house.roof_rows),
                                        k, conn_type_group, gridCol, gridRow,
                                        distByCol)
                                elif conn.edge == 1:
                                    k = 1.0
                                    self.redistribute_to_nearest_zone(
                                        z, reversed(range(0, gridRow)), k,
                                        conn_type_group, gridCol, gridRow,
                                        distByCol)
                            else:
                                if conn.edge == 0:
                                    k = 0.5
                                    if not self.redistribute_to_nearest_zone(
                                            z, range(gridCol + 1,
                                                     self.house.roof_columns),
                                            k, conn_type_group, gridCol,
                                            gridRow, distByCol):
                                        k = 1.0
                                    if not self.redistribute_to_nearest_zone(
                                            z, reversed(range(0, gridCol)), k,
                                            conn_type_group, gridCol, gridRow,
                                            distByCol):
                                        self.redistribute_to_nearest_zone(
                                            z, range(gridCol + 1,
                                                     self.house.roof_columns),
                                            k, conn_type_group, gridCol,
                                            gridRow, distByCol)
                                elif conn.edge == 2:
                                    k = 1.0
                                    self.redistribute_to_nearest_zone(
                                        z, range(gridCol + 1,
                                                 self.house.roof_columns),
                                        k, conn_type_group, gridCol, gridRow,
                                        distByCol)
                                elif conn.edge == 1:
                                    k = 1.0
                                    self.redistribute_to_nearest_zone(
                                        z, reversed(range(0, gridCol)), k,
                                        conn_type_group, gridCol, gridRow,
                                        distByCol)

                    if conn_type_group.set_zone_to_zero > 0:
                        z.result_effective_area = 0.0
                    conn.result_damage_distributed = True

                    # print '{}:{}:{}'.format(i, j, k)

    @staticmethod
    def redistribute_to_nearest_zone(zoneSrc, connRange, k, ctgroup, gridCol,
                                     gridRow, distByCol):
        # return intact connection found
        # compute sampled_cpe and result_effective_area
        for line in connRange:

            if distByCol:
                r = line
                c = gridCol
            else:
                r = gridRow
                c = line

            zoneDest = zoneByLocationMap[zone.getZoneLocFromGrid(c, r)]
            conn = connByZoneTypeMap[zoneDest.zone_name].get(ctgroup.group_name)
            try:
                if not conn.result_damaged and zoneDest.result_effective_area > 0:
                    zoneDest.sampled_cpe = (
                                           zoneDest.result_effective_area * zoneDest.sampled_cpe +
                                           k * zoneSrc.result_effective_area * zoneSrc.sampled_cpe) / \
                                           (
                                           zoneDest.result_effective_area + k * zoneSrc.result_effective_area)
                    zoneDest.result_effective_area += k * zoneSrc.result_effective_area
                    return True
                if conn.edge > 0:
                    return False
            except BaseException, error:
                print '{}'.format(error)
        return False
    """

    # def plot_connection_damage(self, vRed, vBlue, id_sim):
    #     selected_ctg = {'sheeting', 'batten', 'rafter', 'piersgroup', 'Truss',
    #                     'wallracking'}.intersection(set(connByTypeGroupMap))
    #
    #     for ctg_name in selected_ctg:
    #         vgrid = vBlue * np.ones((self.house.roof_rows,
    #                                  self.house.roof_columns)) + 10.0
    #         file_name = os.path.join(self.cfg.output_path,
    #                                  '{}_id{}'.format(ctg_name, id_sim))
    #         for conn in connByTypeGroupMap[ctg_name]:
    #             gridCol, gridRow = \
    #                 zone.getGridFromZoneLoc(conn.location_zone.zone_name)
    #             if conn.result_failure_v > 0:
    #                 vgrid[gridRow][gridCol] = conn.result_failure_v
    #
    #         output.plot_damage_show(ctg_name, vgrid, self.house.roof_columns,
    #                                 self.house.roof_rows, vRed, vBlue, file_name)

            # if 'plot_wall_damage_show' in output.__dict__:
            #     wall_major_rows = 2
            #     wall_major_cols = self.house.roof_columns
            #     wall_minor_rows = 2
            #     wall_minor_cols = 8
            #
            #     for conn_type_group_name in ('wallcladding', 'wallcollapse'):
            #         if ctgMap.get(ctg_name, None) is None:
            #             continue
            #
            #         v_south_grid = np.ones((wall_major_rows, wall_major_cols),
            #                                dtype=np.float32) * vBlue + 10.0
            #         v_north_grid = np.ones((wall_major_rows, wall_major_cols),
            #                                dtype=np.float32) * vBlue + 10.0
            #         v_west_grid = np.ones((wall_minor_rows, wall_minor_cols),
            #                               dtype=np.float32) * vBlue + 10.0
            #         v_east_grid = np.ones((wall_minor_rows, wall_minor_cols),
            #                               dtype=np.float32) * vBlue + 10.0
            #
            #         # construct south grid
            #         for gridCol in range(0, wall_major_cols):
            #             for gridRow in range(0, wall_major_rows):
            #                 colChar = chr(ord('A')+gridCol)
            #                 loc = 'WS%s%d' % (colChar, gridRow+1)
            #                 conn = connByZoneTypeMap[loc].get(ctg_name)
            #                 if conn and conn.result_failure_v > 0:
            #                     v_south_grid[gridRow][gridCol] = \
            #                         conn.result_failure_v
            #
            #         # construct north grid
            #         for gridCol in range(0, wall_major_cols):
            #             for gridRow in range(0, wall_major_rows):
            #                 colChar = chr(ord('A')+gridCol)
            #                 loc = 'WN%s%d' % (colChar, gridRow+1)
            #                 conn = connByZoneTypeMap[loc].get(ctg_name)
            #                 if conn and conn.result_failure_v > 0:
            #                     v_north_grid[gridRow][gridCol] = \
            #                         conn.result_failure_v
            #
            #         # construct west grid
            #         for gridCol in range(0, wall_minor_cols):
            #             for gridRow in range(0, wall_minor_rows):
            #                 loc = 'WW%d%d' % (gridCol+2, gridRow+1)
            #                 conn = connByZoneTypeMap[loc].get(ctg_name)
            #                 if conn and conn.result_failure_v > 0:
            #                     v_west_grid[gridRow][gridCol] = \
            #                         conn.result_failure_v
            #
            #         # construct east grid
            #         for gridCol in range(0, wall_minor_cols):
            #             for gridRow in range(0, wall_minor_rows):
            #                 loc = 'WE%d%d' % (gridCol+2, gridRow+1)
            #                 conn = connByZoneTypeMap[loc].get(ctg_name)
            #                 if conn and conn.result_failure_v > 0:
            #                     v_east_grid[gridRow][gridCol] = \
            #                         conn.result_failure_v
            #
            #         output.plot_wall_damage_show(
            #             ctg_name,
            #             v_south_grid, v_north_grid, v_west_grid, v_east_grid,
            #             wall_major_cols, wall_major_rows,
            #             wall_minor_cols, wall_minor_rows,
            #             vRed, vBlue)

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
    parser.add_option("-v", "--verbose",
                      action="store_true",
                      dest="verbose",
                      default=False,
                      help="show verbose simulator output")
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
                                                             './output'))
    else:
        options.output_folder = os.path.abspath(
            os.path.join(os.getcwd(), options.output_folder))

    if options.verbose:
        logger.configure(logger.LOGGING_CONSOLE)
    else:
        logger.configure(logger.LOGGING_NONE)

    if options.data_folder:
        db = database.DatabaseManager(options.model_database,
                                      verbose=options.verbose)
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


if __name__ == '__main__':
    main()
