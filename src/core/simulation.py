# adjust python path so we may import things from peer packages
import sys
import os
import time
import parmap
import itertools
import pandas as pd
import numpy as np
from optparse import OptionParser
from version import VERSION_DESC

import terrain
import connection
import zone
import scenario
import curve
import curve_log
import dbimport
import logger
import database
import debris
import wateringress
import engine
from house import zoneByLocationMap, connByZoneTypeMap, ctgMap, \
    connByTypeGroupMap, inflZonesByConn, connByTypeMap, connByIDMap, zoneByIDMap


class CurvePlot(object):
    def __init__(self, coeff, method, xmin, xmax, label='_nolegend_', col='b'):
        self.coeff = coeff
        self.method = method
        self.x_arr = np.linspace(xmin, xmax, 500)
        self.label = label
        self.col = col

    def plot_vuln(self, faint=False):
        alpha = 0.3 if faint else 1.0
        col = 'r' if faint else 'b'
        if self.method == 'lognormal':
            obs = curve_log.generate_observations(self.coeff, self.x_arr)
            output.plot_fitted_curve(self.x_arr, obs, self.label, alpha, col)
        else:
            obs = curve.generate_observations(self.coeff, self.x_arr)
            output.plot_fitted_curve(self.x_arr, obs, self.label, alpha, col)

    def plot_frag(self, faint=False):
        alpha = 0.3 if faint else 1.0
        obs = curve_log.generate_observations(self.coeff, self.x_arr)
        output.plot_fragility_curve(self.x_arr, obs, self.label, alpha,
                                    self.col)


def simulate_wind_damage_to_house(cfg, options):

    # setup file based reporting (files must exist and be runnable)
    if not os.path.exists(options.output_folder):
        os.makedirs(options.output_folder)

    cfg.file_cpis = os.path.join(options.output_folder, 'house_cpi.csv')
    cfg.file_debris = os.path.join(options.output_folder, 'wind_debris.csv')
    cfg.file_damage = os.path.join(options.output_folder,'house_damage.csv')
    cfg.file_dmg = os.path.join(options.output_folder,
                                'houses_damaged_at_v.csv')
    cfg.file_frag = os.path.join(options.output_folder,'fragilities.csv')
    cfg.file_water = os.path.join(options.output_folder, 'wateringress.csv')

    cfg.cols = [chr(x) for x in range(ord('A'), ord('A') +
                                      cfg.house.roof_columns)]
    cfg.rows = range(1, cfg.house.roof_rows + 1)

    # # need to move to HouseDamage class
    # cfg.result_buckets = dict()
    # for item in ['mean', 'pressurized_count']:
    #     cfg.result_buckets[item] = pd.Series(
    #         0.0, index=range(cfg.wind_speed_num_steps))
    #
    # cfg.result_buckets['fragility'] = pd.DataFrame(
    #     0.0, index=range(cfg.wind_speed_num_steps),
    #     columns=cfg.fragility_thresholds.index)
    #
    # for item in ['dmg_index', 'debris', 'water_ingress', 'debris_nv',
    #              'debris_num', 'pressurized']:
    #     cfg.result_buckets[item] = pd.DataFrame(
    #         0.0, index=range(cfg.wind_speed_num_steps),
    #         columns=range(cfg.no_sims))

    # optionally seed random numbers
    if cfg.flags['random_seed']:
        np.random.seed(42)
        zone.seed_scipy(42)
        engine.seed(42)

    # setup speeds and buckets
    cfg.speeds = np.linspace(cfg.wind_speed_min,
                             cfg.wind_speed_max,
                             cfg.wind_speed_num_steps)

    # simulator main_loop
    # should be class level
    cfg.house.clear_sim_results()
    calculate_connection_group_areas(cfg)

    # should be class level
    if cfg.flags['debris']:
        cfg.debris_manager = debris.DebrisManager(
            cfg.house,
            cfg.region,
            cfg.wind_speed_min,
            cfg.wind_speed_max,
            cfg.wind_speed_num_steps,
            cfg.flags['debris_staggered_sources'],
            cfg.debris_radius,
            cfg.debris_angle,
            cfg.debris_extension,
            cfg.building_spacing,
            cfg.source_items,
            cfg.flight_time_mean,
            cfg.flight_time_stddev)

    tic = time.time()

    #parmap.map(run_simulation_per_house, lines)
    list_house_damage = [HouseDamage(cfg) for id_sim in range(cfg.no_sims)]

    # collection = dict()
    # for id_sim in range(cfg.no_sims):
    #     collection[id_sim] = HouseDamage(cfg)
    #     run_simulation_per_house(collection[id_sim], cfg)

    list_results = parmap.map(run_simulation_per_house, list_house_damage, cfg)

    print('{}'.format(time.time()-tic))


    # post processing of results (aggregations)
    # write debris output file
    mean_debris = pd.concat([x.result_buckets['debris']
                             for x in collection.itervalues()],
                            axis=1).mean(axis=1) * 100.0

    perc = pd.concat([x.result_buckets['pressurized']
                      for x in collection.itervalues()], axis=1).sum(axis=1)\
           / float(cfg.no_sims) * 100.0

    pd.concat([pd.Series(cfg.speeds), perc, mean_debris], axis=1).to_csv(
        cfg.file_debris, index=False, header=False, float_format='%.3f')

    # calculate and store DI mean
    df_dmg_index = pd.concat([x.result_buckets['dmg_index']
                               for x in collection.itervalues()],
                               axis=1)
    mean_dmg_index = df_dmg_index.mean(axis=1)

    # calculate damage probability
    counted = dict()
    for state, value in cfg.fragility_thresholds.iterrows():
        counted[state] = (df_dmg_index > value['threshold']).sum(axis=1)\
                         / float(cfg.no_sims)
    counted = pd.DataFrame.from_dict(counted)

    # cleanup: close output files
    cfg.file_cpis.close()
    cfg.file_debris.close()
    cfg.file_damage.close()
    cfg.file_dmg.close()
    cfg.file_water.close()

    '''


    # produce damage map report
    cfg.file_dmg.write('Number of Damaged Houses\n')
    cfg.file_dmg.write('Num Houses,%d\n' % cfg.no_sims)
    # cfg.file_dmg.write('Wind Direction,%s\n' % )
    cfg.file_dmg.write('Wind Speed(m/s)')

    # setup headers and counts
    str_ = [conn_type.connection_type for conn_type in
            cfg.house.conn_types]
    cfg.file_dmg.write(','.join(str_))
    cfg.file_dmg.write('\n')

    # we need to count houses damaged by type for each v
    counts = {}
    for wind_speed in self.cfg.speeds:
        self.cfg.file_dmg.write(str(wind_speed))

        # initialise damage counts for each conn_type to zero
        for conn_type in self.cfg.house.conn_types:
            counts[conn_type.connection_type] = 0

        # for all houses, increment type counts
        # if wind_speed exceeds minimum observed damages.
        for hr in house_results:
            dmg_map = hr[1]
            for conn_type in self.cfg.house.conn_types:
                dmg_min = dmg_map[conn_type.connection_type]
                if wind_speed >= dmg_min:
                    counts[conn_type.connection_type] += 1

        # write accumulated counts for this wind speed
        str_ = [str(counts[conn_type.connection_type]) for conn_type
                in self.cfg.house.conn_types]
        self.cfg.file_dmg.write(','.join(str_))
        self.cfg.file_dmg.write('\n')

    # cleanup: close output files
    cfg.file_cpis.close()
    cfg.file_debris.close()
    cfg.file_damage.close()
    cfg.file_dmg.close()
    cfg.file_water.close()
    #self.debrisManager = None

    self.fit_fragility_curves()
    self.cfg.file_frag.close()
    runTime = time.time() - tic
    '''
    return

# # move to class level
# def set_wind_direction(cfg):
#     wind_dir_index = cfg.get_wind_dir_index()
#     if cfg.debris_manager:
#         cfg.debris_manager.set_wind_direction_index(wind_dir_index)
#     return wind_dir_index


def calculate_connection_group_areas(cfg):
    for ctg in cfg.house.conn_type_groups:
        ctg.result_area = 0.0
    for c in cfg.house.connections:
        c.ctype.group.result_area += c.ctype.costing_area


def run_simulation_per_house(house_damage, cfg):

    # sample new house and wind direction (if random)
    if cfg.wind_dir_index == 8:
        house_damage.wind_orientation = cfg.get_wind_dir_index()
    else:
        house_damage.wind_orientation = cfg.wind_dir_index

    if cfg.debris_manager:
        house_damage.cfg.debris_manager.set_wind_direction_index(
            house_damage.wind_orientation)

    house_damage.sample_house_and_wind_params()

    # prime damage map where we track min() V that damage occurs
    # across types for this house (reporting)
    house_damage.dmg_map = {}
    for conn_type in cfg.house.conn_types:
        house_damage.dmg_map[conn_type.connection_type] = 99999

    # iteration over wind speed list
    for id_speed, wind_speed in enumerate(cfg.speeds):

        # simulate sampled house
        house_damage.clear_loop_results()
        house_damage.run_simulation(wind_speed)

        # collect results
        house_damage.result_buckets['water_ingress'][id_speed] = \
            house_damage.water_ingress_cost

        house_damage.result_buckets['dmg_index'][id_speed] = house_damage

        if cfg.flags['debris']:
            house_damage.result_buckets['debris'][id_speed] = \
                house_damage.cfg.debris_manager.result_dmgperc
            house_damage.result_buckets['debris_nv'][id_speed] = \
                house_damage.cfg.debris_manager.result_nv
            house_damage.result_buckets['debris_num'][id_speed] = \
                house_damage.cfg.debris_manager.result_num_items

        # for all houses, count the number that were pressurized at
        # this wind_speed
        if house_damage.cpi != 0:
            house_damage.result_buckets['pressurized'][id_speed] = True

        # # interact with GUI listener
        # if self.diCallback:
        #     currentLoop += 1
        #     percLoops = (float(currentLoop) / float(totalLoops)) * 100.0
        #     keep_looping = self.diCallback(wind_speed, self.di, percLoops)
        #     if not keep_looping:
        #         break

    # collect results to be used by the GUI client
    for z in house_damage.cfg.house.zones:
        house_damage.zone_results[z.zone_name] = [z.zone_name,
                                          z.sampled_cpe,
                                          z.sampled_cpe_struct,
                                          z.sampled_cpe_eaves]

    for c in house_damage.cfg.house.connections:
        house_damage.conn_results.append([c.ctype.connection_type,
                                  c.location_zone.zone_name,
                                  c.result_failure_v_raw,
                                  c.result_strength,
                                  c.result_deadload,
                                  c.result_damaged_report,
                                  c.ctype.group.group_name,
                                  c.id])
    return house_damage


class HouseDamage(object):
    """
    WindDamageSimulator: Stores sampled state (PDF) for house and wind for
    current simulation loop

    # results (for v) stored in dictionary bucket keyed with wind_speed
    # each entry has list form: (FLD_MEAN, [FLD_DIARRAY], [FLD_FRAGILITIES],
    # FLD_PRESSURIZED_COUNT, [FLD_DEBRIS_AT])
    # subarrays are indexed by house(iteration)

    """

    id_sim_gen = itertools.count()

    def __init__(self, cfg, diCallback=None, mplDict=None):
        self.cfg = cfg
        self.id_sim = next(self.id_sim_gen)

        # self.debrisManager = None
        self.A_final = None
        self.diCallback = diCallback
        self.wind_orientation = 0
        self.mzcat = 0
        self.di = 0
        self.mplDict = mplDict
        self.prevCurvePlot = None

        # Undefined and later added
        self.result_wall_collapse = None

        # self.cols = None
        # self.rows = None

        self.qz = 0.0
        self.Ms = 1.0
        self.di = 0.0
        self.fragilities = []
        self.profile = None
        self.mzcat = None
        self.cpiAt = None
        self.cpi = None
        self.internally_pressurized = None
        self.construction_level = None
        self.cpiAt = None
        self.prev_di = None
        self.water_ingress_cost = None
        self.speeds = None
        self.dmg_map = None
        self.frag_levels = None
        self.wind_speeds = None
        self.di_means = None
        self.ss = None
        self.zone_results = dict()
        self.conn_results = []

        self.result_buckets = dict()
        for item in ['dmg_index', 'debris', 'water_ingress', 'debris_nv',
                     'debris_num', 'pressurized']:
            self.result_buckets[item] = pd.Series(
                0.0, index=range(cfg.wind_speed_num_steps))

        # self.set_scenario()

        # global output
        # if mplDict is not None:
        #     output = __import__('gui.output').output
        #     output.hookupWidget(mplDict)
        #     self.mplDict = mplDict
        # else:
        #     output = __import__('core.output').output

        # terrain.populate_wind_profile_by_terrain()
        # self.clear_loop_results()

    # @staticmethod
    # def set_fragility_thresholds(thresholds):
    #     # thresholds is a dict in form {'slight': 0.0}
    #
    #     df_frag = pd.DataFrame.from_dict(thresholds, orient='index')
    #     df_frag.columns = ['threshold']
    #     df_frag['color'] = ['b', 'g', 'y', 'r']
    #     df_frag['object'] = [None, None, None, None]
    #     return df_frag

    # def set_scenario(self):

        # self.cfg.house.clear_sim_results()

    # def simulator_mainloop(self):

        # tic = time.time()

        # for wind_speed in self.speeds:
        #     self.result_buckets[wind_speed] = [0., [], [], 0, [], [], [], []]

        # setup connections and groups
        # self.cfg.house.clear_sim_results()
        # self.calculate_connection_group_areas()

        # optionally create the debris manager and
        # make sure a wind orientation is set

        # bDebris = self.cfg.flags['debris']
        # if self.cfg.flags['debris']:
        #     self.debrisManager = debris.DebrisManager(
        #         self.cfg.house,
        #         self.cfg.region,
        #         self.cfg.wind_speed_min,
        #         self.cfg.wind_speed_max,
        #         self.cfg.wind_speed_num_steps,
        #         self.cfg.flags['debris_staggered_sources'],
        #         self.cfg.debris_radius,
        #         self.cfg.debris_angle,
        #         self.cfg.debris_extension,
        #         self.cfg.building_spacing,
        #         self.cfg.source_items,
        #         self.cfg.flight_time_mean,
        #         self.cfg.flight_time_stddev)
        # self.wind_orientation = set_wind_direction(self.cfg)

        # # gui bookkeeping
        # if self.diCallback:
        #     totalLoops = self.cfg.no_sims * len(self.cfg.speeds)
        #     currentLoop = 1

        # SIMULATE HOUSES
        # house_results = []
        # keep_looping = True

        # iteration over samples
        # for id_sim in range(self.cfg.no_sims):
        #     if not keep_looping:
        #         break

        # # sample new house and wind direction (if random)
        # if self.cfg.wind_dir_index == 8:
        #     self.wind_orientation = self.cfg.get_wind_dir_index()
        # else:
        #     self.wind_orientation = self.cfg.wind_dir_index
        #
        # if self.cfg.debris_manager:
        #     self.cfg.debris_manager.set_wind_direction_index(
        #         self.wind_orientation)
        #
        # self.sample_house_and_wind_params()
        #
        # # prime damage map where we track min() V that damage occurs
        # # across types for this house (reporting)
        # self.dmg_map = {}
        # for conn_type in self.cfg.house.conn_types:
        #     self.dmg_map[conn_type.connection_type] = 99999
        #
        # # iteration over wind speed list
        # for id_speed, wind_speed in enumerate(self.cfg.speeds):
        #
        #     # simulate sampled house
        #     self.clear_loop_results()
        #     self.run_simulation(wind_speed)
        #
        #     # collect results
        #     self.result_buckets['water_ingress'][id_speed] = \
        #         self.water_ingress_cost
        #
        #     self.result_buckets['dmg_index'][id_speed] = self.di
        #
        #     if self.cfg.flags['debris']:
        #         self.result_buckets['debris'][id_speed] = \
        #             self.cfg.debris_manager.result_dmgperc
        #         self.result_buckets['debris_nv'][id_speed] = \
        #             self.cfg.debris_manager.result_nv
        #         self.result_buckets['debris_num'][id_speed] = \
        #             self.cfg.debris_manager.result_num_items
        #
        #     # for all houses, count the number that were pressurized at
        #     # this wind_speed
        #     if self.cpi != 0:
        #         self.result_buckets['pressurized'][id_speed] = True
        #
        #     # # interact with GUI listener
        #     # if self.diCallback:
        #     #     currentLoop += 1
        #     #     percLoops = (float(currentLoop) / float(totalLoops)) * 100.0
        #     #     keep_looping = self.diCallback(wind_speed, self.di, percLoops)
        #     #     if not keep_looping:
        #     #         break
        #
        # # collect results to be used by the GUI client
        # for z in self.cfg.house.zones:
        #     self.zone_results[z.zone_name] = [z.zone_name,
        #                                       z.sampled_cpe,
        #                                       z.sampled_cpe_struct,
        #                                       z.sampled_cpe_eaves]
        #
        # for c in self.cfg.house.connections:
        #     self.conn_results.append([c.ctype.connection_type,
        #                               c.location_zone.zone_name,
        #                               c.result_failure_v_raw,
        #                               c.result_strength,
        #                               c.result_deadload,
        #                               c.result_damaged_report,
        #                               c.ctype.group.group_name,
        #                               c.id])

        # house_results.append([zone_results,
        #                       self.dmg_map,
        #                       self.wind_orientation,
        #                       self.cpiAt,
        #                       conn_results,
        #                       self.construction_level])

        # if keep_looping:
        #     # post processing of results (aggregations)
        #     for id_speed, wind_speed in enumerate(self.cfg.speeds):
        #
        #         # write debris output file
        #         mean_debris = self.cfg.result_buckets['debris'].loc[id_speed].mean()
        #         perc = self.cfg.result_buckets['pressurized_count'].loc[id_speed]\
        #                / float(self.cfg.no_sims) * 100.0
        #         self.cfg.file_debris.write('{:.3f},{:.3f},{:.3f}\n'.format(
        #             wind_speed, perc, mean_debris * 100.0))
        #
        #         # calculate and store DI mean
        #         self.cfg.result_buckets['mean'].loc[id_speed] = \
        #             self.cfg.result_buckets['dmg_index'].loc[id_speed].mean()
        #
        #         # calculate damage probability
        #         for state, value in self.cfg.fragility_thresholds.iterrows():
        #             self.cfg.result_buckets['fragility'].loc[id_speed, state] = \
        #                 (self.cfg.result_buckets['dmg_index'].loc[id_speed]
        #                  > value['threshold']).sum() / float(self.cfg.no_sims)
        #
        # # produce damage map report
        # self.cfg.file_dmg.write('Number of Damaged Houses\n')
        # self.cfg.file_dmg.write('Num Houses,%d\n' % self.cfg.no_sims)
        # self.cfg.file_dmg.write('Wind Direction,%s\n' % scenario.Scenario.dirs[self.wind_orientation])
        # self.cfg.file_dmg.write('Wind Speed(m/s)')
        #
        # # setup headers and counts
        # str_ = [conn_type.connection_type for conn_type in
        #         self.cfg.house.conn_types]
        # self.cfg.file_dmg.write(','.join(str_))
        # self.cfg.file_dmg.write('\n')
        #
        # # we need to count houses damaged by type for each v
        # counts = {}
        # for wind_speed in self.cfg.speeds:
        #     self.cfg.file_dmg.write(str(wind_speed))
        #
        #     # initialise damage counts for each conn_type to zero
        #     for conn_type in self.cfg.house.conn_types:
        #         counts[conn_type.connection_type] = 0
        #
        #     # for all houses, increment type counts
        #     # if wind_speed exceeds minimum observed damages.
        #     for hr in house_results:
        #         dmg_map = hr[1]
        #         for conn_type in self.cfg.house.conn_types:
        #             dmg_min = dmg_map[conn_type.connection_type]
        #             if wind_speed >= dmg_min:
        #                 counts[conn_type.connection_type] += 1
        #
        #     # write accumulated counts for this wind speed
        #     str_ = [str(counts[conn_type.connection_type]) for conn_type
        #             in self.cfg.house.conn_types]
        #     self.cfg.file_dmg.write(','.join(str_))
        #     self.cfg.file_dmg.write('\n')
        #
        # # cleanup: close output files
        # self.cfg.file_cpis.close()
        # self.cfg.file_debris.close()
        # self.cfg.file_damage.close()
        # self.cfg.file_dmg.close()
        # self.cfg.file_water.close()
        # #self.debrisManager = None
        #
        # if keep_looping:
        #     self.fit_fragility_curves()
        #     self.cfg.file_frag.close()
        #     runTime = time.time() - tic
        #     return runTime, house_results
        # else:
        #     self.cfg.file_frag.close()
        #     return None, None

    def run_simulation(self, wind_speed):
        self.check_pressurized_failure(wind_speed)

        self.calculate_qz(wind_speed)
        zone.calc_zone_pressures(self.cfg.house.zones,
                                 self.wind_orientation,
                                 self.cpi,
                                 self.qz,
                                 self.Ms,
                                 self.cfg.building_spacing,
                                 self.cfg.flags['diff_shielding'])

        # self.cfg.file_damage.write('%d,%.3f,%s' % (self.id_sim + 1,
        #                                        wind_speed,
        #                                       scenario.Scenario.dirs[self.wind_orientation]))
        for ctg in self.cfg.house.conn_type_groups:
            connection.calc_connection_loads(wind_speed,
                                             ctg,
                                             self.cfg.house,
                                             self.cfg.file_damage,
                                             self.dmg_map,
                                             inflZonesByConn,
                                             connByTypeMap)
        if self.cfg.flags['dmg_distribute']:
            for ctg in self.cfg.house.conn_type_groups:
                self.redistribute_damage(ctg)
        # self.cfg.file_damage.write('\n')

        self.check_house_collapse(wind_speed)
        self.calculate_damage_ratio(wind_speed)

    def clear_loop_results(self):
        self.qz = 0.0
        self.Ms = 1.0
        self.di = 0.0
        self.fragilities = []

    # def set_wind_direction(self):
    #     self.wind_orientation = self.cfg.get_wind_dir_index()
    #     if self.cfg.debris_manager:
    #         self.cfg.debris_manager.set_wind_direction_index(self.wind_orientation)

    def set_wind_profile(self):
        self.profile = np.random.random_integers(1, 10)
        self.mzcat = terrain.calculateMZCAT(self.cfg.wind_profile,
                                            self.cfg.terrain_category,
                                            self.profile,
                                            self.cfg.house.height)

    def calculate_qz(self, wind_speed):
        if self.cfg.regional_shielding_factor <= 0.85:
            thresholds = np.array([63, 63+15])
            ms_dic = {0: 1.0, 1: 0.85, 2: 0.95}
            idx = sum(thresholds <= np.random.random_integers(0, 100))
            self.Ms = ms_dic[idx]
            Vmod = (wind_speed * self.Ms) / self.cfg.regional_shielding_factor
            self.qz = 0.6 * 1.0e-3 * (Vmod * self.mzcat)**2
        else:
            self.qz = 0.6 * 1.0e-3 * (wind_speed * self.mzcat)**2

    def check_pressurized_failure(self, v):
        if self.cfg.flags['debris']:
            self.cfg.debris_manager.run(v)
            if self.cpi == 0 and self.cfg.debris_manager.get_breached():
                self.cpi = 0.7
                self.cpiAt = v
                # self.cfg.file_cpis.write('%d,%.3f\n' % (self.id_sim + 1, v))

    def sample_house_and_wind_params(self):
        self.cpi = 0
        self.cpiAt = 0
        self.internally_pressurized = False
        self.set_wind_profile()
        self.cfg.house.reset_results()
        self.prev_di = 0

        self.construction_level = 'na'
        mean_factor = 1.0
        cov_factor = 1.0
        if self.cfg.flags['construction_levels']:
            self.construction_level, mean_factor, cov_factor = \
                self.cfg.sampleConstructionLevel()

        # print('{}'.format(self.construction_level))
        connection.assign_connection_strengths(self.cfg.house.connections,
                                               mean_factor,
                                               cov_factor)

        connection.assign_connection_deadloads(self.cfg.house.connections)

        zone.sample_zone_pressures(self.cfg.house.zones,
                                   self.wind_orientation,
                                   self.cfg.house.cpe_V,
                                   self.cfg.house.cpe_k,
                                   self.cfg.house.cpe_struct_V)
        # we don't want to collapse multiple times (no need)
        self.result_wall_collapse = False

    def check_house_collapse(self, wind_speed):
        if not self.result_wall_collapse:
            for ctg in self.cfg.house.conn_type_groups:
                if ctg.trigger_collapse_at > 0:
                    perc_damaged = 0
                    for ct in ctg.conn_types:
                        perc_damaged += ct.perc_damaged()
                    if perc_damaged >= ctg.trigger_collapse_at:
                        for c in self.cfg.house.connections:
                            c.damage(wind_speed, 99.9, inflZonesByConn[c])
                        for z in self.cfg.house.zones:
                            z.result_effective_area = 0
                        self.result_wall_collapse = True


    def calculate_damage_ratio(self, wind_speed):

        # calculate damage percentages        
        for ctg in self.cfg.house.conn_type_groups:
            ctg.result_percent_damaged = 0.0
            if ctg.group_name == 'debris':
                if not self.cfg.debris_manager:
                    ctg.result_percent_damaged = 0
                else:
                    ctg.result_percent_damaged = \
                        self.cfg.debris_manager.result_dmgperc
            else:
                for ct in ctg.conn_types:
                    for c in ct.connections_of_type:
                        if c.result_damaged:
                            ctg.result_percent_damaged += \
                                c.ctype.costing_area / float(ctg.result_area)

        # calculate repair cost
        repair_cost = 0
        for ctg in self.cfg.house.conn_type_groups:
            ctg_perc = ctg.result_percent_damaged
            if ctg_perc > 0:
                fact_arr = [0]
                for factor in self.cfg.house.factorings:
                    if factor.parent_id == ctg.id:
                        factor_perc = factor.factor.result_percent_damaged
                        if factor_perc:
                            fact_arr.append(factor_perc)
                max_factor_perc = max(fact_arr)
                if ctg_perc > max_factor_perc:
                    ctg_perc = ctg_perc - max_factor_perc
                    repair_cost += ctg.costing.calculate_damage(ctg_perc)

        # calculate initial envelope repair cost before water ingress is added
        self.di = repair_cost / self.cfg.house.replace_cost
        if self.di > 1.0:
            self.di = 1.0
        else:
            self.water_ingress_cost = 0
            if self.cfg.flags['water_ingress']:
                self.water_ingress_cost = \
                    wateringress.get_costing_for_envelope_damage_at_v(
                        self.di,
                        wind_speed,
                        self.cfg.house.water_groups,
                        self.cfg.file_water)
                repair_cost += self.water_ingress_cost

        # combined internal + envelope damage costing can now be calculated
        self.di = repair_cost / self.cfg.house.replace_cost
        if self.di > 1.0:
            self.di = 1.0
        self.prev_di = self.di

    def redistribute_damage(self, ctg):
        # setup for distribution
        if ctg.distribution_order <= 0:
            return
        distByCol = ctg.distribution_direction == 'col'
        primaryDir = self.cfg.cols
        secondaryDir = self.cfg.rows
        if not distByCol:
            primaryDir = self.cfg.rows
            secondaryDir = self.cfg.cols

        # walk the zone grid for current group
        # (only one conn of each group per zone)
        for i in primaryDir:
            for j in secondaryDir:
                # determine zoneLocation and then zone
                zoneLoc = i if distByCol else j
                zoneLoc += str(j) if distByCol else str(i)

                # not all grid locations have a zone
                if zoneLoc not in zoneByLocationMap:
                    continue

                # not all zones have area or connections remaining
                z = zoneByLocationMap[zoneLoc]
                if z.result_effective_area == 0.0 or len(z.located_conns) == 0:
                    continue

                # not all zones have connections of all types
                if ctg.group_name not in connByZoneTypeMap[z.zone_name]:
                    continue

                # grab appropriate connection from zone
                c = connByZoneTypeMap[z.zone_name][ctg.group_name]

                # if that connection is (newly) damaged then redistribute
                # load/infl/area
                if c.result_damaged and not c.result_damage_distributed:
                    # print 'Connection: %s newly damaged' % c

                    if ctg.patch_distribution == 1:
                        patchList = \
                            database.db.qryConnectionPatchesFromDamagedConn(c.id)

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
                        if c.edge != 3:
                            if distByCol:
                                if c.edge == 0:
                                    k = 0.5
                                    if not self.redistribute_to_nearest_zone(z, range(gridRow+1, self.cfg.house.roof_rows), k, ctg, gridCol, gridRow, distByCol):
                                        k = 1.0
                                    if not self.redistribute_to_nearest_zone(z, reversed(range(0, gridRow)), k, ctg, gridCol, gridRow, distByCol):
                                        self.redistribute_to_nearest_zone(z, range(gridRow+1, self.cfg.house.roof_rows), k, ctg, gridCol, gridRow, distByCol)
                                elif c.edge == 2:
                                    k = 1.0
                                    self.redistribute_to_nearest_zone(z, range(gridRow+1, self.cfg.house.roof_rows), k, ctg, gridCol, gridRow, distByCol)
                                elif c.edge == 1:
                                    k = 1.0
                                    self.redistribute_to_nearest_zone(z, reversed(range(0, gridRow)), k, ctg, gridCol, gridRow, distByCol)
                            else:
                                if c.edge == 0:
                                    k = 0.5
                                    if not self.redistribute_to_nearest_zone(z, range(gridCol+1, self.cfg.house.roof_columns), k, ctg, gridCol, gridRow, distByCol):
                                        k = 1.0
                                    if not self.redistribute_to_nearest_zone(z, reversed(range(0, gridCol)), k, ctg, gridCol, gridRow, distByCol):
                                        self.redistribute_to_nearest_zone(z, range(gridCol+1, self.cfg.house.roof_columns), k, ctg, gridCol, gridRow, distByCol)
                                elif c.edge == 2:
                                    k = 1.0
                                    self.redistribute_to_nearest_zone(z, range(gridCol+1, self.cfg.house.roof_columns), k, ctg, gridCol, gridRow, distByCol)
                                elif c.edge == 1:
                                    k = 1.0
                                    self.redistribute_to_nearest_zone(z, reversed(range(0, gridCol)), k, ctg, gridCol, gridRow, distByCol)

                    if ctg.set_zone_to_zero > 0:
                        z.result_effective_area = 0.0
                    c.result_damage_distributed = True

    @staticmethod
    def redistribute_to_nearest_zone(zoneSrc, connRange, k, ctgroup, gridCol,
                                     gridRow, distByCol):
        for line in connRange:
            r = line
            c = gridCol
            if not distByCol:
                r = gridRow
                c = line
            zoneDest = zoneByLocationMap[zone.getZoneLocFromGrid(c, r)]
            conn = connByZoneTypeMap[zoneDest.zone_name].get(ctgroup.group_name)
            if conn:
                if not conn.result_damaged and zoneDest.result_effective_area > 0:
                    zoneDest.sampled_cpe = ((zoneDest.result_effective_area * zoneDest.sampled_cpe) +
                                            (k * zoneSrc.result_effective_area * zoneSrc.sampled_cpe)) / (zoneDest.result_effective_area +
                                            (k * zoneSrc.result_effective_area))
                    zoneDest.result_effective_area = zoneDest.result_effective_area + (k * zoneSrc.result_effective_area)
                    return True
                if conn.edge > 0:
                    return False
        return False

    def get_windresults_perc_houses_breached(self):
        breaches = []
        for id_speed, wind_speed in enumerate(self.cfg.speeds):
            perc = self.cfg.result_buckets['pressurized_count'].loc[id_speed] \
                   / float(self.cfg.no_sims) * 100.0
            breaches.append(perc)
        return self.cfg.speeds, breaches

    def get_windresults_samples_perc_debris_damage(self):
        samples = {}
        for wind_speed in self.cfg.speeds:
            samples[wind_speed] = self.cfg.result_buckets[wind_speed][type(self).FLD_DEBRIS_AT]
        return self.cfg.speeds, samples

    def get_windresults_samples_nv(self):
        samples = {}
        for wind_speed in self.cfg.speeds:
            samples[wind_speed] = self.cfg.result_buckets[wind_speed][type(self).FLD_DEBRIS_NV_AT]
        return self.cfg.speeds, samples

    def get_windresults_samples_num_items(self):
        samples = {}
        for wind_speed in self.cfg.speeds:
            samples[wind_speed] = self.cfg.result_buckets[wind_speed][type(self).FLD_DEBRIS_NUM_AT]
        return self.cfg.speeds, samples

    def get_windresults_samples_perc_water_ingress(self):
        samples = {}
        for wind_speed in self.cfg.speeds:
            samples[wind_speed] = self.cfg.result_buckets[wind_speed][type(self).FLD_WI_AT]
        return self.cfg.speeds, samples

    def plot_connection_damage(self, vRed, vBlue):
        for ctg_name in ['sheeting', 'batten', 'rafter', 'piersgroup',
                         'wallracking', 'Truss']:
            if ctgMap.get(ctg_name, None) is None:
                continue
            vgrid = np.ones((self.cfg.house.roof_rows, self.cfg.house.roof_columns),
                            dtype=np.float32) * vBlue + 10.0
            for conn in connByTypeGroupMap[ctg_name]:
                gridCol, gridRow = \
                    zone.getGridFromZoneLoc(conn.location_zone.zone_name)
                if conn.result_failure_v > 0:
                    vgrid[gridRow][gridCol] = conn.result_failure_v
            output.plot_damage_show(ctg_name, vgrid, self.cfg.house.roof_columns,
                                    self.cfg.house.roof_rows, vRed, vBlue)

        if 'plot_wall_damage_show' in output.__dict__:
            wall_major_rows = 2
            wall_major_cols = self.cfg.house.roof_columns
            wall_minor_rows = 2
            wall_minor_cols = 8

            for ctg_name in ('wallcladding', 'wallcollapse'):
                if ctgMap.get(ctg_name, None) is None:
                    continue

                v_south_grid = np.ones((wall_major_rows, wall_major_cols),
                                       dtype=np.float32) * vBlue + 10.0
                v_north_grid = np.ones((wall_major_rows, wall_major_cols),
                                       dtype=np.float32) * vBlue + 10.0
                v_west_grid = np.ones((wall_minor_rows, wall_minor_cols),
                                      dtype=np.float32) * vBlue + 10.0
                v_east_grid = np.ones((wall_minor_rows, wall_minor_cols),
                                      dtype=np.float32) * vBlue + 10.0

                # construct south grid
                for gridCol in range(0, wall_major_cols):
                    for gridRow in range(0, wall_major_rows):
                        colChar = chr(ord('A')+gridCol)
                        loc = 'WS%s%d' % (colChar, gridRow+1)
                        conn = connByZoneTypeMap[loc].get(ctg_name)
                        if conn and conn.result_failure_v > 0:
                            v_south_grid[gridRow][gridCol] = \
                                conn.result_failure_v

                # construct north grid
                for gridCol in range(0, wall_major_cols):
                    for gridRow in range(0, wall_major_rows):
                        colChar = chr(ord('A')+gridCol)
                        loc = 'WN%s%d' % (colChar, gridRow+1)
                        conn = connByZoneTypeMap[loc].get(ctg_name)
                        if conn and conn.result_failure_v > 0:
                            v_north_grid[gridRow][gridCol] = \
                                conn.result_failure_v

                # construct west grid
                for gridCol in range(0, wall_minor_cols):
                    for gridRow in range(0, wall_minor_rows):
                        loc = 'WW%d%d' % (gridCol+2, gridRow+1)
                        conn = connByZoneTypeMap[loc].get(ctg_name)
                        if conn and conn.result_failure_v > 0:
                            v_west_grid[gridRow][gridCol] = \
                                conn.result_failure_v

                # construct east grid
                for gridCol in range(0, wall_minor_cols):
                    for gridRow in range(0, wall_minor_rows):
                        loc = 'WE%d%d' % (gridCol+2, gridRow+1)
                        conn = connByZoneTypeMap[loc].get(ctg_name)
                        if conn and conn.result_failure_v > 0:
                            v_east_grid[gridRow][gridCol] = \
                                conn.result_failure_v

                output.plot_wall_damage_show(
                    ctg_name,
                    v_south_grid, v_north_grid, v_west_grid, v_east_grid,
                    wall_major_cols, wall_major_rows,
                    wall_minor_cols, wall_minor_rows,
                    vRed, vBlue)


    # This needs to be done outside of the plotting function
    # as these coefficients are
    # the final output of this program in batch... they are all that matters.
    #
    def fit_fragility_curves(self):
        # unpack the fragility means into separate arrays for fit/plot.
        self.frag_levels = []
        coeff_arr = []
        for frag_ind, (state, value) in enumerate(
                self.cfg.fragility_thresholds.iterrows()):

            self.frag_levels.append(np.zeros(len(self.cfg.speeds)))

            for id_speed, wind_speed in enumerate(self.cfg.speeds):
                self.frag_levels[frag_ind][id_speed] = \
                    self.cfg.result_buckets['fragility'].loc[id_speed, state]

            try:
                coeff_arr, ss = curve_log.fit_curve(self.cfg.speeds,
                                                    self.frag_levels[frag_ind],
                                                    False)
            except Exception:
                msg = 'fit_fragility_curves failed: {}'.format(coeff_arr)
                print(msg)

            else:

                if frag_ind > 0:
                    # self.cfg.file_frag.write(',')
                    pass
                # self.cfg.file_frag.write('{:f},{:f}'.format(coeff_arr[0],
                #                                         coeff_arr[1]))
                # label = '{}({:.2f})'.format(state, value['threshold'])
                self.cfg.fragility_thresholds.loc[state, 'object'] = \
                    CurvePlot(coeff_arr,
                              'lognormal',
                              self.cfg.wind_speed_min,
                              self.cfg.wind_speed_max,
                              label,
                              self.cfg.fragility_thresholds.loc[state, 'color'])

        # self.cfg.file_frag.write('\n')

    def plot_fragility(self, output_folder):
        for frag_ind, (state, value) in \
                self.cfg.fragility_thresholds.iterrows():
            output.plot_fragility_curve(self.cfg.speeds,
                                        self.frag_levels[frag_ind],
                                        '_nolegend_',
                                        0.3,
                                        self.cfg.fragility_thresholds.loc[state, 'color'])
            plot_obj = self.cfg.fragility_thresholds.loc[state, 'object']
            if plot_obj:
                plot_obj.plot_frag()

        output.plot_fragility_show(self.cfg.no_sims,
                                   self.cfg.wind_speed_min,
                                   self.cfg.wind_speed_max, output_folder)

    def fit_vuln_curve(self):
        self.wind_speeds = np.zeros(len(self.cfg.speeds))
        self.di_means = np.zeros(len(self.cfg.speeds))

        ss = 0
        for i, wind_speed in enumerate(self.cfg.speeds):
            self.wind_speeds[i] = wind_speed
            self.di_means[i] = self.cfg.result_buckets[wind_speed][type(self).FLD_MEAN]

        if self.cfg.flags['vul_fig_log']:
            self.A_final, self.ss = curve_log.fit_curve(self.wind_speeds,
                                                        self.di_means)
        else:
            self.A_final, self.ss = curve.fit_curve(self.wind_speeds,
                                                    self.di_means)

    def plot_vulnerability(self, output_folder, label="Fitted Curve"):
        # fit current observations
        self.fit_vuln_curve()

        # plot means
        if self.cfg.no_sims <= 100:
            for wind_speed in self.cfg.speeds:
                damage_indexes = self.cfg.result_buckets[wind_speed][type(self).FLD_DIARRAY]
                output.plot_wind_event_damage([wind_speed]*len(damage_indexes),
                                              damage_indexes)
        output.plot_wind_event_mean(self.wind_speeds, self.di_means)

        # plot fitted curve (with previous dimmed red)
        if self.cfg.flags['vul_fit_log']:
            fn_form = 'lognormal'
        else:
            fn_form = 'original'

        cp = CurvePlot(self.A_final,
                       fn_form,
                       self.cfg.wind_speed_min,
                       self.cfg.wind_speed_max,
                       "Fitted Curve")

        cp.plot_vuln()
        if self.prevCurvePlot:
            self.prevCurvePlot.plot_vuln(True)
        self.prevCurvePlot = cp

    def show_results(self, output_folder=None, vRed=40, vBlue=80):
        if self.mplDict:
            self.mplDict['fragility'].axes.cla()
            self.mplDict['fragility'].axes.figure.canvas.draw()
            self.mplDict['vulnerability'].axes.cla()
            self.mplDict['vulnerability'].axes.figure.canvas.draw()
        if self.cfg.flags['dmg_plot_fragility']:
            self.plot_fragility(output_folder)
        if self.cfg.flags['dmg_plot_vul']:
            self.plot_vulnerability(output_folder)
            output.plot_wind_event_show(self.cfg.no_sims,
                                        self.cfg.wind_speed_min,
                                        self.cfg.wind_speed_max,
                                        output_folder)
        self.plot_connection_damage(vRed, vBlue)

    def clear_connection_damage(self):
        v = np.ones((self.cfg.house.roof_rows, self.cfg.house.roof_columns),
                    dtype=np.float32) * self.cfg.wind_speed_max
        for ctname in ['sheeting', 'batten', 'rafter', 'piersgroup',
                       'wallracking']:
            output.plot_damage_show(ctname, v, self.cfg.house.roof_columns,
                                    self.cfg.house.roof_rows,
                                    self.cfg.wind_speed_min,
                                    self.cfg.wind_speed_max)
        for ctname in ['wallcladding', 'wallcollapse']:
            wall_major_rows = 2
            wall_major_cols = self.cfg.house.roof_columns
            wall_minor_rows = 2
            wall_minor_cols = 8
            v_major_grid = np.ones((wall_major_rows, wall_major_cols),
                                   dtype=np.float32) * self.cfg.wind_speed_max
            v_minor_grid = np.ones((wall_minor_rows, wall_minor_cols),
                                   dtype=np.float32) * self.cfg.wind_speed_max
            output.plot_wall_damage_show(
                ctname,
                v_major_grid, v_major_grid, v_minor_grid, v_minor_grid,
                wall_major_cols, wall_major_rows,
                wall_minor_cols, wall_minor_rows,
                self.cfg.wind_speed_min, self.cfg.wind_speed_max)


lastiPerc = -1


def simProgressCallback(V, di, percLoops):
    global lastiPerc
    iPerc = int(percLoops)
    if iPerc != lastiPerc:
        lastiPerc = iPerc
        sys.stdout.write('.')
    return True


# @profile
def simulate(cfg, options):
    if options.verbose:
        arg = simProgressCallback
    else:
        arg = None
    mySim = WindDamageSimulator(cfg, options, arg, None)
    # mySim.set_scenario(cfg)
    runTime, hr = mySim.simulator_mainloop(options.verbose)
    if runTime:
        if options.plot_frag:
            mySim.plot_fragility(options.output_folder)
        if options.plot_vuln:
            mySim.plot_vulnerability(options.output_folder, None)
            output.plot_wind_event_show(cfg.no_sims,
                                        cfg.wind_speed_min,
                                        cfg.wind_speed_max,
                                        options.output_folder)
    return runTime


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

    if options.model_database is None:
        model_db = None
    else:
        model_db = os.path.abspath(os.path.join(os.getcwd(),
                                                options.model_database))
    if options.output_folder is None:
        options.output_folder = os.path.abspath(os.path.join(path_,
                                                             './outputs'))
    else:
        options.output_folder = os.path.abspath(os.path.join(
            os.getcwd(), options.output_folder))
    print 'output directory: %s' % options.output_folder

    if options.verbose:
        logger.configure(logger.LOGGING_CONSOLE)
    else:
        logger.configure(logger.LOGGING_NONE)

    if options.data_folder:
        print ('Importing database from folder: {} '
               'to: {}').format(options.data_folder, options.model_database)

        database.configure(model_db, flag_make=True)
        dbimport.import_model(options.data_folder, options.model_database)
        database.db.close()
        return

    if options.scenario_filename:
        database.configure(model_db)
        conf = scenario.loadFromCSV(options.scenario_filename)
        simulate(conf, options)
        database.db.close()
    else:
        print '\nERROR: Must provide as scenario file to run simulator...\n'
        parser.print_help()


if __name__ == '__main__':
    main()
