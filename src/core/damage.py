# adjust python path so we may import things from peer packages
import sys
import os
import numpy as np
import pandas as pd
import datetime
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


class WindDamageSimulator(object):
    """
    WindDamageSimulator: Stores sampled state (PDF) for house and wind for
    current simulation loop

    # results (for v) stored in dictionary bucket keyed with wind_speed
    # each entry has list form: (FLD_MEAN, [FLD_DIARRAY], [FLD_FRAGILITIES],
    # FLD_PRESSURIZED_COUNT, [FLD_DEBRIS_AT])
    # subarrays are indexed by house(iteration)

    """
    FLD_MEAN = 0
    FLD_DIARRAY = 1
    FLD_FRAGILITIES = 2
    FLD_PRESSURIZED_COUNT = 3
    FLD_DEBRIS_AT = 4
    FLD_WI_AT = 5
    FLD_DEBRIS_NV_AT = 6
    FLD_DEBRIS_NUM_AT = 7
    result_buckets = {}

    # record: fragility level name, level, plot color, CurvePlot
    fragility_thresholds = [['slight', 0.0, 'b', None],
                            ['medium', 0.0, 'g', None],
                            ['severe', 0.0, 'y', None],
                            ['complete', 0.0, 'r', None]]

    def __init__(self, options, diCallback=None, mplDict=None):
        self.s = None
        self.debrisManager = None
        self.A_final = None
        self.diCallback = diCallback
        self.wind_orientation = 0
        self.mzcat = 0
        self.di = 0
        self.mplDict = mplDict
        self.prevCurvePlot = None
        self.options = options

        # Undefined and later added
        self.result_wall_collapse = None
        self.cols = None
        self.rows = None
        self.house_number = None
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
        self.file_cpis = None
        self.file_debris = None
        self.file_damage = None
        self.file_dmg = None
        self.file_frag = None
        self.file_water = None
        self.speeds = None
        self.dmg_map = None
        self.frag_levels = None
        self.wind_speeds = None
        self.di_means = None
        self.ss = None

        global output
        if mplDict is not None:
            output = __import__('gui.output').output
            output.hookupWidget(mplDict)
            self.mplDict = mplDict
        else:
            output = __import__('core.output').output

        terrain.populate_wind_profile_by_terrain()
        self.clear_loop_results()

    @staticmethod
    def set_fragility_thresholds(thresholds):
        # thresholds is a dict in form {'slight': 0.0}
        for thresh_key in thresholds:
            for tup in WindDamageSimulator.fragility_thresholds:
                if tup[0] == thresh_key:
                    tup[1] = thresholds[thresh_key]
                    break

    def set_scenario(self, s):
        """
        simulate an intact house
        Args:
            s: an instance of Scenario class

        Returns:

        """
        self.s = s
        self.cols = [chr(x) for x in range(ord('A'), ord('A') +
                                           self.s.house.roof_columns)]
        self.rows = range(1, self.s.house.roof_rows + 1)
        self.set_fragility_thresholds(s.fragility_thresholds)
        self.s.house.clear_sim_results()

    def clear_loop_results(self):
        self.qz = 0.0
        self.Ms = 1.0
        self.di = 0.0
        self.fragilities = []

    def set_wind_direction(self):
        self.wind_orientation = self.s.getWindDirIndex()
        if self.debrisManager:
            self.debrisManager.set_wind_direction_index(self.wind_orientation)

    def set_wind_profile(self):
        self.profile = np.random.random_integers(1, 10)
        self.mzcat = terrain.calculateMZCAT(self.s.terrain_category,
                                            self.profile,
                                            self.s.getHouseHeight())

    def calculate_qz(self, wind_speed):
        if self.s.regional_shielding_factor <= 0.85:
            thresholds = np.array([63, 63+15])
            ms_dic = {0: 1.0, 1: 0.85, 2: 0.95}
            idx = sum(thresholds <= np.random.random_integers(0, 100))
            self.Ms = ms_dic[idx]
            Vmod = (wind_speed * self.Ms) / self.s.regional_shielding_factor
            self.qz = 0.6 * 1.0e-3 * (Vmod * self.mzcat)**2
        else:
            self.qz = 0.6 * 1.0e-3 * (wind_speed * self.mzcat)**2

    def check_pressurized_failure(self, v):
        if self.s.getOpt_Debris():
            self.debrisManager.run(v)
            if self.cpi == 0 and self.debrisManager.get_breached():
                self.cpi = 0.7
                self.cpiAt = v
                self.file_cpis.write('%d,%.3f\n' % (self.house_number + 1, v))

    def sample_house_and_wind_params(self):
        self.cpi = 0
        self.cpiAt = 0
        self.internally_pressurized = False
        self.set_wind_profile()
        self.s.house.reset_results()
        self.prev_di = 0

        self.construction_level = 'na'
        mean_factor = 1.0
        cov_factor = 1.0
        if self.s.getOpt_ConstructionLevels():
            self.construction_level, mean_factor, cov_factor = \
                self.s.sampleConstructionLevel()

        connection.assign_connection_strengths(self.s.house.connections,
                                               mean_factor,
                                               cov_factor)

        connection.assign_connection_deadloads(self.s.house.connections)

        zone.sample_zone_pressures(self.s.house.zones,
                                   self.wind_orientation,
                                   self.s.house.cpe_V,
                                   self.s.house.cpe_k,
                                   self.s.house.cpe_struct_V)
        # we don't want to collapse multiple times (no need)
        self.result_wall_collapse = False

    def check_house_collapse(self, wind_speed):
        if not self.result_wall_collapse:
            for ctg in self.s.house.conn_type_groups:
                if ctg.trigger_collapse_at > 0:
                    perc_damaged = 0
                    for ct in ctg.conn_types:
                        perc_damaged += ct.perc_damaged()
                    if perc_damaged >= ctg.trigger_collapse_at:
                        for c in self.s.house.connections:
                            c.damage(wind_speed, 99.9, inflZonesByConn[c])
                        for z in self.s.house.zones:
                            z.result_effective_area = 0
                        self.result_wall_collapse = True

    def calculate_connection_group_areas(self):
        for ctg in self.s.house.conn_type_groups:
            ctg.result_area = 0.0
        for c in self.s.house.connections:
            c.ctype.group.result_area += c.ctype.costing_area

    def calculate_damage_ratio(self, wind_speed):

        # calculate damage percentages        
        for ctg in self.s.house.conn_type_groups:
            ctg.result_percent_damaged = 0.0
            if ctg.group_name == 'debris':
                if not self.debrisManager:
                    ctg.result_percent_damaged = 0
                else:
                    ctg.result_percent_damaged = \
                        self.debrisManager.result_dmgperc
            else:
                for ct in ctg.conn_types:
                    for c in ct.connections_of_type:
                        if c.result_damaged:
                            ctg.result_percent_damaged += \
                                c.ctype.costing_area / float(ctg.result_area)

        # calculate repair cost
        repair_cost = 0
        for ctg in self.s.house.conn_type_groups:
            ctg_perc = ctg.result_percent_damaged
            if ctg_perc > 0:
                fact_arr = [0]
                for factor in self.s.house.factorings:
                    if factor.parent_id == ctg.id:
                        factor_perc = factor.factor.result_percent_damaged
                        if factor_perc:
                            fact_arr.append(factor_perc)
                max_factor_perc = max(fact_arr)
                if ctg_perc > max_factor_perc:
                    ctg_perc = ctg_perc - max_factor_perc
                    repair_cost += ctg.costing.calculate_damage(ctg_perc)

        # calculate initial envelope repair cost before water ingress is added
        self.di = repair_cost / self.s.house.replace_cost
        if self.di > 1.0:
            self.di = 1.0
        else:
            self.water_ingress_cost = 0
            if self.s.getOpt_WaterIngress():
                self.water_ingress_cost = \
                    wateringress.get_costing_for_envelope_damage_at_v(
                        self.di,
                        wind_speed,
                        self.s.house.water_groups,
                        self.file_water)
                repair_cost += self.water_ingress_cost

        # combined internal + envelope damage costing can now be calculated
        self.di = repair_cost / self.s.house.replace_cost
        if self.di > 1.0:
            self.di = 1.0
        self.prev_di = self.di

    def redistribute_damage(self, ctg):
        # setup for distribution
        if ctg.distribution_order <= 0:
            return
        distByCol = ctg.distribution_direction == 'col'
        primaryDir = self.cols
        secondaryDir = self.rows
        if not distByCol:
            primaryDir = self.rows
            secondaryDir = self.cols

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
                                    if not self.redistribute_to_nearest_zone(z, range(gridRow+1, self.s.house.roof_rows), k, ctg, gridCol, gridRow, distByCol):
                                        k = 1.0
                                    if not self.redistribute_to_nearest_zone(z, reversed(range(0, gridRow)), k, ctg, gridCol, gridRow, distByCol):
                                        self.redistribute_to_nearest_zone(z, range(gridRow+1, self.s.house.roof_rows), k, ctg, gridCol, gridRow, distByCol)
                                elif c.edge == 2:
                                    k = 1.0
                                    self.redistribute_to_nearest_zone(z, range(gridRow+1, self.s.house.roof_rows), k, ctg, gridCol, gridRow, distByCol)
                                elif c.edge == 1:
                                    k = 1.0
                                    self.redistribute_to_nearest_zone(z, reversed(range(0, gridRow)), k, ctg, gridCol, gridRow, distByCol)
                            else:
                                if c.edge == 0:
                                    k = 0.5
                                    if not self.redistribute_to_nearest_zone(z, range(gridCol+1, self.s.house.roof_columns), k, ctg, gridCol, gridRow, distByCol):
                                        k = 1.0
                                    if not self.redistribute_to_nearest_zone(z, reversed(range(0, gridCol)), k, ctg, gridCol, gridRow, distByCol):
                                        self.redistribute_to_nearest_zone(z, range(gridCol+1, self.s.house.roof_columns), k, ctg, gridCol, gridRow, distByCol)
                                elif c.edge == 2:
                                    k = 1.0
                                    self.redistribute_to_nearest_zone(z, range(gridCol+1, self.s.house.roof_columns), k, ctg, gridCol, gridRow, distByCol)
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

    def run_simulation(self, wind_speed):
        self.check_pressurized_failure(wind_speed)

        self.calculate_qz(wind_speed)
        zone.calc_zone_pressures(self.s.house.zones,
                                 self.wind_orientation,
                                 self.cpi,
                                 self.qz,
                                 self.Ms,
                                 self.s.building_spacing,
                                 self.s.getOpt_DiffShielding())

        self.file_damage.write('%d,%.3f,%s' % (self.house_number+1,
                                               wind_speed,
                                               scenario.Scenario.dirs[self.wind_orientation]))
        for ctg in self.s.house.conn_type_groups:
            connection.calc_connection_loads(wind_speed,
                                             ctg,
                                             self.s.house,
                                             self.file_damage,
                                             self.dmg_map,
                                             inflZonesByConn,
                                             connByTypeMap)
        if self.s.getOpt_DmgDistribute():
            for ctg in self.s.house.conn_type_groups:
                self.redistribute_damage(ctg)
        self.file_damage.write('\n')

        self.check_house_collapse(wind_speed)
        self.calculate_damage_ratio(wind_speed)

    def simulator_mainloop(self, verbose=False):
        date_run = datetime.datetime.now()

        # setup file based reporting (files must exist and be runnable)
        if not os.path.exists(self.options.output_folder):
            os.makedirs(self.options.output_folder)

        self.file_cpis = open(os.path.join(self.options.output_folder,
                                           'house_cpi.csv'), 'w')
        self.file_cpis.write('Simulated House #, Cpi Changed At\n')
        self.file_debris = open(os.path.join(self.options.output_folder,
                                             'wind_debris.csv'), 'w')
        header_ = ('Wind Speed(m/s),% Houses Internally Pressurized,'
                   '% Debris Damage Mean\n')
        self.file_debris.write(header_)
        self.file_damage = open(os.path.join(self.options.output_folder,
                                             'house_damage.csv'), 'w')
        self.file_dmg_idx = open(os.path.join(self.options.output_folder,
                                             'house_dmg_idx.csv'), 'w')
        self.file_dmg = open(os.path.join(self.options.output_folder,
                                          'houses_damaged_at_v.csv'), 'w')
        self.file_frag = open(os.path.join(self.options.output_folder,
                                           'fragilities.csv'), 'w')
        header_ = ('Slight Median,Slight Beta,Medium Median,Median Beta,'
                   'Severe Median,Severe Beta,Complete Median,Complete Beta\n')
        self.file_frag.write(header_)
        self.file_water = open(os.path.join(self.options.output_folder,
                                            'wateringress.csv'), 'w')
        header_ = ('V,Envelope DI,Water Damage,Damage Scenario,'
                   'Water Damage Cost,WaterCosting\n')
        self.file_water.write(header_)

        header = 'Simulated House #,Wind Speed(m/s),Wind Direction,'

        list_ = [ct.connection_type for ctg in self.s.house.conn_type_groups
                 if ctg.enabled for ct in ctg.conn_types]
        header += ','.join(list_)
        header += '\n'
        self.file_damage.write(header)

        # optionally seed random numbers
        if self.s.getOpt_SampleSeed():
            np.random.seed(42)
            zone.seed_scipy(42)
            engine.seed(42)

        # setup speeds and buckets
        self.speeds = np.linspace(self.s.wind_speed_min,
                                  self.s.wind_speed_max,
                                  self.s.wind_speed_num_steps)

        for wind_speed in self.speeds:
            type(self).result_buckets[wind_speed] = \
                [0., [], [], 0, [], [], [], []]

        # setup connections and groups
        self.s.house.clear_sim_results()
        self.calculate_connection_group_areas()

        # optionally create the debris manager and
        # make sure a wind orientation is set
        bDebris = self.s.getOpt_Debris()
        if bDebris:
            self.debrisManager = debris.DebrisManager(
                self.s.house,
                self.s.region,
                self.s.wind_speed_min,
                self.s.wind_speed_max,
                self.s.wind_speed_num_steps,
                self.s.getOpt_DebrisStaggeredSources(),
                self.s.debris_radius,
                self.s.debris_angle,
                self.s.debris_extension,
                self.s.building_spacing,
                self.s.source_items,
                self.s.flighttime_mean,
                self.s.flighttime_stddev)
        self.set_wind_direction()

        # gui bookkeeping
        if self.diCallback:
            totalLoops = self.s.num_iters * len(self.speeds)
            currentLoop = 1

        # SIMULATE HOUSES
        house_results = []
        keep_looping = True

        # iteration over samples
        for house_number in range(self.s.num_iters):
            self.house_number = house_number
            if not keep_looping:
                break

            # sample new house and wind direction (if random)
            if self.s.wind_dir_index == 8:
                self.set_wind_direction()

            self.sample_house_and_wind_params()

            # prime damage map where we track min() V that damage occurs
            # across types for this house (reporting)
            self.dmg_map = {}
            for conn_type in self.s.house.conn_types:
                self.dmg_map[conn_type.connection_type] = 99999

            # iteration over wind speed list
            for wind_speed in self.speeds:

                # simulate sampled house
                self.clear_loop_results()
                self.run_simulation(wind_speed)

                # collect results
                type(self).result_buckets[wind_speed][type(self).FLD_WI_AT].append(self.water_ingress_cost)
                type(self).result_buckets[wind_speed][type(self).FLD_DIARRAY].append(self.di)
                if bDebris:
                    type(self).result_buckets[wind_speed][type(self).FLD_DEBRIS_AT].append(self.debrisManager.result_dmgperc)
                    type(self).result_buckets[wind_speed][type(self).FLD_DEBRIS_NV_AT].append(self.debrisManager.result_nv)
                    type(self).result_buckets[wind_speed][type(self).FLD_DEBRIS_NUM_AT].append(self.debrisManager.result_num_items)
                else:
                    type(self).result_buckets[wind_speed][type(self).FLD_DEBRIS_AT].append(0.0)
                    type(self).result_buckets[wind_speed][type(self).FLD_DEBRIS_NV_AT].append(0)
                    type(self).result_buckets[wind_speed][type(self).FLD_DEBRIS_NUM_AT].append(0)

                # for all houses, count the number that were pressurized at this wind_speed   
                if self.cpi != 0:
                    type(self).result_buckets[wind_speed][type(self).FLD_PRESSURIZED_COUNT] += 1

                # interact with GUI listener
                if self.diCallback:
                    currentLoop += 1
                    percLoops = (float(currentLoop) / float(totalLoops)) * 100.0
                    keep_looping = self.diCallback(wind_speed, self.di, percLoops)
                    if not keep_looping:
                        break

            # collect results to be used by the GUI client
            zone_results = {}
            for z in self.s.house.zones:
                zone_results[z.zone_name] = [z.zone_name,
                                             z.sampled_cpe,
                                             z.sampled_cpe_struct,
                                             z.sampled_cpe_eaves]

            conn_results = []
            for c in self.s.house.connections:
                conn_results.append([c.ctype.connection_type,
                                     c.location_zone.zone_name,
                                     c.result_failure_v_raw,
                                     c.result_strength,
                                     c.result_deadload,
                                     c.result_damaged_report,
                                     c.ctype.group.group_name,
                                     c.id])

            house_results.append([zone_results,
                                  self.dmg_map,
                                  self.wind_orientation,
                                  self.cpiAt,
                                  conn_results,
                                  self.construction_level])

        if keep_looping:
            # post processing of results (aggregations)
            for wind_speed in self.speeds:

                # write debris output file
                debrisarray = np.array(type(self).result_buckets[wind_speed][type(self).FLD_DEBRIS_AT])
                perc = type(self).result_buckets[wind_speed][type(self).FLD_PRESSURIZED_COUNT]/float(self.s.num_iters) * 100.0
                self.file_debris.write('%.3f,%.3f,%.3f\n' % (wind_speed,
                                                             perc,
                                                             np.mean(debrisarray) * 100.0))

                # calculate and store DI mean
                diarray = np.array(type(self).result_buckets[wind_speed][type(self).FLD_DIARRAY])
                type(self).result_buckets[wind_speed][type(self).FLD_MEAN] = np.mean(diarray)

                # calculate fragilities
                for thr in type(self).fragility_thresholds:
                    filter_ = diarray > thr[1]
                    p = (float(diarray[filter_].size)) / (float(diarray.size))
                    type(self).result_buckets[wind_speed][type(self).FLD_FRAGILITIES].append(p)

        # produce damage map report
        self.file_dmg.write('Number of Damaged Houses\n')
        self.file_dmg.write('Num Houses,%d\n' % self.s.num_iters)
        self.file_dmg.write('Wind Direction,%s\n' % scenario.Scenario.dirs[self.wind_orientation])
        self.file_dmg.write('Wind Speed(m/s)')

        # setup headers and counts
        str_ = [conn_type.connection_type for conn_type in
                self.s.house.conn_types]
        self.file_dmg.write(','.join(str_))
        self.file_dmg.write('\n')

        # we need to count houses damaged by type for each v
        counts = {}
        for wind_speed in self.speeds:
            self.file_dmg.write(str(wind_speed))

            # initialise damage counts for each conn_type to zero
            for conn_type in self.s.house.conn_types:
                counts[conn_type.connection_type] = 0

            # for all houses, increment type counts
            # if wind_speed exceeds minimum observed damages.
            for hr in house_results:
                dmg_map = hr[1]
                for conn_type in self.s.house.conn_types:
                    dmg_min = dmg_map[conn_type.connection_type]
                    if wind_speed >= dmg_min:
                        counts[conn_type.connection_type] += 1

            # write accumulated counts for this wind speed
            str_ = [str(counts[conn_type.connection_type]) for conn_type
                    in self.s.house.conn_types]
            self.file_dmg.write(','.join(str_))
            self.file_dmg.write('\n')

        # cleanup: close output files
        self.file_cpis.close()
        self.file_debris.close()
        self.file_damage.close()
        self.file_dmg.close()
        self.file_water.close()
        self.debrisManager = None

        # temporary
        di_summary = []
        for speed, value in type(self).result_buckets.iteritems():
            tmp = np.array(value[1])
            tmp_added = np.append(tmp, [speed, tmp.mean()])
            di_summary.append(tmp_added)

        columns_str = [str(x) for x in range(self.s.num_iters)]
        columns_str.append('speed')
        columns_str.append('mean')
        df_di_summary = pd.DataFrame(di_summary, columns=columns_str)
        df_di_summary.to_csv(self.file_dmg_idx, index=False)
        self.file_dmg_idx.close()

        if keep_looping:
            self.fit_fragility_curves()
            self.file_frag.close()
            runTime = (datetime.datetime.now() - date_run)
            return runTime, house_results
        else:
            self.file_frag.close()
            return None, None

    def get_windresults_perc_houses_breached(self):
        breaches = []
        for wind_speed in self.speeds:
            perc = type(self).result_buckets[wind_speed][type(self).FLD_PRESSURIZED_COUNT]/float(self.s.num_iters) * 100.0
            breaches.append(perc)
        return self.speeds, breaches

    def get_windresults_samples_perc_debris_damage(self):
        samples = {}
        for wind_speed in self.speeds:
            samples[wind_speed] = type(self).result_buckets[wind_speed][type(self).FLD_DEBRIS_AT]
        return self.speeds, samples

    def get_windresults_samples_nv(self):
        samples = {}
        for wind_speed in self.speeds:
            samples[wind_speed] = type(self).result_buckets[wind_speed][type(self).FLD_DEBRIS_NV_AT]
        return self.speeds, samples

    def get_windresults_samples_num_items(self):
        samples = {}
        for wind_speed in self.speeds:
            samples[wind_speed] = type(self).result_buckets[wind_speed][type(self).FLD_DEBRIS_NUM_AT]
        return self.speeds, samples

    def get_windresults_samples_perc_water_ingress(self):
        samples = {}
        for wind_speed in self.speeds:
            samples[wind_speed] = type(self).result_buckets[wind_speed][type(self).FLD_WI_AT]
        return self.speeds, samples

    def plot_connection_damage(self, vRed, vBlue):
        for ctg_name in ['sheeting', 'batten', 'rafter', 'piersgroup',
                         'wallracking', 'Truss']:
            if ctgMap.get(ctg_name, None) is None:
                continue
            vgrid = np.ones((self.s.house.roof_rows, self.s.house.roof_columns),
                            dtype=np.float32) * vBlue + 10.0
            for conn in connByTypeGroupMap[ctg_name]:
                gridCol, gridRow = \
                    zone.getGridFromZoneLoc(conn.location_zone.zone_name)
                if conn.result_failure_v > 0:
                    vgrid[gridRow][gridCol] = conn.result_failure_v
            output.plot_damage_show(ctg_name, vgrid, self.s.house.roof_columns,
                                    self.s.house.roof_rows, vRed, vBlue)

        if 'plot_wall_damage_show' in output.__dict__:
            wall_major_rows = 2
            wall_major_cols = self.s.house.roof_columns
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
        # unpack the fragility means into seperate arrays for fit/plot.
        self.frag_levels = []
        for thr in type(self).fragility_thresholds:
            self.frag_levels.append(np.zeros(len(self.speeds)))
        for i, wind_speed in enumerate(self.speeds):
            for frag_ind in range(len(self.frag_levels)):
                self.frag_levels[frag_ind][i] = \
                    type(self).result_buckets[wind_speed][type(self).FLD_FRAGILITIES][frag_ind]

        # fit curves and store results
        coeff_arr = []
        ss = 0
        for frag_ind in range(len(self.frag_levels)):
            try:
                coeff_arr, ss = curve_log.fit_curve(self.speeds,
                                                    self.frag_levels[frag_ind],
                                                    False)
                if frag_ind > 0:
                    self.file_frag.write(',')

                label = '%s(%.2f)' % (type(self).fragility_thresholds[frag_ind][0],
                                    type(self).fragility_thresholds[frag_ind][1])
                self.file_frag.write('%f,%f' % (coeff_arr[0], coeff_arr[1]))
                type(self).fragility_thresholds[frag_ind][3] = \
                    CurvePlot(coeff_arr,
                              'lognormal',
                              self.s.wind_speed_min,
                              self.s.wind_speed_max,
                              label,
                              type(self).fragility_thresholds[frag_ind][2])
            except Exception, e:
                print 'fit_fragility_curves failed to fit: coeff_arr: %s' % coeff_arr
                print e

        self.file_frag.write('\n')

    def plot_fragility(self, output_folder):
        for frag_ind in range(len(self.frag_levels)):
            output.plot_fragility_curve(self.speeds,
                                        self.frag_levels[frag_ind],
                                        '_nolegend_',
                                        0.3,
                                        type(self).fragility_thresholds[frag_ind][2])
            plot_obj = type(self).fragility_thresholds[frag_ind][3]
            if plot_obj:
                plot_obj.plot_frag()

        output.plot_fragility_show(self.s.num_iters,
                                   self.s.wind_speed_min,
                                   self.s.wind_speed_max, output_folder)

    def fit_vuln_curve(self):
        self.wind_speeds = np.zeros(len(self.speeds))
        self.di_means = np.zeros(len(self.speeds))

        ss = 0
        for i, wind_speed in enumerate(self.speeds):
            self.wind_speeds[i] = wind_speed
            self.di_means[i] = type(self).result_buckets[wind_speed][type(self).FLD_MEAN]

        if self.s.getOpt_VulnFitLog():
            self.A_final, self.ss = curve_log.fit_curve(self.wind_speeds,
                                                        self.di_means)
        else:
            self.A_final, self.ss = curve.fit_curve(self.wind_speeds,
                                                    self.di_means)

    def plot_vulnerability(self, output_folder, label="Fitted Curve"):
        # fit current observations
        self.fit_vuln_curve()

        # plot means
        if self.s.num_iters <= 100:
            for wind_speed in self.speeds:
                damage_indexes = type(self).result_buckets[wind_speed][type(self).FLD_DIARRAY]
                output.plot_wind_event_damage([wind_speed]*len(damage_indexes),
                                              damage_indexes)
        output.plot_wind_event_mean(self.wind_speeds, self.di_means)

        # plot fitted curve (with previous dimmed red)
        if self.s.getOpt_VulnFitLog():
            fn_form = 'lognormal'
        else:
            fn_form = 'original'

        cp = CurvePlot(self.A_final,
                       fn_form,
                       self.s.wind_speed_min,
                       self.s.wind_speed_max,
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
        if self.s.getOpt_DmgPlotFragility():
            self.plot_fragility(output_folder)
        if self.s.getOpt_DmgPlotVuln():
            self.plot_vulnerability(output_folder)
            output.plot_wind_event_show(self.s.num_iters,
                                        self.s.wind_speed_min,
                                        self.s.wind_speed_max,
                                        output_folder)
        self.plot_connection_damage(vRed, vBlue)

    def clear_connection_damage(self):
        v = np.ones((self.s.house.roof_rows, self.s.house.roof_columns),
                    dtype=np.float32) * self.s.wind_speed_max
        for ctname in ['sheeting', 'batten', 'rafter', 'piersgroup',
                       'wallracking']:
            output.plot_damage_show(ctname, v, self.s.house.roof_columns,
                                    self.s.house.roof_rows,
                                    self.s.wind_speed_min,
                                    self.s.wind_speed_max)
        for ctname in ['wallcladding', 'wallcollapse']:
            wall_major_rows = 2
            wall_major_cols = self.s.house.roof_columns
            wall_minor_rows = 2
            wall_minor_cols = 8
            v_major_grid = np.ones((wall_major_rows, wall_major_cols),
                                   dtype=np.float32) * self.s.wind_speed_max
            v_minor_grid = np.ones((wall_minor_rows, wall_minor_cols),
                                   dtype=np.float32) * self.s.wind_speed_max
            output.plot_wall_damage_show(
                ctname,
                v_major_grid, v_major_grid, v_minor_grid, v_minor_grid,
                wall_major_cols, wall_major_rows,
                wall_minor_cols, wall_minor_rows,
                self.s.wind_speed_min, self.s.wind_speed_max)


lastiPerc = -1


def simProgressCallback(V, di, percLoops):
    global lastiPerc
    iPerc = int(percLoops)
    if iPerc != lastiPerc:
        lastiPerc = iPerc
        sys.stdout.write('.')
    return True


# @profile
def simulate(s, options):
    if options.verbose:
        arg = simProgressCallback
    else:
        arg = None
    mySim = WindDamageSimulator(options, arg, None)
    mySim.set_scenario(s)
    runTime, hr = mySim.simulator_mainloop(options.verbose)
    if runTime:
        if options.plot_frag:
            mySim.plot_fragility(options.output_folder)
        if options.plot_vuln:
            mySim.plot_vulnerability(options.output_folder, None)
            output.plot_wind_event_show(mySim.s.num_iters,
                                        mySim.s.wind_speed_min,
                                        mySim.s.wind_speed_max,
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
        s = scenario.loadFromCSV(options.scenario_filename)
        simulate(s, options)
        database.db.close()
    else:
        print '\nERROR: Must provide as scenario file to run simulator...\n'
        parser.print_help()


if __name__ == '__main__':
    main()
