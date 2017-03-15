import sys
import os
import time
import logging

# from pathos.multiprocessing import ProcessingPool
import pandas as pd
import numpy as np
from optparse import OptionParser

from house_damage import HouseDamage
from scenario import Scenario
from version import VERSION_DESC


def simulate_wind_damage_to_houses(cfg, call_back):
    """

    Args:
        cfg:
        call_back:

    Returns:

    """

    # simulator main_loop
    tic = time.time()

    mean_damage = dict()
    damage_incr = 0.0
    list_results = list()
    calc_count = 0

    if cfg.parallel:
        cfg.parallel = False
        logging.info('parallel running is not implemented yet')

        # very slow
        # iteration over wind speed list
        # for wind_speed in cfg.speeds:
        #
        #     list_ = ProcessingPool().map(HouseDamage.run_simulation,
        #         list_house_damage, [wind_speed]*cfg.no_sims)

    logging.info('Starting simulation in serial')

    # generate instances of house_damage
    list_house_damage = [create_house_damage(i, cfg) for i in
                         range(cfg.no_sims)]

    for ispeed, wind_speed in enumerate(cfg.speeds):

        list_results_by_speed = list()

        for house_damage in list_house_damage:

            if cfg.flags['debris']:
                house_damage.house.debris.no_items_mean = damage_incr

            if call_back:
                percent_done = (calc_count + ispeed * len(cfg.speeds)) / (
                    len(cfg.speeds) * cfg.no_sims)

                if not call_back(int(percent_done * 100)):
                    return

            list_ = house_damage.run_simulation(wind_speed)
            list_results_by_speed.append(list_)

            calc_count += 1

        damage_incr, mean_damage = cal_damage_increment(
            list_results_by_speed, mean_damage, ispeed)

        list_results.append(list_results_by_speed)

    save_results_to_files(cfg, list_results)

    logging.info('Time taken for simulation {}'.format(time.time()-tic))

    return time.time()-tic, list_results


def cal_damage_increment(list_, dic_, index_):
    """

    Args:
        list_: list of results
        dic_: dic of damge index by wind step
        index_: index of wind step

    Returns:

    """
    dic_[index_] = np.array([x['house']['di'] for x in list_]).mean()
    damage_incr = 0.0  # default value

    # only index >= 1
    if index_:

        damage_incr = dic_[index_] - dic_[index_ - 1]

        if damage_incr < 0:
            logging.warn('damage increment is less than zero')
            damage_incr = 0.0

    return damage_incr, dic_


def save_results_to_files(cfg, list_results):
    """

    Args:
        cfg:
        list_results: [wind_speed_steps][no_sims]{'house': df, 'group': df,
        'type': df, 'conn': df, 'zone': df }

    Returns:

    """

    # file_house (attribute: DataFrame(wind_speed_steps, no_sims)
    hdf = pd.HDFStore(cfg.file_house, mode='w')
    for item in cfg.list_house_bucket + cfg.list_house_damage_bucket + \
            cfg.list_debris_bucket:
        df_ = pd.DataFrame(dtype=float, columns=range(cfg.no_sims),
                           index=range(cfg.wind_speed_steps))
        for ispeed in range(cfg.wind_speed_steps):
            df_.loc[ispeed] = [x['house'][item] for x in list_results[ispeed]]
        hdf.append(item, df_, format='t')
    hdf.close()

    # file_group (group: Panel(attribute, wind_speed_steps, no_sims)
    hdf = pd.HDFStore(cfg.file_group, mode='w')
    for group_name in list_results[0][0]['group'].index.tolist():
        _panel = pd.Panel(dtype=float, items=cfg.list_group_bucket,
                          major_axis=range(cfg.wind_speed_steps),
                          minor_axis=range(cfg.no_sims))
        for item in cfg.list_group_bucket:
            for ispeed in range(cfg.wind_speed_steps):
                _panel[item].loc[ispeed] = [x['group'][item].values[0] for x
                                            in list_results[ispeed]]
        hdf[group_name] = _panel
    hdf.close()

    # results by type for each model
    hdf = pd.HDFStore(cfg.file_type, mode='w')
    for type_name in list_results[0][0]['type'].index.tolist():
        _panel = pd.Panel(dtype=float, items=cfg.list_type_bucket,
                          major_axis=range(cfg.wind_speed_steps),
                          minor_axis=range(cfg.no_sims))
        for item in cfg.list_type_bucket:
            for ispeed in range(cfg.wind_speed_steps):
                _panel[item].loc[ispeed] = [x['type'][item].values[0] for x
                                            in list_results[ispeed]]
        hdf[type_name] = _panel
    hdf.close()

    # results by connection for each model
    hdf = pd.HDFStore(cfg.file_conn, mode='w')
    for conn_name in list_results[0][0]['conn'].index.tolist():
        _panel = pd.Panel(dtype=float, items=cfg.list_conn_bucket,
                          major_axis=range(cfg.wind_speed_steps),
                          minor_axis=range(cfg.no_sims))
        for item in cfg.list_conn_bucket:
            for ispeed in range(cfg.wind_speed_steps):
                _panel[item].loc[ispeed] = [x['conn'][item].values[0] for x
                                            in list_results[ispeed]]
        hdf['c' + str(conn_name)] = _panel
    hdf.close()

    # results by zone for each model
    hdf = pd.HDFStore(cfg.file_zone, mode='w')
    for zone_name in list_results[0][0]['zone'].index.tolist():
        _panel = pd.Panel(dtype=float, items=cfg.list_zone_bucket,
                          major_axis=range(cfg.wind_speed_steps),
                          minor_axis=range(cfg.no_sims))
        for item in cfg.list_zone_bucket:
            for ispeed in range(cfg.wind_speed_steps):
                _panel[item].loc[ispeed] = [x['zone'][item].values[0] for x
                                            in list_results[ispeed]]
        hdf[zone_name] = _panel
    hdf.close()


def create_house_damage(id_sim, cfg):

    seed = cfg.flags['random_seed'] + id_sim
    house_damage = HouseDamage(cfg, seed)

    return house_damage


def process_commandline():
    usage = '%prog -s <scenario_file> [-o <output_folder>] [-v]'
    parser = OptionParser(usage=usage, version=VERSION_DESC)
    parser.add_option("-s", "--scenario",
                      dest="scenario_filename",
                      help="read scenario description from FILE",
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
    return parser


def main():

    parser = process_commandline()

    (options, args) = parser.parse_args()

    path_, _ = os.path.split(sys.argv[0])

    if options.output_folder is None:
        options.output_folder = os.path.abspath(
            os.path.join(path_, '../../outputs/output'))
    else:
        options.output_folder = os.path.abspath(
            os.path.join(os.getcwd(), options.output_folder))

    if options.verbose:
        file_logger = os.path.join(options.output_folder, 'log.txt')
        logging.basicConfig(filename=file_logger,
                            filemode='w',
                            level=logging.DEBUG,
                            format='%(levelname)s %(message)s')

    if options.scenario_filename:
        conf = Scenario(cfg_file=options.scenario_filename,
                        output_path=options.output_folder)
        _ = simulate_wind_damage_to_houses(conf, call_back=None)
    else:
        print '\nERROR: Must provide a scenario file to run simulator...\n'
        parser.print_help()

    logging.info('Program finished')

if __name__ == '__main__':
    main()
