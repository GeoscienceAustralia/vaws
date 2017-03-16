import sys
import os
import time
import logging
import warnings

# from pathos.multiprocessing import ProcessingPool
import pandas as pd
import numpy as np
from optparse import OptionParser

from house_damage import HouseDamage
from scenario import Scenario
from curve import fit_fragility_curves, fit_vulnerability_curve
from version import VERSION_DESC


def simulate_wind_damage_to_houses(cfg, call_back=None):
    """

    Args:
        cfg: instance of Scenario class
        call_back: used by gui

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
        print('parallel running is not implemented yet, '
              'switched to serial mode')

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

                print('{}'.format(percent_done))

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

    # results by group for each model
    save_panel_to_hdf(list_results,
                      file_=cfg.file_group,
                      key='group',
                      list_key=cfg.list_groups,
                      list_items=cfg.list_group_bucket,
                      list_major_axis=range(cfg.wind_speed_steps),
                      list_minor_axis=range(cfg.no_sims))

    # results by type for each model
    save_panel_to_hdf(list_results,
                      file_=cfg.file_type,
                      key='type',
                      list_key=cfg.list_types,
                      list_items=cfg.list_type_bucket,
                      list_major_axis=range(cfg.wind_speed_steps),
                      list_minor_axis=range(cfg.no_sims))

    # results by connection for each model
    save_panel_to_hdf(list_results,
                      file_=cfg.file_conn,
                      key='conn',
                      list_key=cfg.list_conns,
                      list_items=cfg.list_conn_bucket,
                      list_major_axis=range(cfg.wind_speed_steps),
                      list_minor_axis=range(cfg.no_sims))

    # results by zone for each model
    save_panel_to_hdf(list_results,
                      file_=cfg.file_zone,
                      key='zone',
                      list_key=cfg.list_zones,
                      list_items=cfg.list_zone_bucket,
                      list_major_axis=range(cfg.wind_speed_steps),
                      list_minor_axis=range(cfg.no_sims))

    # plot fragility and vulnerability curves
    df_damage_index = pd.read_hdf(cfg.file_house, 'di')

    frag_counted = fit_fragility_curves(cfg, df_damage_index)
    pd.DataFrame.from_dict(frag_counted).transpose().to_csv(cfg.file_curve)

    popt, perror = fit_vulnerability_curve(cfg, df_damage_index)
    print('{}:{}'.format(popt, perror))




def save_panel_to_hdf(list_results, file_, key, list_key, list_items,
                      list_major_axis, list_minor_axis):
    """

    Args:
        list_results:
        file_:
        key:
        list_key:
        list_items:
        list_major_axis:
        list_minor_axis:

    Returns:

    """

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        # file (key: Panel(attribute, wind_speed_steps, no_sims)
        hdf = pd.HDFStore(file_, mode='w')
        for _name in list_key:
            _panel = pd.Panel(dtype=float,
                              items=list_items,
                              major_axis=list_major_axis,
                              minor_axis=list_minor_axis)
            for item in list_items:
                for ispeed in list_major_axis:
                    _panel[item].loc[ispeed] = [x[key][item].values[0] for x
                                                in list_results[ispeed]]
            hdf[str(_name)] = _panel
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
