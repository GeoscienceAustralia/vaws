import sys
import os
import time
import logging
import warnings

# from pathos.multiprocessing import ProcessingPool
import pandas as pd
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

    damage_incr = 0.0
    dic_panels = init_panels(cfg)

    if cfg.parallel:
        cfg.parallel = False
        print('parallel running is not implemented yet, '
              'switched to serial mode')

    logging.info('Starting simulation in serial')

    # generate instances of house_damage
    list_house_damage = [HouseDamage(cfg, i) for i in range(cfg.no_sims)]

    for ispeed, wind_speed in enumerate(cfg.speeds):

        list_results_by_speed = list()

        for house_damage in list_house_damage:

            if cfg.flags['debris']:
                house_damage.house.debris.no_items_mean = damage_incr

            list_ = house_damage.run_simulation(wind_speed)

            list_results_by_speed.append(list_)

        damage_incr = update_panels(dic_panels, list_results_by_speed, ispeed)

        if not call_back:
            sys.stdout.write('{} out of {} completed \n'.format(
                ispeed + 1, len(cfg.speeds)))
            sys.stdout.flush()
        else:
            percent_done = 100.0 * (ispeed + 1) / len(cfg.speeds)
            if not call_back(int(percent_done)):
                return

    save_results_to_files(cfg, dic_panels)

    logging.info('Time taken for simulation {}'.format(time.time()-tic))

    return time.time()-tic, dic_panels


def init_panels(cfg):

    dic_panels = dict()

    # house
    list_atts = cfg.list_house_bucket + cfg.list_house_damage_bucket + \
                cfg.list_debris_bucket
    dic_panels['house'] = pd.Panel(dtype=float,
                                   items=list_atts,
                                   major_axis=range(cfg.wind_speed_steps),
                                   minor_axis=range(cfg.no_sims))
    # other components
    for item in ['group', 'type', 'conn', 'zone']:
        for att in getattr(cfg, 'list_{}s'.format(item)):
            dic_panels.setdefault(item, {})[att] = \
                pd.Panel(dtype=float,
                         items=getattr(cfg, 'list_{}_bucket'.format(item)),
                         major_axis=range(cfg.wind_speed_steps),
                         minor_axis=range(cfg.no_sims))

    return dic_panels


def update_panels(dic_, list_results_by_speed, ispeed):

    # house
    for att in dic_['house'].items.tolist():
        dic_['house'][att].loc[ispeed] = [x['house'][att] for x in
                                          list_results_by_speed]

    # other components
    for item in ['group', 'type', 'conn', 'zone']:
        for key, value in dic_[item].iteritems():
            for att in value.items.tolist():
                value[att].loc[ispeed] = [x[item].loc[key, att] for x in
                                          list_results_by_speed]

    # compute damage index increment
    damage_incr = 0.0  # default value

    if ispeed:

        damage_incr = dic_['house']['di'].loc[ispeed].mean(axis=0) - \
                      dic_['house']['di'].loc[ispeed - 1].mean(axis=0)

        if damage_incr < 0:
            logging.warn('damage increment is less than zero')
            damage_incr = 0.0

    return damage_incr


def save_results_to_files(cfg, dic_panels):
    """

    Args:
        cfg:
        dic_panels:

    Returns:

    """

    # file_house (attribute: DataFrame(wind_speed_steps, no_sims)
    dic_panels['house'].to_hdf(cfg.file_house, key='house', mode='w')

    # results by group for each model
    # other components
    for item in ['group', 'type', 'conn', 'zone']:

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")
            #  warnings.filterwarnings('error')

            # file (key: Panel(attribute, wind_speed_steps, no_sims)
            hdf = pd.HDFStore(getattr(cfg, 'file_{}'.format(item)), mode='w')

            for key, value in dic_panels[item].iteritems():
                try:
                    hdf[str(key)] = value
                except Warning, msg:
                    logging.warning(msg)

            hdf.close()

    # plot fragility and vulnerability curves
    frag_counted = fit_fragility_curves(cfg, dic_panels['house']['di'])
    pd.DataFrame.from_dict(frag_counted).transpose().to_csv(cfg.file_curve)

    fitted_curve = fit_vulnerability_curve(cfg, dic_panels['house']['di'])

    with open(cfg.file_curve, 'a') as f:
        pd.DataFrame.from_dict(fitted_curve).transpose().to_csv(f, header=None)


def process_commandline():
    usage = '%prog -s <scenario_file> [-o <output_folder>] [-v <logging_level>]'
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
                      dest="verbose",
                      default=False,
                      metavar="logging_level",
                      help="set logging level")
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

        logger = logging.getLogger()
        file_logger = os.path.join(options.output_folder, 'log.txt')
        file_handler = logging.FileHandler(filename=file_logger, mode='w')
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        try:
            logger.setLevel(getattr(logging, options.verbose.upper()))
        except AttributeError:
            print('{} is not a logging level'.format(options.verbose))
            logger.setLevel(logging.DEBUG)

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
