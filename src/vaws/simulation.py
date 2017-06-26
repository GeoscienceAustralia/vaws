import sys
import os
import time
import logging
import warnings

# from pathos.multiprocessing import ProcessingPool
import pandas as pd
import h5py
import numpy as np
from optparse import OptionParser

from vaws.house_damage import HouseDamage
from vaws.config import Config
from vaws.curve import fit_fragility_curves, fit_vulnerability_curve
from vaws.output import plot_heatmap
from vaws.version import VERSION_DESC


def simulate_wind_damage_to_houses(cfg, call_back=None):
    """

    Args:
        cfg: instance of Config class
        call_back: used by gui

    Returns:

    """

    # simulator main_loop
    tic = time.time()

    damage_incr = 0.0
    dic_panels = init_panels(cfg)

    # if cfg.parallel:
    #     cfg.parallel = False
    #     logging.info('parallel running is not implemented yet, '
    #                  'switched to serial mode')

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

        damage_incr = update_panels(cfg, dic_panels, list_results_by_speed,
                                    ispeed)

        if not call_back:
            sys.stdout.write('{} out of {} completed \n'.format(
                ispeed + 1, len(cfg.speeds)))
            sys.stdout.flush()
        else:
            percent_done = 100.0 * (ispeed + 1) / len(cfg.speeds)
            if not call_back(int(percent_done)):
                return

    save_results_to_files(cfg, dic_panels)

    elapsed = time.time()-tic
    logging.info('Time taken for simulation {}'.format(elapsed))

    return elapsed, dic_panels


def init_panels(cfg):

    dic_panels = dict()

    # house
    for item in (cfg.list_house_bucket + cfg.list_house_damage_bucket +
                 cfg.list_debris_bucket):
        dic_panels.setdefault('house', {})[item] = np.empty(
            shape=(cfg.wind_speed_steps, cfg.no_sims), dtype=float)

    # components
    # dic[comp][att][_conn]
    for item in cfg.list_components:
        dic_panels[item] = {}
        for att in getattr(cfg, 'list_{}_bucket'.format(item)):
            dic_panels[item][att] = {}
            for _conn in getattr(cfg, 'list_{}s'.format(item)):
                dic_panels[item][att][_conn] = np.empty(
                    shape=(cfg.wind_speed_steps, cfg.no_sims), dtype=float)

    return dic_panels


def update_panels(cfg, dic_, list_results_by_speed, ispeed):

    # house
    for att in (cfg.list_house_bucket + cfg.list_house_damage_bucket +
                cfg.list_debris_bucket):
        try:
            dic_['house'][att][ispeed] = [x['house'][att] for x in
                                          list_results_by_speed]
        except ValueError:
            print('{} can not be stored'.format(att))

    # components
    for item in cfg.list_components:
        for att, chunk in dic_[item].iteritems():
            for _conn, value in chunk.iteritems():
                value[ispeed] = [x[item][_conn][att] for x in
                                 list_results_by_speed]

    # compute damage index increment
    damage_incr = 0.0  # default value

    if ispeed:
        damage_incr = (dic_['house']['di'][ispeed].mean(axis=0) -
                       dic_['house']['di'][ispeed - 1].mean(axis=0))

        if damage_incr < 0:
            logging.warning('damage increment is less than zero')
            damage_incr = 0.0

    return damage_incr


def save_results_to_files(cfg, dic_panels):
    """

    Args:
        cfg:
        dic_panels:

    Returns:

    """

    # file_house ('house': Panel(attribute, wind_speed_steps, no_sims)
    with h5py.File(cfg.file_house, 'w') as hf:
        for item, value in dic_panels['house'].iteritems():
            hf.create_dataset(item, data=value)
    # dic_panels['house'].to_hdf(cfg.file_house, key='house', mode='w')

    for item in cfg.list_components:
        with h5py.File(getattr(cfg, 'file_{}'.format(item)), 'w') as hf:
            for key, chunk in dic_panels[item].iteritems():
                _group = hf.create_group(key)
                for _conn, value in chunk.iteritems():
                    _group.create_dataset(str(_conn), data=value)

        # # other components
        # for item in cfg.list_components:
        #     # file (key: Panel(attribute, wind_speed_steps, no_sims)
        #     hdf = pd.HDFStore(getattr(cfg, 'file_{}'.format(item)), mode='w')
        #     for key, value in dic_panels[item].iteritems():
        #         hdf[str(key)] = value


    # with warnings.catch_warnings():
    #
    #     warnings.simplefilter("ignore")
    #     #  warnings.filterwarnings('error')
    #
    #     # other components
    #     for item in cfg.list_components:
    #         # file (key: Panel(attribute, wind_speed_steps, no_sims)
    #         hdf = pd.HDFStore(getattr(cfg, 'file_{}'.format(item)), mode='w')
    #         for key, value in dic_panels[item].iteritems():
    #             hdf[str(key)] = value
    #
    #         hdf.close()

    # plot fragility and vulnerability curves
    frag_counted = fit_fragility_curves(cfg, dic_panels['house']['di'])
    if frag_counted:
        pd.DataFrame.from_dict(frag_counted).transpose().to_csv(cfg.file_curve)
    else:
        with open(cfg.file_curve, 'w') as fid:
            fid.write(', error, param1, param2\n')

    if cfg.flags['plot_fragility']:
        pass

    if cfg.flags['plot_vulnerability']:
        fitted_curve = fit_vulnerability_curve(cfg, dic_panels['house']['di'])
        if not os.path.isfile(cfg.file_curve):
            with open(cfg.file_curve, 'w') as fid:
                fid.write(', error, param1, param2\n')
        with open(cfg.file_curve, 'a') as f:
            pd.DataFrame.from_dict(fitted_curve).transpose().to_csv(f, header=None)

    if cfg.flags['plot_connection_damage']:

        for group_name, grouped in cfg.connections.groupby('group_name'):

            for id_sim, df_ in dic_panels['connection']['capacity'].loc[
                               grouped.index, cfg.wind_speed_steps - 1,
                               :].iterrows():
                file_name = os.path.join(cfg.path_output,
                                         '{}_id{}'.format(group_name,
                                                          id_sim))
                plot_heatmap(grouped,
                             df_.values,
                             vmin=cfg.heatmap_vmin,
                             vmax=cfg.heatmap_vmax,
                             vstep=cfg.heatmap_vstep,
                             xlim_max=cfg.house['length'],
                             ylim_max=cfg.house['width'],
                             file_name=file_name)


def show_results(self, output_folder=None, vRed=40, vBlue=80):
    if self.mplDict:
        self.mplDict['fragility'].axes.cla()
        self.mplDict['fragility'].axes.figure.canvas.draw()
        self.mplDict['vulnerability'].axes.cla()
        self.mplDict['vulnerability'].axes.figure.canvas.draw()
    if self.cfg.flags['dmg_plot_fragility']:
        self.plot_fragility(output_folder)
    if self.cfg.flags['plot_vulnerability']:
        self.plot_vulnerability(output_folder)
        output.plot_wind_event_show(self.cfg.no_sims,
                                    self.cfg.wind_speed_min,
                                    self.cfg.wind_speed_max,
                                    output_folder)
    self.plot_connection_damage(vRed, vBlue)


def set_logger(path_cfg, logging_level=None):
    """
        
    Args:
        path_cfg: path of configuration file 
        logging_level: Level of messages that will be logged
    Returns:
    """

    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')

    # create console handler and set level to WARNING
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.WARNING)
    logger.addHandler(ch)

    if logging_level:

        # create file handler
        path_logger = os.path.join(path_cfg, 'output')
        if not os.path.exists(path_logger):
            os.makedirs(path_logger)
        fh = logging.FileHandler(os.path.join(path_logger, 'log.txt'), mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        try:
            fh.setLevel(getattr(logging, logging_level.upper()))
        except (AttributeError, TypeError):
            logging.warning('{} is not valid; WARNING is set instead'.format(
                logging_level))
            fh.setLevel(logging.WARNING)


def process_commandline():
    usage = '%prog -c <config_file> [-v <logging_level>]'
    parser = OptionParser(usage=usage, version=VERSION_DESC)
    parser.add_option("-c", "--config",
                      dest="config_file",
                      help="read configuration from FILE",
                      metavar="FILE")
    parser.add_option("-v", "--verbose",
                      dest="verbose",
                      default=False,
                      metavar="logging_level",
                      help="set logging level")
    return parser


def main():
    parser = process_commandline()

    (options, args) = parser.parse_args()

    if options.config_file:
        path_cfg = os.path.dirname(os.path.realpath(options.config_file))
        set_logger(path_cfg, options.verbose)

        conf = Config(cfg_file=options.config_file)
        _ = simulate_wind_damage_to_houses(conf)

    else:
        print('Error: Must provide a config file to run')
        parser.print_help()

if __name__ == '__main__':
    main()
