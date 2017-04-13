import sys
import os
import time
import logging
import warnings

# from pathos.multiprocessing import ProcessingPool
import pandas as pd
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

    if cfg.parallel:
        cfg.parallel = False
        logging.info('parallel running is not implemented yet, '
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

    logging.info('Time taken for simulation {}'.format(time.time()-tic))

    return time.time()-tic, dic_panels


def init_panels(cfg):

    dic_panels = dict()

    # house
    list_atts = (cfg.list_house_bucket + cfg.list_house_damage_bucket +
                 cfg.list_debris_bucket)
    dic_panels['house'] = pd.Panel(dtype=float,
                                   items=list_atts,
                                   major_axis=range(cfg.wind_speed_steps),
                                   minor_axis=range(cfg.no_sims))
    # components
    for item in cfg.list_components:
        for att in getattr(cfg, 'list_{}_bucket'.format(item)):
            dic_panels.setdefault(item, {})[att] = \
                pd.Panel(dtype=float,
                         items=getattr(cfg, 'list_{}s'.format(item)),
                         major_axis=range(cfg.wind_speed_steps),
                         minor_axis=range(cfg.no_sims))

    return dic_panels


def update_panels(cfg, dic_, list_results_by_speed, ispeed):

    # house
    list_atts = (cfg.list_house_bucket + cfg.list_house_damage_bucket +
                 cfg.list_debris_bucket)
    for att in list_atts:
        dic_['house'][att].at[ispeed] = [x['house'][att] for x in
                                          list_results_by_speed]

    # components
    for item in cfg.list_components:
        for key, value in dic_[item].iteritems():
            for att in value.items.tolist():
                value[att].at[ispeed] = [x[item].at[att, key] for x in
                                          list_results_by_speed]

    # compute damage index increment
    damage_incr = 0.0  # default value

    if ispeed:
        damage_incr = (dic_['house']['di'].loc[ispeed].mean(axis=0) -
                       dic_['house']['di'].loc[ispeed - 1].mean(axis=0))

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

    # file_house ('house': Panel(attribute, wind_speed_steps, no_sims)
    dic_panels['house'].to_hdf(cfg.file_house, key='house', mode='w')

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")
        #  warnings.filterwarnings('error')

        # other components
        for item in cfg.list_components:
            # file (key: Panel(attribute, wind_speed_steps, no_sims)
            hdf = pd.HDFStore(getattr(cfg, 'file_{}'.format(item)), mode='w')
            for key, value in dic_panels[item].iteritems():
                hdf[str(key)] = value

            hdf.close()

    # plot fragility and vulnerability curves
    frag_counted = fit_fragility_curves(cfg, dic_panels['house']['di'])
    if frag_counted:
        pd.DataFrame.from_dict(frag_counted).transpose().to_csv(cfg.file_curve)
    else:
        with open(cfg.file_curve, 'w') as fid:
            fid.write(', error, param1, param2\n')

    fitted_curve = fit_vulnerability_curve(cfg, dic_panels['house']['di'])
    with open(cfg.file_curve, 'a') as f:
        pd.DataFrame.from_dict(fitted_curve).transpose().to_csv(f, header=None)

    if cfg.flags['plot_connection_damage']:

        for group_name, grouped in cfg.df_connections.groupby('group_name'):

            for id_sim, df_ in dic_panels['connection']['capacity'].loc[
                               grouped.index, cfg.wind_speed_steps - 1, :].iterrows():

                file_name = os.path.join(cfg.path_output,
                                         '{}_id{}'.format(group_name, id_sim))
                plot_heatmap(grouped,
                             df_.values,
                             vmin=cfg.red_v,
                             vmax=cfg.blue_v,
                             vstep=21,
                             file_name=file_name)


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


def set_logger(config, logging_level=logging.INFO):
    """
        
    Args:
        options: Logging configuration 
        logging_level: Level of messages that will be logged
    Returns:
    """

    logger = logging.getLogger()
    file_logger = os.path.join(config.path_cfg, 'output/log.txt')
    file_handler = logging.FileHandler(filename=file_logger, mode='w')
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    try:
        file_handler.setLevel(getattr(logging, logging_level.upper()))
    except (AttributeError, TypeError):
        logging.info('{} is not a logging level; DEBUG is set instead'.format(
            logging_level))
        file_handler.setLevel(logging.DEBUG)


def process_commandline():
    usage = '%prog -c <config_file> [-v <logging_level>]'
    parser = OptionParser(usage=usage, version=VERSION_DESC)
    parser.add_option("-c", "--config",
                      dest="config_filename",
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

    if options.config_filename:
        conf = Config(cfg_file=options.config_filename)

        if options.verbose:
            set_logger(conf, options.verbose)

        _ = simulate_wind_damage_to_houses(conf)
    else:
        print('Error: Must provide a config file to run')
        parser.print_help()

if __name__ == '__main__':
    main()
