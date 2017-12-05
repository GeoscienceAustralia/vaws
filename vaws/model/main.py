import sys
import os
import time
import logging

import pandas as pd
import h5py
from numpy import zeros, array, string_
from optparse import OptionParser

from vaws.model.house_damage import HouseDamage
from vaws.model.config import Config
from vaws.model.curve import fit_fragility_curves, fit_vulnerability_curve
from vaws.model.output import plot_heatmap
from vaws.model.version import VERSION_DESC


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
    bucket = init_bucket(cfg)

    logging.info('Starting simulation in serial')

    # generate instances of house_damage
    list_house_damage = [HouseDamage(cfg, i + cfg.random_seed)
                         for i in range(cfg.no_models)]

    for ispeed, wind_speed in enumerate(cfg.speeds):

        results_by_speed = []

        for ihouse, house_damage in enumerate(list_house_damage):

            logging.info('model {}'.format(ihouse))

            house_damage.house.debris.no_items_mean = damage_incr

            result = house_damage.run_simulation(wind_speed)

            results_by_speed.append(result)

        damage_incr = update_bucket(cfg, bucket, results_by_speed, ispeed)

        logging.debug('damage index increment {}'.format(damage_incr))

        if not call_back:
            sys.stdout.write('{} out of {} completed \n'.format(
                ispeed + 1, len(cfg.speeds)))
            sys.stdout.flush()
        else:
            percent_done = 100.0 * (ispeed + 1) / len(cfg.speeds)
            if not call_back(int(percent_done)):
                return

    save_results_to_files(cfg, bucket)

    elapsed = time.time()-tic
    logging.info('Time taken for simulation {}'.format(elapsed))

    return elapsed, bucket


def init_bucket(cfg):

    bucket = {item: {} for item in ['house', 'house_damage', 'debris']}

    for att in cfg.house_bucket:
        if att in cfg.att_non_float:
            bucket['house'][att] = zeros(shape=(1, cfg.no_models), dtype=str)
        else:
            bucket['house'][att] = zeros(shape=(1, cfg.no_models), dtype=float)

    for item in ['house_damage', 'debris']:
        bucket[item] = {}
        for att in getattr(cfg, '{}_bucket'.format(item)):
            if att in cfg.att_time_invariant:
                bucket[item][att] = zeros(
                    shape=(1, cfg.no_models), dtype=float)
            else:
                bucket[item][att] = zeros(
                    shape=(cfg.wind_speed_steps, cfg.no_models), dtype=float)

    # components: group, connection, zone
    for comp in cfg.list_components:
        bucket[comp] = {}
        for att in getattr(cfg, '{}_bucket'.format(comp)):
            bucket[comp][att] = {}
            try:
                for item in getattr(cfg, 'list_{}s'.format(comp)):
                    if att in cfg.att_time_invariant:
                        bucket[comp][att][item] = zeros(
                            shape=(1, cfg.no_models), dtype=float)
                    else:
                        bucket[comp][att][item] = zeros(
                            shape=(cfg.wind_speed_steps, cfg.no_models), dtype=float)
            except TypeError:
                pass
    return bucket


def update_bucket(cfg, bucket, results_by_speed, ispeed):

    for item in ['house_damage', 'debris']:
        for att in getattr(cfg, '{}_bucket'.format(item)):
            if att not in cfg.att_time_invariant:
                bucket[item][att][ispeed] = [x[item][att] for x in results_by_speed]

    for comp in cfg.list_components:
        for att, chunk in bucket[comp].iteritems():
            if att not in cfg.att_time_invariant:
                for item in chunk.iterkeys():
                    bucket[comp][att][item][ispeed] = \
                        [x[comp][att][item] for x in results_by_speed]

    # compute damage index increment
    damage_incr = 0.0  # default value

    if ispeed:
        damage_incr = (bucket['house_damage']['di'][ispeed].mean(axis=0) -
                       bucket['house_damage']['di'][ispeed - 1].mean(axis=0))

        if damage_incr < 0:
            logging.warning('damage increment is less than zero')
            damage_incr = 0.0

        if ispeed == cfg.wind_speed_steps - 1:

            # doing nothing but save house attributes
            for att in cfg.house_bucket:
                bucket['house'][att] = [x['house'][att] for x in results_by_speed]

            for item in ['house_damage', 'debris']:
                for att in getattr(cfg, '{}_bucket'.format(item)):
                    if att in cfg.att_time_invariant:
                        bucket[item][att] = [x[item][att] for x in results_by_speed]

            for comp in cfg.list_components:
                for att, chunk in bucket[comp].iteritems():
                    if att in cfg.att_time_invariant:
                        for item in chunk.iterkeys():
                            bucket[comp][att][item] = \
                                [x[comp][att][item] for x in results_by_speed]

    return damage_incr


def save_results_to_files(cfg, bucket):
    """

    Args:
        cfg:
        bucket:

    Returns:

    """

    # file_house
    with h5py.File(cfg.file_results, 'w') as hf:
        for item in ['house', 'house_damage', 'debris']:
            _group = hf.create_group(item)
            for att, value in bucket[item].iteritems():
                _group.create_dataset(att, data=value)

        # file_group, file_connection, file_zone
        for comp in cfg.list_components:
            _group = hf.create_group(comp)
            for att, chunk in bucket[comp].iteritems():
                _subgroup = _group.create_group(att)
                for item, value in chunk.iteritems():
                    _subgroup.create_dataset(str(item), data=value)

    # plot fragility and vulnerability curves

    if cfg.no_models > 1:
        frag_counted = fit_fragility_curves(cfg, bucket['house_damage']['di'])
        if frag_counted:
            pd.DataFrame.from_dict(frag_counted).transpose().to_csv(
                cfg.file_curve)
        else:
            with open(cfg.file_curve, 'w') as fid:
                fid.write(', error, param1, param2\n')

    fitted_curve = fit_vulnerability_curve(cfg, bucket['house_damage']['di'])
    if not os.path.isfile(cfg.file_curve):
        with open(cfg.file_curve, 'w') as fid:
            fid.write(', error, param1, param2\n')
    with open(cfg.file_curve, 'a') as f:
        pd.DataFrame.from_dict(fitted_curve).transpose().to_csv(f, header=None)

    if cfg.flags['save_heatmaps']:

        for group_name, grouped in cfg.connections.groupby('group_name'):

            for id_sim in range(cfg.no_models):

                value = array([bucket['connection']['capacity'][i][id_sim]
                               for i in grouped.index])

                file_name = os.path.join(cfg.path_output,
                                         '{}_id{}'.format(group_name,
                                                          id_sim))
                plot_heatmap(grouped,
                             value,
                             vmin=cfg.heatmap_vmin,
                             vmax=cfg.heatmap_vmax,
                             vstep=cfg.heatmap_vstep,
                             xlim_max=cfg.house['length'],
                             ylim_max=cfg.house['width'],
                             file_name=file_name)


def set_logger(path_cfg, logging_level=None):
    """
        
    Args:
        path_cfg: path of configuration file 
        logging_level: Level of messages that will be logged
    Returns:
    """

    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)
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
