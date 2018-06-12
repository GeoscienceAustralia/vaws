import sys
import os
import time
import logging

import h5py
import numpy as np
from optparse import OptionParser

from vaws.model.house import House
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

    # generate instances of house
    list_house_damage = [House(cfg, i + cfg.random_seed)
                         for i in range(cfg.no_models)]

    for ispeed, wind_speed in enumerate(cfg.speeds):

        results_by_speed = []

        for ihouse, house in enumerate(list_house_damage):

            logging.info('model {}'.format(ihouse))

            house.damage_incr = damage_incr

            result = house.run_simulation(wind_speed)

            results_by_speed.append(result)

        bucket = update_bucket(cfg, bucket, results_by_speed, ispeed)
        damage_incr = compute_damage_increment(bucket, ispeed)

        logging.debug('damage index increment {}'.format(damage_incr))

        if not call_back:
            sys.stdout.write('{} out of {} completed \n'.format(
                ispeed + 1, len(cfg.speeds)))
            sys.stdout.flush()
        else:
            percent_done = 100.0 * (ispeed + 1) / len(cfg.speeds)
            if not call_back(int(percent_done)):  # stop triggered
                return

    save_results_to_files(cfg, bucket)

    elapsed = time.time()-tic
    logging.info('Time taken for simulation {}'.format(elapsed))

    return elapsed, bucket


def init_bucket(cfg):

    bucket = {item: {} for item in ['house', 'debris']}

    for item in ['house', 'debris']:
        bucket[item] = {}
        for att in getattr(cfg, '{}_bucket'.format(item)):
            if att in cfg.att_time_invariant:
                if att in cfg.att_non_float:
                    bucket[item][att] = np.zeros(shape=(1, cfg.no_models),
                                                 dtype=str)
                else:
                    bucket[item][att] = np.zeros(shape=(1, cfg.no_models),
                                                 dtype=float)
            else:
                bucket[item][att] = np.zeros(
                    shape=(cfg.wind_speed_steps, cfg.no_models), dtype=float)

    # components: group, connection, zone, coverage
    for comp in cfg.list_components:
        bucket[comp] = {}
        for att in getattr(cfg, '{}_bucket'.format(comp)):
            bucket[comp][att] = {}
            try:
                for item in getattr(cfg, 'list_{}s'.format(comp)):
                    if att in cfg.att_time_invariant:
                        bucket[comp][att][item] = np.zeros(
                            shape=(1, cfg.no_models), dtype=float)
                    else:
                        bucket[comp][att][item] = np.zeros(
                            shape=(cfg.wind_speed_steps, cfg.no_models), dtype=float)
            except TypeError:
                pass
    return bucket


def update_bucket(cfg, bucket, results_by_speed, ispeed):

    for item in ['house', 'debris']:
        for att in getattr(cfg, '{}_bucket'.format(item)):
            if att not in cfg.att_time_invariant:
                bucket[item][att][ispeed] = [x[item][att] for x in results_by_speed]

    for comp in cfg.list_components:
        for att, chunk in bucket[comp].items():
            if att not in cfg.att_time_invariant:
                for item in chunk.iterkeys():
                    bucket[comp][att][item][ispeed] = \
                        [x[comp][att][item] for x in results_by_speed]

    # save time invariant attribute
    if ispeed == cfg.wind_speed_steps-1:

        for item in ['house', 'debris']:
            for att in getattr(cfg, '{}_bucket'.format(item)):
                if att in cfg.att_time_invariant:
                    bucket[item][att] = [x[item][att] for x in results_by_speed]

        for comp in cfg.list_components:
            for att, chunk in bucket[comp].items():
                if att in cfg.att_time_invariant:
                    for item in chunk.iterkeys():
                        bucket[comp][att][item] = \
                            [x[comp][att][item] for x in results_by_speed]

    return bucket


def compute_damage_increment(bucket, ispeed):

    # compute damage index increment
    damage_incr = 0.0  # default value

    if ispeed:
        damage_incr = (bucket['house']['di'][ispeed].mean(axis=0) -
                       bucket['house']['di'][ispeed - 1].mean(axis=0))

        if damage_incr < 0:
            logging.warning('damage increment is less than zero')
            damage_incr = 0.0

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

        for item in ['house', 'debris']:
            _group = hf.create_group(item)
            for att, value in bucket[item].items():
                _group.create_dataset(att, data=value)

        for comp in cfg.list_components:
            _group = hf.create_group(comp)
            for att, chunk in bucket[comp].items():
                _subgroup = _group.create_group(att)
                for item, value in chunk.items():
                    _subgroup.create_dataset(str(item), data=value)

        # fragility curves
        if cfg.no_models > 1:
            frag_counted = fit_fragility_curves(cfg, bucket['house']['di'])

            if frag_counted:
                _group = hf.create_group('fragility')
                for key, value in frag_counted.items():
                    for sub_key, sub_value in value.items():
                        _group.create_dataset('{}/{}'.format(key, sub_key),
                                              data=sub_value)

        # vulnerability curves
        fitted_curve = fit_vulnerability_curve(cfg, bucket['house']['di'])

        _group = hf.create_group('vulnearbility')
        for key, value in fitted_curve.items():
            for sub_key, sub_value in value.items():
                _group.create_dataset('{}/{}'.format(key, sub_key), data=sub_value)

    if cfg.flags['save_heatmaps']:

        for group_name, grouped in cfg.connections.groupby('group_name'):

            for id_sim in range(cfg.no_models):

                value = np.array([bucket['connection']['capacity'][i][id_sim]
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

    add_stream = True
    for h in list(logger.handlers):
        if h.__class__.__name__ == 'StreamHandler':
            add_stream = False
            break

    if add_stream:
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

        for h in list(logger.handlers):
            if h.__class__.__name__ == 'FileHandler':
                logger.removeHandler(h)
                break
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
        elapsed, _ = simulate_wind_damage_to_houses(conf)
        print('Simulation completed: {}'.format(elapsed))
    else:
        print('Error: Must provide a config file to run')
        parser.print_help()

if __name__ == '__main__':
    main()
