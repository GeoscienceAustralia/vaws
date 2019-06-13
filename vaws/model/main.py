import sys
import os
import time
import logging.config
# import json

import h5py
import numpy as np
from optparse import OptionParser

from vaws.model.house import House
from vaws.model.config import Config
from vaws.model.curve import fit_fragility_curves, fit_vulnerability_curve
from vaws.model.output import plot_heatmap
from vaws.model.version import VERSION_DESC

DT = h5py.special_dtype(vlen=str)


def simulate_wind_damage_to_houses(cfg, call_back=None):
    """

    Args:
        cfg: instance of Config class
        call_back: used by gui

    Returns:
        elapsed: float: elapsed time
        bucket: list: list of results

    """

    logger = logging.getLogger(__name__)

    # simulator main_loop
    tic = time.time()

    damage_increment = 0.0
    bucket = init_bucket(cfg)

    # generate instances of house
    list_house_damage = [House(cfg, i + cfg.random_seed)
                         for i in range(cfg.no_models)]

    for ispeed, wind_speed in enumerate(cfg.wind_speeds):

        results_by_speed = []

        for ihouse, house in enumerate(list_house_damage):

            logger.debug('model {}'.format(ihouse))

            house.damage_increment = damage_increment

            result = house.run_simulation(wind_speed)

            results_by_speed.append(result)

        bucket = update_bucket(cfg, bucket, results_by_speed, ispeed)
        damage_increment = compute_damage_increment(bucket, ispeed)

        logger.debug('damage index increment {}'.format(damage_increment))
        percent_done = 100.0 * (ispeed + 1) / len(cfg.wind_speeds)
        int_percent = int(percent_done)
        if not call_back:
            sys.stdout.write(
                ('=' * int_percent) + ('' * (100 - int_percent)) +
                ("\r [ %d" % percent_done + "% ] "))
            sys.stdout.flush()
        else:
            if not call_back(int_percent):  # stop triggered
                return

    save_results_to_files(cfg, bucket)

    elapsed = time.time()-tic
    logging.info('Simulation completed: {:.4f}'.format(elapsed))

    return elapsed, bucket


def init_bucket(cfg):

    bucket = {'house': {}}
    for att, flag_time in cfg.house_bucket:
        if flag_time:
            bucket['house'][att] = np.zeros(
                shape=(cfg.wind_speed_steps, cfg.no_models), dtype=float)
        else:
            # if att in cfg.att_non_float:
            #     bucket['house'][att] = np.zeros(shape=(1, cfg.no_models),
            #                                     dtype=str)
            # else:
            bucket['house'][att] = np.zeros(shape=(1, cfg.no_models),
                                            dtype=float)

    # components: group, connection, zone, coverage
    for comp in cfg.list_components:
        bucket[comp] = {}
        if comp == 'debris':
            for att, _ in getattr(cfg, '{}_bucket'.format(comp)):
                bucket[comp][att] = np.zeros(
                    shape=(cfg.wind_speed_steps, cfg.no_models), dtype=object)
        else:
            for att, flag_time in getattr(cfg, '{}_bucket'.format(comp)):
                bucket[comp][att] = {}
                try:
                    for item in getattr(cfg, 'list_{}s'.format(comp)):
                        if flag_time:
                            bucket[comp][att][item] = np.zeros(
                                shape=(cfg.wind_speed_steps, cfg.no_models), dtype=float)
                        else:
                            bucket[comp][att][item] = np.zeros(
                                shape=(1, cfg.no_models), dtype=float)
                except TypeError:
                    pass
    return bucket


def update_bucket(cfg, bucket, results_by_speed, ispeed):

    for att, flag_time in cfg.house_bucket:
        if flag_time:
            bucket['house'][att][ispeed] = [x['house'][att] for x in results_by_speed]

    for comp in cfg.list_components:
        if comp == 'debris':
            for att, flag_time in cfg.debris_bucket:
                bucket['debris'][att][ispeed] = [x['debris'][att] for x in results_by_speed]
        else:
            for att, flag_time in getattr(cfg, '{}_bucket'.format(comp)):
                if flag_time:
                    for item, value in bucket[comp][att].items():
                        value[ispeed] = [x[comp][att][item] for x in
                                         results_by_speed]

    # save time invariant attribute
    if ispeed == cfg.wind_speed_steps-1:

        for att, flag_time in cfg.house_bucket:
            if not flag_time:
                bucket['house'][att] = [x['house'][att] for x in results_by_speed]

        for comp in cfg.list_components:
            for att, flag_time in getattr(cfg, '{}_bucket'.format(comp)):
                if not flag_time:
                    for item, value in bucket[comp][att].items():
                        value[0] = [x[comp][att][item] for x in results_by_speed]

    return bucket


def compute_damage_increment(bucket, ispeed):

    logger = logging.getLogger(__name__)

    # compute damage index increment
    damage_increment = 0.0  # default value

    if ispeed:
        damage_increment = (bucket['house']['di'][ispeed].mean(axis=0) -
                            bucket['house']['di'][ispeed - 1].mean(axis=0))

        if damage_increment < 0:
            logger.warning('damage increment is less than zero')
            damage_increment = 0.0

    return damage_increment


def save_results_to_files(cfg, bucket):
    """

    Args:
        cfg:
        bucket:

    Returns:

    """

    # file_house
    with h5py.File(cfg.file_results, 'w') as hf:

        hf.create_dataset('wind_speeds', data=cfg.wind_speeds)

        group = hf.create_group('house')
        for att, value in bucket['house'].items():
            group.create_dataset(att, data=value)

        for comp in cfg.list_components:
            group = hf.create_group(comp)
            if comp == 'debris':
                for att, value in bucket[comp].items():
                    group.create_dataset(att, data=value, dtype=DT)
            else:
                for att, chunk in bucket[comp].items():
                    subgroup = group.create_group(att)
                    for item, value in chunk.items():
                        subgroup.create_dataset(str(item), data=value)

        # fragility curves
        if cfg.no_models > 1:
            frag_counted, df_counted = fit_fragility_curves(cfg, bucket['house']['di'])

            group = hf.create_group('fragility')
            bucket['fragility'] = {}
            group.create_dataset('counted', data=df_counted)
            try:
                group['counted'].attrs.create('column_names', df_counted.columns.tolist())
            except TypeError:
                print(df_counted.columns.tolist())

            bucket['fragility']['counted'] = df_counted

            for fitting in ['MLE', 'OLS']:
                if frag_counted[fitting]:
                    for key, value in frag_counted[fitting].items():
                        for sub_key, sub_value in value.items():
                            name = '{}/{}/{}'.format(fitting, key, sub_key)
                            group.create_dataset(name=name, data=sub_value)
                            bucket['fragility'].setdefault(fitting, {}).setdefault(key, {})[sub_key] = sub_value

        # vulnerability curves
        fitted_curve = fit_vulnerability_curve(cfg, bucket['house']['di'])

        group = hf.create_group('vulnerability')
        bucket['vulnerability'] = {}
        for key, value in fitted_curve.items():
            for sub_key, sub_value in value.items():
                group.create_dataset('{}/{}'.format(key, sub_key),
                                     data=sub_value)
                bucket['vulnerability'].setdefault(key, {})[
                    sub_key] = sub_value

    if cfg.flags['save_heatmaps']:

        for group_name, grouped in cfg.connections.groupby('group_name'):

            for id_sim in range(cfg.no_models):

                value = np.array([bucket['connection']['capacity'][i]
                                 for i in grouped.index])[:, 0, id_sim]

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
    """debug, info, warning, error, critical"""

    config_dic = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "[%(levelname)s] %(name)s: %(message)s"
            }
        },

        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },

        },

        "loggers": {
        },

        "root": {
            "level": "INFO",
            "handlers": ["console"]
        }
    }

    if logging_level:

        try:
            level = getattr(logging, logging_level.upper())
        except (AttributeError, TypeError):
            logging_level = 'DEBUG'
            level = 'DEBUG'
        finally:
            file_log = os.path.join(path_cfg, 'output', '{}.log'.format(logging_level))
            added_file_handler = {"added_file_handler": {
                                  "class": "logging.handlers.RotatingFileHandler",
                                  "level": level,
                                  "formatter": "simple",
                                  "filename": file_log,
                                  "encoding": "utf8",
                                  "mode": "w"}
                            }
            config_dic['handlers'].update(added_file_handler)
            config_dic['root']['handlers'].append('added_file_handler')
            config_dic['root']['level'] = "DEBUG"

    logging.config.dictConfig(config_dic)


def process_commandline():
    usage = '%prog -c <config_file> [-v <logging_level>]'
    parser = OptionParser(usage=usage, version=VERSION_DESC)
    parser.add_option("-c", "--config",
                      dest="config_file",
                      help="read configuration from FILE",
                      metavar="FILE")
    parser.add_option("-v", "--verbose",
                      dest="verbose",
                      default=None,
                      metavar="logging_level",
                      help="set logging level")
    return parser


def main():
    parser = process_commandline()

    (options, args) = parser.parse_args()

    if options.config_file:
        path_cfg = os.path.dirname(os.path.realpath(options.config_file))
        set_logger(path_cfg, options.verbose)

        conf = Config(file_cfg=options.config_file)
        _ = simulate_wind_damage_to_houses(conf)
    else:
        print('Error: Must provide a config file to run')
        parser.print_help()

if __name__ == '__main__':
    main()
