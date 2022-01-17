import sys
import os
import time
import logging.config
# import json

import h5py
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from dask.distributed import Client, LocalCluster

from vaws.model.house import House, run_simulation
from vaws.model.config import Config, WIND_DIR, DEBRIS_ITEMS, WATER_INGRESS_ITEMS
from vaws.model.curve import fit_fragility_curves, fit_vulnerability_curve
from vaws.model.output import plot_heatmap
from vaws.model.version import VERSION_DESC

DT = h5py.special_dtype(vlen=str)


def simulate_wind_dmg_to_houses(cfg, call_back=None):
    """

    Args:
        cfg: instance of Config class
        call_back: used by gui

    Returns:
        elapsed: float: elapsed time
        bucket: list: list of results

    """

    logger = logging.getLogger(__name__)
    cluster = LocalCluster(threads_per_worker=1)
    client = Client(cluster)

    # simulator main_loop
    tic = time.time()

    di_prior = 0.0
    dmg_increment = 0.0
    prop_water_ingress = 0.0
    #bucket = init_bucket(cfg)

    # generate instances of house
    houses = [House(cfg, i + cfg.random_seed) for i in range(cfg.no_models)]

    #water_ingress_perc = np.zeros(shape=(cfg.no_models, cfg.wind_speed_steps))


    dump = []

    dmg_increment = 0
    di = 0
    prop_water_ingress = 0

    for ispeed, wind_speed in enumerate(cfg.wind_speeds):

        results = []

        for ihouse, house in enumerate(houses):


            #logger.debug(f'model {ihouse}')

            result = client.submit(run_simulation, house, wind_speed, ispeed, dmg_increment, prop_water_ingress, cfg)
            #result = run_simulation(house, wind_speed, ispeed, dmg_increment, prop_water_ingress, cfg)

            results.append(result)

        #bucket = update_bucket(cfg, bucket, results, ispeed)

        #results = client.gather(results)

        #dmg_increment, prop_water_ingress = compute_dmg_increment_prop_water_ingress(results, cfg)
        #out = compute_dmg_increment_prop_water_ingress(results, ispeed, out[1], cfg)

        prop_water_ingress = client.submit(compute_prop_water_ingress, results, cfg)

        dump.append(prop_water_ingress)

        #logger.debug(f'dmg index increment {dmg_increment}')
        #logger.debug(f'prop. water ingress {prop_water_ingress}')

        #percent_done = 100.0 * (ispeed + 1) / len(cfg.wind_speeds)
        #int_percent = int(percent_done)
        #if not call_back:
        #    sys.stdout.write(
        #        ('=' * int_percent) + ('' * (100 - int_percent)) +
        #        ("\r [ %d" % percent_done + "% ] "))
        #    sys.stdout.flush()
        #else:
        #    if not call_back(int_percent):  # stop triggered
        #        return

    _ = client.gather(dump)

    client.close()

    #save_results_to_files(cfg, bucket)

    elapsed = time.time()-tic
    logging.info(f'Simulation completed: {elapsed:.4f}')

    return elapsed


def init_bucket(cfg):

    bucket = {'house': {}}
    for att, flag_time in cfg.house_bucket:
        if flag_time:
            if att == 'repair_cost_by_scenario':
                bucket['house'][att] = {}
                for item in cfg.costings.keys():
                    bucket['house'][att][item] = np.zeros(
                                shape=(cfg.wind_speed_steps, cfg.no_models), dtype=float)
            else:
                bucket['house'][att] = np.zeros(
                    shape=(cfg.wind_speed_steps, cfg.no_models), dtype=float)
        else:
            bucket['house'][att] = np.zeros(shape=(1, cfg.no_models),
                                            dtype=float)

    # components: group, connection, zone, coverage
    for comp in cfg.list_components:
        bucket[comp] = {}
        if comp == 'debris':
            for att, _ in getattr(cfg, f'{comp}_bucket'):
                bucket[comp][att] = np.zeros(
                    shape=(cfg.wind_speed_steps, cfg.no_models), dtype=object)
        else:
            for att, flag_time in getattr(cfg, f'{comp}_bucket'):
                bucket[comp][att] = {}
                try:
                    for item in getattr(cfg, f'list_{comp}s'):
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
            if att == 'repair_cost_by_scenario':
                for item, value in bucket['house'][att].items():
                    try:
                        value[ispeed] = [x['house'][att][item] for x in
                                         results_by_speed]
                    except KeyError:
                        pass
            else:
                bucket['house'][att][ispeed] = [x['house'][att] for x in results_by_speed]

    for comp in cfg.list_components:
        if comp == 'debris':
            for att, flag_time in cfg.debris_bucket:
                bucket['debris'][att][ispeed] = [x['debris'][att] for x in results_by_speed]
        else:
            for att, flag_time in getattr(cfg, f'{comp}_bucket'):
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
            for att, flag_time in getattr(cfg, f'{comp}_bucket'):
                if not flag_time:
                    for item, value in bucket[comp][att].items():
                        value[0] = [x[comp][att][item] for x in results_by_speed]

    return bucket


def compute_dmg_increment(cfg, bucket, ispeed):

    logger = logging.getLogger(__name__)

    # compute dmg index increment
    dmg_increment = 0.0  # default value

    if cfg.flags['debris_vulnerability']:

        if ispeed:
            dmg_increment = cfg.debris_vulnerability.cdf(cfg.wind_speeds[ispeed]) - \
                               cfg.debris_vulnerability.cdf(cfg.wind_speeds[ispeed-1])
    else:

        if ispeed:
            dmg_increment = (bucket['house']['di'][ispeed].mean(axis=0) -
                                bucket['house']['di'][ispeed - 1].mean(axis=0))

            if dmg_increment < 0:
                logger.warning('dmg increment is less than zero')
                dmg_increment = 0.0

    return dmg_increment


def compute_prop_water_ingress(results, cfg):

    if cfg.flags['water_ingress']:
        prop_water_ingress = sum([x['water_ingress_perc'] > 0 for x in results])/cfg.no_models
    else:
        prop_water_ingress = 0.0  # default value


    return prop_water_ingress



def compute_dmg_increment_prop_water_ingress(results, ispeed, di_prior, cfg):

    # compute dmg index increment
    if cfg.flags['debris_vulnerability']:
        di = cfg.debris_vulnerability.cdf(cfg.wind_speeds[ispeed])
        di_prior = cfg.debris_vulnerability.cdf(cfg.wind_speeds[ispeed-1])
    else:
        di = sum([x['di'] for x in results])/cfg.no_models

    if ispeed:

        dmg_increment = di - di_prior

        if dmg_increment < 0:
            #logger.warning('dmg increment is less than zero')
            dmg_increment = 0.0
    else:
        dmg_increment = 0.0  # default value

    if cfg.flags['water_ingress']:
        prop_water_ingress = sum([x['water_ingress_perc'] > 0 for x in results])/cfg.no_models
    else:
        prop_water_ingress = 0.0  # default value

    return dmg_increment, di, prop_water_ingress




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
            if att == 'repair_cost_by_scenario':
                subgroup = group.create_group(att)
                for item, value in bucket['house'][att].items():
                    subgroup.create_dataset(item, data=value)
            else:
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
            column_names = ','.join([x for x in df_counted.columns.tolist()])
            group['counted'].attrs['column_names'] = column_names
            bucket['fragility']['counted'] = df_counted

            for fitting in ['MLE', 'OLS']:
                if frag_counted[fitting]:
                    for key, value in frag_counted[fitting].items():
                        for sub_key, sub_value in value.items():
                            name = f'{fitting}/{key}/{sub_key}'
                            group.create_dataset(name=name, data=sub_value)
                            bucket['fragility'].setdefault(fitting, {}).setdefault(key, {})[sub_key] = sub_value

        # vulnerability curves
        fitted_curve = fit_vulnerability_curve(cfg, bucket['house']['di'])

        group = hf.create_group('vulnerability')
        bucket['vulnerability'] = {}
        group.create_dataset('mean_di', data=np.mean(bucket['house']['di'], axis=1))
        # bucket['fragility']['counted'] = df_counted

        for key, value in fitted_curve.items():
            for sub_key, sub_value in value.items():
                group.create_dataset(f'{key}/{sub_key}', data=sub_value)
                bucket['vulnerability'].setdefault(key, {})[
                    sub_key] = sub_value

    # save input
        group = hf.create_group('input')
        for item in ['no_models', 'model_name', 'random_seed', 'wind_speed_min', 'wind_speed_max',
                     'wind_speed_increment', 'file_wind_profiles', 'regional_shielding_factor']:
            group.create_dataset(item, data=getattr(cfg, item))
        group.create_dataset('wind_direction', data=WIND_DIR[cfg.wind_dir_index])

        for item in ['water_ingress', 'debris', 'debris_vulnerability' , 'save_heatmaps',
                     'differential_shielding', 'wall_collapse']:
            group.create_dataset(name=f'flags/{item}', data=getattr(cfg, 'flags')[item])

        for item in ['vmin', 'vmax', 'vstep']:
            group.create_dataset(f'heatmap_{item}', data=getattr(cfg, f'heatmap_{item}'))

        for item in ['states', 'thresholds']:
            _item = f'fragility_{item}'
            _value = getattr(cfg, f'fragility_{item}')
            value =  ','.join([str(x) for x in _value])
            group.create_dataset(_item, data=value)

        if cfg.flags['debris']:
            for item in DEBRIS_ITEMS:
                group.create_dataset(item, data=getattr(cfg, item))

        if cfg.flags['water_ingress']:
            for item in WATER_INGRESS_ITEMS[:-1]:
                _item = f'water_ingress_{item}'
                _value = getattr(cfg, f'water_ingress_{item}')
                value =  ','.join([str(x) for x in _value])
                group.create_dataset(_item, data=value)
            group.create_dataset('water_ingress_di_threshold_wi', data=cfg.water_ingress_di_threshold_wi)

        if cfg.flags['debris_vulnerability']:
            for item in ['function' , 'param1' , 'param2']:
                value = getattr(cfg, 'debris_vuln_input')[item]
                group.create_dataset(f'debris_vuln_input/{item}', data=value)

    # save water ingress proprtion
        '''
        water_ingress_perc = bucket['house']['water_ingress_perc'].copy()
        water_ingress_perc[water_ingress_perc > 0] = 1

        plt.figure()
        pngfile = os.path.join(cfg.path_output, 'comp_prop_water_ingress.png')
        plt.plot(cfg.water_ingress_ref_prop_v, cfg.water_ingress_ref_prop, '--')
        plt.plot(cfg.wind_speeds, water_ingress_perc.sum(axis=1)/cfg.no_models, '-')
        plt.legend(['Target', 'Simulation'])
        plt.xlabel('Wind speed (m/s)')
        plt.ylabel('Prop. of houses dmgd due to water ingress')
        plt.savefig(pngfile)
        plt.close('all')
        '''
    if cfg.flags['save_heatmaps']:

        for group_name, grouped in cfg.connections.groupby('group_name'):

            for id_sim in range(cfg.no_models):

                value = np.array([bucket['connection']['capacity'][i]
                                 for i in grouped.index])[:, 0, id_sim]

                file_name = os.path.join(cfg.path_output,
                                         f'{group_name}_id{id_sim}')
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
            file_log = os.path.join(path_cfg, 'output', f'{logging_level}.log')
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
    parser = ArgumentParser()
    parser.add_argument("-c", "--config",
                        dest="config_file",
                        help="file to read the config from")
    parser.add_argument("-v", "--verbose",
                        dest="verbose",
                        default=None,
                        metavar="logging_level",
                        help="set logging level")
    parser.add_argument('--version',
                        action='version',
                        version=VERSION_DESC)
    return parser


def main():
    parser = process_commandline()

    args = parser.parse_args()

    if args.config_file:
        path_cfg = os.path.dirname(os.path.realpath(args.config_file))
        set_logger(path_cfg, args.verbose)

        conf = Config(file_cfg=args.config_file)
        _ = simulate_wind_dmg_to_houses(conf)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
