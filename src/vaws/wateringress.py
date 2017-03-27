"""
    Water Ingress Module - damage and costing aspects of water ingress based
    off Martin's work.
"""
import scipy.stats
import numpy as np


def get_watercost_for_damage_at_wi(damage_name, water_ingress):
    wiarr = cached_water_costs[damage_name]
    last_valid_row = wiarr[0]
    for wi in wiarr:
        if water_ingress >= wi.wi:
            last_valid_row = wi
    return last_valid_row




def cal_water_ingress_costing_given_damage(damage_index, wind_speed, water_groups,
                                         out_file=None):
    water_ratio = (di, wind_speed)
    water_ingress_perc = water_ratio * 100.0

    damage_name = 'WI only'
    for ctg in water_groups:
        if ctg.water_ingress_order > 0 and ctg.result_percent_damaged > 0:
            damage_name = ctg.costing.costing_name
            break

    watercosting = get_watercost_for_damage_at_wi(damage_name,
                                                  water_ingress_perc)
    # water_ingress_cost = 0.0
    if watercosting.formula_type == 1:
        water_ingress_cost = watercosting.base_cost * (
            watercosting.coeff1 * di ** 2 + watercosting.coeff2 * di +
            watercosting.coeff3)
    else:
        water_ingress_cost = watercosting.base_cost * watercosting.coeff1 * \
                             di ** watercosting.coeff2

    watercosting_str = '{}'.format(watercosting)

    if out_file:
        out_file.write('\n%f,%f,%f,%s,%f,%s' % (wind_speed, di, water_ratio,
                                                damage_name, water_ingress_cost,
                                                watercosting))
    return water_ratio, damage_name, water_ingress_cost, watercosting_str

