"""
    Connection Module - reference storage Connections
        - loaded from database
        - imported from '../data/houses/subfolder'
"""
from sqlalchemy import Integer, String, Column, ForeignKey, orm
from sqlalchemy.orm import relation, backref
import database
import influence

use_struct_pz_for = ['rafter', 'piersgroup', 'wallracking']



def reset_connection(house_conn):
    """

    Args:
        house_conn:

    Returns:

    """
    # assert huse_conn
    house_conn.result_strength = 0.0
    self.result_deadload = 0.0
    self.result_failure_v_raw = 0.0
    self.result_damaged = False
    self.result_damaged_report = {}
    self.result_damage_distributed = False

def calc_connection_damage(house_conn, V, result_load, infl_by_zone_map, use_struct_pz=None):
    if self.result_damaged:
        return
    if use_struct_pz is None:
        use_struct_pz = self.ctype.group.group_name in use_struct_pz_for
    self.ctype.incr_damaged()
    self.result_damaged = True
    self.result_damaged_report['load'] = result_load
    infls_arr = []
    for z in sorted(infl_by_zone_map):
        infls_dict = {}
        infl = infl_by_zone_map[z]
        infls_dict['infl'] = infl
        area = float(z.result_effective_area)
        infls_dict['area'] = area
        pz = z.result_pz_struct if use_struct_pz else z.result_pz
        infls_dict['pz'] = pz
        infls_dict['load'] = infl * area * pz
        infls_dict['name'] = z.zone_name
        infls_arr.append(infls_dict)
    self.result_damaged_report['infls'] = infls_arr
    self.result_damage_distributed = False
    self.result_failure_v_i += 1
    self.result_failure_v_raw = V
    num_ = V + float(self.result_failure_v_i - 1) * self.result_failure_v
    denom_ = self.result_failure_v_i
    self.result_failure_v = num_ / denom_

#    def assing_connection_strengths(self, mean_factor, cov_factor):

def assign_connection_strengths(house_conns, mean_factor, cov_factor):
    """
    FIXME: it may put into connection class not outside.
    Args:
        house_conns:
        mean_factor:
        cov_factor:

    Returns:

    """
    for conn in house_conns:
        conn.result_strength = conn.ctype.sample_strength(mean_factor,
                                                          cov_factor)


def assign_connection_deadloads(house_conns):
    """
    FIXME: it may put into connection class not outside.

    Args:
        house_conns:

    Returns:

    """
    for conn in house_conns:
        conn.result_deadload = conn.ctype.sample_deadload()


def calc_connection_loads(V, ctg, dmg_map, inflZonesByConn, connByTypeMap):
    """
    FIXME: it may be inside of the class rather than outside.
    Args:
        V: wind speed
        ctg: connection type group
        dmg_map: damage map
        inflZonesByConn: influence factor by connection
        connByTypeMap:

    Returns:
        damage_conn_type:

    """
    if not ctg.enabled or len(ctg.conn_types) == 0:
        return dict()

    # hard coded for optimization
    use_struct_pz = ctg.group_name in use_struct_pz_for

    damage_conn_type = dict()
    # for all connection types belonging to the specified group
    for ct in ctg.conn_types:

        # for all connections belonging to the current connection type
        for c in connByTypeMap[ct]:
            if c.result_damaged:
                continue

            # sum loads from influencing zones
            result_load = 0.0
            dict_ = inflZonesByConn[c]
            for z in dict_:
                if z.result_effective_area > 0.0:
                    if use_struct_pz:
                        result_load += (
                            dict_[z] * z.result_effective_area *
                            z.result_pz_struct)
                    else:
                        result_load += (
                            dict_[z] * z.result_effective_area * z.result_pz)

            # add dead load and if load is negative check for strength
            # failure
            result_load += c.result_deadload
            if result_load < 0.0:
                if -result_load >= c.result_strength:
                    # now a function so that collapse can record proper
                    # audit info.
                    c.damage(V, result_load, inflZonesByConn[c], use_struct_pz)

                    # if the damaged wind speed is less than current type min
                    # then store (for reporting of min damage speeds by type)
                    curr = dmg_map[ct.connection_type]
                    dmg_map[ct.connection_type] = min(V, curr)

                    # report percentage of connection type damaged at this V
                    # to file.
        # file_damage.write(',')
        # file_damage.write(str(ct.perc_damaged() * 100.0))
        damage_conn_type[ct.connection_type] = ct.perc_damaged() * 100.0

    return damage_conn_type
