"""
    Zone Module - reference storage
        - loaded from database.
        - imported from '../data/houses/subfolder/zones.csv'
        - holds zone area and CPE means.
        - holds runtime sampled CPE per zone.
        - calculates Cpe pressure load from wind pressure.
"""

import logging

from stats import sample_gev


class Zone(object):

    flags_pressure = ['cpe', 'cpe_str']

    def __init__(self, zone_name=None, **kwargs):
        """
        Args:
            zone_name
        """

        assert isinstance(zone_name, str)
        self.name = zone_name

        self.area = None
        self.cpi_alpha = None
        self.cpe_mean = {}
        self.cpe_str_mean = {}
        self.cpe_eave_mean = {}
        self.is_roof_edge = {}

        default_attr = dict(area=self.area,
                            cpi_alpha=self.cpi_alpha,
                            # wall_dir=self.wall_dir,
                            cpe_mean=self.cpe_mean,
                            cpe_str_mean=self.cpe_str_mean,
                            cpe_eave_mean=self.cpe_eave_mean,
                            is_roof_edge=self.is_roof_edge)

        default_attr.update(kwargs)
        for key, value in default_attr.iteritems():
            setattr(self, key, value)

        # FIXME
        # should be a better way to identify wall zone
        if len(self.name) > 3 and self.name[0] == 'W':
            self.is_wall_zone = True
        else:
            self.is_wall_zone = False

        self.diff_shielding = None
        self.cpe = None
        self.cpe_str = None
        self.cpe_eave = None
        self.pressure_cpe = 0.0
        self.pressure_cpe_str = 0.0

    def sample_cpe(self, wind_dir_index, cpe_cov, cpe_k, big_a, big_b,
                   cpe_str_cov, cpe_str_k, big_a_str, big_b_str, rnd_state):

        """
        Sample external Zone Pressures for sheeting, structure and eaves Cpe,
        based on TypeIII General Extreme Value distribution. Prepare effective
        zone areas for load calculations.

        Args:
            wind_dir_index: 0 to 7
            cpe_cov: cov of dist. of CPE for sheeting and batten
            cpe_k: shape parameter of dist. of CPE
            big_a:
            big_b:
            cpe_str_cov: cov. of dist of CPE for rafter
            cpe_str_k
            big_a_str:
            big_b_str:
            rnd_state: default None

        Returns: cpe, cpe_str, cpe_eave

        """

        self.cpe = sample_gev(self.cpe_mean[wind_dir_index],
                              cpe_cov, big_a, big_b, cpe_k, rnd_state)

        self.cpe_str = sample_gev(self.cpe_str_mean[wind_dir_index],
                                  cpe_str_cov, big_a_str, big_b_str, cpe_k, rnd_state)

        self.cpe_eave = sample_gev(self.cpe_eave_mean[wind_dir_index],
                                   cpe_str_cov, big_a_str, big_b_str, cpe_str_k, rnd_state)

    def calc_zone_pressure(self, wind_dir_index, cpi, qz, ms, building_spacing,
                           flag_diff_shielding=False):
        """
        Determine wind pressure loads (Cpe) on each zone (to be distributed onto
        connections)

        Args:
            wind_dir_index:
            cpi: internal pressure coeff
            qz:
            ms:
            building_spacing:
            flag_diff_shielding: flag for differential shielding (default: False)

        Returns:
            pressure : zone pressure

        """

        self.diff_shielding = self.calc_diff_shielding(ms, building_spacing,
                                                       flag_diff_shielding,
                                                       wind_dir_index)

        for item in self.__class__.flags_pressure:
            value = qz * (
                getattr(self, item) - self.cpi_alpha * cpi - self.cpe_eave
                ) * self.diff_shielding
            setattr(self, 'pressure_{}'.format(item), value)

    def calc_diff_shielding(self, ms, building_spacing, flag_diff_shielding,
                            wind_dir_index):

        dsn, dsd = 1.0, 1.0

        if building_spacing > 0 and flag_diff_shielding:
            front_facing = self.is_roof_edge[wind_dir_index]
            if building_spacing == 40 and ms >= 1.0 and front_facing == 0:
                dsd = ms ** 2.0
            elif building_spacing == 20 and front_facing == 1:
                dsd = ms ** 2.0
                if ms <= 0.85:
                    dsn = 0.7 ** 2.0
                else:
                    dsn = 0.8 ** 2.0

        return dsn / dsd

    @staticmethod
    def get_zone_location_from_grid(_zone_grid):
        """
        Create a string location (eg 'A10') from zero based grid refs (col=0,
        row=9)

        Args:
            _zone_grid: tuple

        Returns: string location

        """
        assert isinstance(_zone_grid, tuple)
        return num2str(_zone_grid[0] + 1) + str(_zone_grid[1] + 1)

    @staticmethod
    def get_grid_from_zone_location(_zone_name):
        """
        Extract 0 based grid refs from string location (eg 'A10' to 0, 9)
        """
        chr_part = ''
        for i, item in enumerate(_zone_name):
            try:
                float(item)
            except ValueError:
                chr_part += item
            else:
                break
        num_part = _zone_name.strip(chr_part)
        return str2num(chr_part) - 1, int(num_part) - 1


def str2num(s):
    """
    s: string
    return number
    """
    s = s.strip()
    n_digit = len(s)
    n = 0
    for i, ch_ in enumerate(s, 1):
        ch2num = ord(ch_.upper()) - 64  # A -> 1
        n += ch2num*26**(n_digit-i)
    return n


def num2str(n):
    """
    n: number
    return string
    """
    div = n
    string = ''
    while div > 0:
        module = (div-1) % 26
        string = chr(65 + module) + string
        div = int((div-module)/26)
    return string
