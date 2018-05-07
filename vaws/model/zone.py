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

    flag_pressure = ['cpe', 'cpe_str']

    def __init__(self, name=None, **kwargs):
        """
        Args:
            name
        """

        try:
            assert isinstance(name, str)
        except AssertionError:
            name = str(name)

        self.name = name
        self.area = None
        self.cpi_alpha = None
        self.cpe_mean = {}
        self.cpe_str_mean = {}
        self.cpe_eave_mean = {}
        self.is_roof_edge = {}
        self.wind_dir_index = None
        self.shielding_multiplier = None
        self.building_spacing = None
        self.flag_differential_shielding = None

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.differential_shielding = None
        self.cpe = None
        self.cpe_str = None
        self.cpe_eave = None
        self.pressure_cpe = 0.0
        self.pressure_cpe_str = 0.0

        self.set_differential_shielding()

    def __str__(self):
        return 'Zone(name={}, area={:.2f}, cpi_alpha={:.2f})'.format(
            self.name, self.area, self.cpi_alpha)

    def __repr__(self):
        return 'Zone(name={})'.format(self.name)

    def sample_cpe(self, cpe_cov, cpe_k, big_a, big_b,
                   cpe_str_cov, cpe_str_k, big_a_str, big_b_str, rnd_state):

        """
        Sample external Zone Pressures for sheeting, structure and eaves Cpe,
        based on TypeIII General Extreme Value distribution. Prepare effective
        zone areas for load calculations.

        Args:
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

        self.cpe = sample_gev(self.cpe_mean[self.wind_dir_index],
                              cpe_cov, big_a, big_b, cpe_k, rnd_state)

        self.cpe_str = sample_gev(self.cpe_str_mean[self.wind_dir_index],
                                  cpe_str_cov, big_a_str, big_b_str, cpe_str_k, rnd_state)

        self.cpe_eave = sample_gev(self.cpe_eave_mean[self.wind_dir_index],
                                   cpe_str_cov, big_a_str, big_b_str, cpe_str_k, rnd_state)

    def calc_zone_pressure(self, cpi, qz):
        """
        Determine wind pressure loads (Cpe) on each zone (to be distributed onto
        connections)

        Args:
            cpi: internal pressure coefficient
            qz: free stream wind pressure

        Returns:
            pressure : zone pressure

        """

        for item in self.__class__.flag_pressure:
            value = qz * (
                getattr(self, item) - self.cpi_alpha * cpi - self.cpe_eave
                ) * self.differential_shielding
            setattr(self, 'pressure_{}'.format(item), value)

    def set_differential_shielding(self):
        """
        The following recommendations for differential shielding for buildings
        deemed to be subject to full shielding are made:

        - For outer suburban situations and country towns:
            Neglect shielding effects except for the leading edges of upwind
            roofs. For the latter an implied pressure ratio of Ms2 (equal to
            0.852 for the full shielding cases, and0.95 for partial shielding
            cases) can be adopted.
        - For inner suburban buildings with full shielding:
            Reduce the shielding multiplier to 0.7 for upwind roof areas,
            except adjacent to the ridge (implying a pressure reduction factor
            of 0.49). Retain a nominal value of Ms of 0.85 for all other surfaces.
        - For inner suburban buildings deemed to have partial shielding:
            Reduce the shielding multiplier to 0.8 for upwind roof areas,
            except adjacent to the ridge (implying a pressure reduction factor
            of 0.64). Retain a nominal value of Ms of 0.95 for all other surfaces.

        """

        dsn, dsd = 1.0, 1.0

        if self.building_spacing > 0 and self.flag_differential_shielding:
            front_facing = self.is_roof_edge[self.wind_dir_index]
            if self.building_spacing == 40 and self.shielding_multiplier >= 1.0 and front_facing == 0:
                dsd = self.shielding_multiplier ** 2.0
            elif self.building_spacing == 20 and front_facing == 1:
                dsd = self.shielding_multiplier ** 2.0
                if self.shielding_multiplier <= 0.85:
                    dsn = 0.7 ** 2.0
                else:
                    dsn = 0.8 ** 2.0

        self.differential_shielding = dsn / dsd

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
