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

        self.cpe = None
        self.cpe_str = None
        self.cpe_eave = None
        self.pressure_cpe = 0.0
        self.pressure_cpe_str = 0.0

    def __str__(self):
        return 'Zone(name={}, area={:.2f}, cpi_alpha={:.2f})'.format(
            self.name, self.area, self.cpi_alpha)

    def __repr__(self):
        return 'Zone(name={})'.format(self.name)

    def sample_cpe(self, cpe_cv, cpe_k, big_a, big_b,
                   cpe_str_cv, cpe_str_k, big_a_str, big_b_str, rnd_state):

        """
        Sample external Zone Pressures for sheeting, structure and eaves Cpe,
        based on TypeIII General Extreme Value distribution. Prepare effective
        zone areas for load calculations.

        Args:
            cpe_cv: cov of dist. of CPE for sheeting and batten
            cpe_k: shape parameter of dist. of CPE
            big_a:
            big_b:
            cpe_str_cv: cov. of dist of CPE for rafter
            cpe_str_k
            big_a_str:
            big_b_str:
            rnd_state: default None

        Returns: cpe, cpe_str, cpe_eave

        """

        self.cpe = sample_gev(self.cpe_mean[self.wind_dir_index],
                              cpe_cv, cpe_k, big_a, big_b, rnd_state)

        self.cpe_str = sample_gev(self.cpe_str_mean[self.wind_dir_index],
                                  cpe_str_cv, cpe_str_k, big_a_str, big_b_str, rnd_state)

        self.cpe_eave = sample_gev(self.cpe_eave_mean[self.wind_dir_index],
                                   cpe_str_cv, cpe_str_k, big_a_str, big_b_str, rnd_state)

    def calc_zone_pressure(self, cpi, qz, combination_factor):
        """
        Determine wind pressure loads (Cpe) on each zone (to be distributed onto
        connections)

        Args:
            cpi: internal pressure coefficient
            qz: free stream wind pressure
            combination_factor: action combination factor

        Returns:
            pressure : zone pressure

        """

        self.pressure_cpe = qz * combination_factor * (
            self.cpe - self.cpi_alpha * cpi - self.cpe_eave) * self.differential_shielding

        self.pressure_cpe_str = qz * combination_factor * (
            self.cpe_str - self.cpi_alpha * cpi - self.cpe_eave) * self.differential_shielding

    @property
    def differential_shielding(self):
        """
        The following recommendations for differential shielding for buildings
        deemed to be subject to full shielding are made:

        - For outer suburban situations and country towns (building_spacing=40m):
            Neglect shielding effects (self.shielding_multiplier=1.0),
            except for the leading edges of upwind roofs.
            For the latter an implied pressure ratio of Ms^2 (equal to
            0.85^2 for the full shielding cases (Ms=0.85), and
            0.95^2 for partial shielding cases (Ms=0.95)) can be adopted.

        - For inner suburban buildings (building_spacing=20m) with full shielding (Ms=0.85):
            Reduce the shielding multiplier to 0.7 for upwind roof areas,
            except adjacent to the ridge (implying a pressure reduction factor
            of 0.49). Retain a nominal value of Ms of 0.85 for all other surfaces.

        - For inner suburban buildings (building_spacing=20m) deemed to have partial shielding (Ms=0.95):
            Reduce the shielding multiplier to 0.8 for upwind roof areas,
            except adjacent to the ridge (implying a pressure reduction factor
            of 0.64). Retain a nominal value of Ms of 0.95 for all other surfaces.

        differential_shielding_factor = Ms,adj**2 / Ms ** 2


        """
        adjusted_shielding_multiplier = self.shielding_multiplier  # default

        if self.flag_differential_shielding:
            edge_of_upwind_roofs = self.is_roof_edge[self.wind_dir_index]
            if self.building_spacing == 40:  # outer suburban buildings
                if not edge_of_upwind_roofs:
                    adjusted_shielding_multiplier = 1.0

            elif self.building_spacing == 20:  # inner suburban buildings
                if edge_of_upwind_roofs:
                    if self.shielding_multiplier == 0.85:
                        adjusted_shielding_multiplier = 0.7
                    elif self.shielding_multiplier == 0.95:
                        adjusted_shielding_multiplier = 0.8

        return (adjusted_shielding_multiplier / self.shielding_multiplier) ** 2

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
