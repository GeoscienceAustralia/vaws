"""Coverage Module

    This module contains Coverage class.

"""

import logging
from numbers import Number

from vaws.model.zone import Zone
from vaws.model.stats import sample_lognormal


class Coverage(Zone):

    def __init__(self, name=None, **kwargs):

        try:
            assert isinstance(name, str)
        except AssertionError:
            name = str(name)

        self.area = None
        self.cpe_mean = {}
        self.coverage_type = None
        self.log_failure_strength_in = None
        self.log_failure_strength_out = None
        self.sign_failure_strength_in = None
        self.sign_failure_strength_out = None
        self.log_failure_momentum = None
        self.wall_name = None
        self.wind_dir_index = None
        self.shielding_multiplier = None
        self.building_spacing = None
        self.flag_differential_shielding = None

        super(Coverage, self).__init__(name=name, **kwargs)

        # default value for coverage
        self.cpi_alpha = 1.0
        self.cpe_str_mean = {i: 0 for i in range(8)}
        self.cpe_eave_mean = {i: 0 for i in range(8)}
        self.is_roof_edge = {i: 0 for i in range(8)}

        self.strength_negative = None
        self.strength_positive = None
        self.load = None
        self.capacity = -1
        self.breached = 0
        self._breached_area = 0

    @property
    def breached_area(self):
        return self._breached_area

    @breached_area.setter
    def breached_area(self, value):
        """

        Args:
            value:

        Returns:

        """

        assert isinstance(value, Number)

        # breached area can be accumulated but not exceeding area
        self._breached_area += value
        self._breached_area = min(self._breached_area, self.area)

    def __str__(self):
        return 'Coverage(name={}, area={:.2f})'.format(
            self.name, self.area)

    def __repr__(self):
        return 'Coverage(name={})'.format(self.name)

    def sample_strength(self, rnd_state):
        """

        :param rnd_state:
        :return:
        """

        for _type in ['in', 'out']:
            _strength = getattr(self, 'log_failure_strength_{}'.format(_type))
            _sign = getattr(self, 'sign_failure_strength_{}'.format(_type))
            _value = sample_lognormal(*(_strength + (rnd_state,)))

            if _sign > 0:
                self.strength_positive = _value
            else:
                self.strength_negative = -1.0 * _value

    def check_damage(self, qz, cpi, wind_speed):
        """

        :param qz:
        :param cpi:
        :param wind_speed:
        :return:
        """

        self.load = 0.0

        if not self.breached:

            self.load = 0.9 * qz * (self.cpe - cpi) * self.area

            logging.debug(
                'load at coverage {}: {:.3f} * ({:.3f} - {:.3f}) * {:.3f}'.format(
                    self.name, qz, self.cpe, cpi, self.area))

            if (self.load > self.strength_positive) or (
                        self.load < self.strength_negative):

                self.breached = 1

                self._breached_area = self.area

                self.capacity = wind_speed

                logging.info(
                    'coverage {} failed at {:.3f} b/c {:.3f} or {:.3f} < {:.3f} -> breached area {:.3f}'.format(
                        self.name, wind_speed, self.strength_positive, self.strength_negative, self.load, self.breached_area))

