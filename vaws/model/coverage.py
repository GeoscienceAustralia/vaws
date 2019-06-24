"""Coverage Module

    This module contains Coverage class.

"""

import logging

from vaws.model.zone import Zone
from vaws.model.stats import sample_lognormal


class Coverage(Zone):

    def __init__(self, name=None, logger=None, **kwargs):

        try:
            assert isinstance(name, str)
        except AssertionError:
            name = str(name)

        self.logger = logger or logging.getLogger(__name__)

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
        self.cpe_cv = None
        self.cpe_k = None
        self.cpe_str_cv = None
        self.cpe_str_k = None
        self.big_a = None
        self.big_b = None
        self.big_a_str = None
        self.big_b_str = None
        self.rnd_state = None

        super(Coverage, self).__init__(name=name, **kwargs)

        # default value for coverage
        self.cpi_alpha = 1.0
        self.cpe_str_mean = {i: 0 for i in range(8)}
        self.cpe_eave_mean = {i: 0 for i in range(8)}
        self.is_roof_edge = {i: 0 for i in range(8)}

        self._strength_negative = None  # out
        self._strength_positive = None  # in
        self._momentum_capacity = None
        self.load = None
        self.capacity = -1
        self.breached = 0
        self._breached_area = 0

    def __str__(self):
        return 'Coverage(name={}, area={:.2f})'.format(
            self.name, self.area)

    def __repr__(self):
        return 'Coverage(name={})'.format(self.name)

    @property
    def momentum_capacity(self):
        if self._momentum_capacity is None:
            self._momentum_capacity = sample_lognormal(
                *self.log_failure_momentum, rnd_state=self.rnd_state)
        return self._momentum_capacity

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

        # assert isinstance(value, Number)

        # breached area can be accumulated but not exceeding area
        self._breached_area += value
        self._breached_area = min(self._breached_area, self.area)

    @property
    def strength_positive(self):
        if self._strength_positive is None:
            self._strength_positive = sample_lognormal(
                *self.log_failure_strength_in, rnd_state=self.rnd_state)
        return self._strength_positive

    @property
    def strength_negative(self):
        if self._strength_negative is None:
            self._strength_negative = -1.0 * sample_lognormal(
                *self.log_failure_strength_out, rnd_state=self.rnd_state)
        return self._strength_negative

    def check_damage(self, qz, cpi, combination_factor, wind_speed):
        """

        :param qz:
        :param cpi:
        :param combination_factor:
        :param wind_speed:
        :return:
        """

        msg1 = 'load at coverage {name}: {qz:.3f} * ({cpe:.3f} - ' \
               '{cpi:.3f}) * {area:.3f}'
        msg2 = 'coverage {name} failed at {speed:.3f} b/c ' \
               '{positive:.3f} or {negative:.3f} < {load:.3f} ' \
               '-> breached area {area:.3f}'

        self.load = 0.0

        # only skip breached window
        if (not (self.breached and self.description == 'window') and
                (self.breached_area < self.area)):

            self.load = qz * (self.cpe - cpi) * self.area * combination_factor

            self.logger.debug(msg1.format(name=self.name,
                                          qz=qz,
                                          cpe=self.cpe,
                                          cpi=cpi,
                                          area=self.area))

            if (self.load > self.strength_positive) or (
                        self.load < self.strength_negative):

                self.breached = 1

                self.breached_area = self.area

                self.capacity = wind_speed

                self.logger.debug(msg2.format(name=self.name,
                                              speed=wind_speed,
                                              positive=self.strength_positive,
                                              negative=self.strength_negative,
                                              load=self.load,
                                              area=self.breached_area))
