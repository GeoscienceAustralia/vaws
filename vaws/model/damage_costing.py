"""
    Damage Costing Module - costing profiles for a "type of damage"
        - loaded from the database
        - imported from house
        - referenced by damage module to cost damages
"""
import logging

from math import pow


class Costing(object):

    dic_costing = {1: 'type1', 2: 'type2'}

    def __init__(self, costing_name=None, **kwargs):
        """

        Args:
            costing_name: str
            **kwargs:
        """

        assert isinstance(costing_name, str)
        self.name = costing_name

        self.surface_area = None

        self.envelope_repair = None
        self.envelope_repair_rate = None
        self.envelope_factor_formula_type = None
        self.envelope_coeff1 = None
        self.envelope_coeff2 = None
        self.envelope_coeff3 = None

        self.internal_repair = None
        self.internal_repair_rate = None
        self.internal_factor_formula_type = None
        self.internal_coeff1 = None
        self.internal_coeff2 = None
        self.internal_coeff3 = None

        for key, value in kwargs.items():
            setattr(self, key, value)

        for key in ['internal', 'envelope']:
            self.assign_costing_function(key)

    def assign_costing_function(self, key):

        try:
            _value = getattr(self, '{}_factor_formula_type'.format(key))
            setattr(self, '{}_factor_formula_type'.format(key), int(_value))
        except ValueError:
            logging.error('Invalid {}_factor_formula_type: {}'.format(
                key, _value))
        else:
            try:
                _value = getattr(self, '{}_factor_formula_type'.format(key))
                assert _value in self.__class__.dic_costing
            except AssertionError:
                logging.error('Invalid {}_factor_formula_type: {}'.format(
                    key, _value))
            else:
                setattr(self, '{}_repair'.format(key),
                        getattr(self, self.__class__.dic_costing[_value]))

    def compute_cost(self, x):
        assert 0.0 <= x <= 1.0
        envelop_costing = self.envelope_repair(x,
                                               self.envelope_coeff1,
                                               self.envelope_coeff2,
                                               self.envelope_coeff3)
        internal_costing = self.internal_repair(x,
                                                self.internal_coeff1,
                                                self.internal_coeff2,
                                                self.internal_coeff3)
        return x * (self.surface_area * envelop_costing * self.envelope_repair_rate +
                    internal_costing * self.internal_repair_rate)

    @staticmethod
    def type1(x, c1, c2, c3):
        """

        Args:
            x: damage ratio between 0 and 1
            c1: coefficients
            c2:
            c3:

        Returns: c1*x**2 + c2*x + c3

        """
        return c1 * x ** 2 + c2 * x + c3

    @staticmethod
    def type2(x, c1, c2, c3):
        """

        Args:
            x: damage ratio between 0 and 1
            c1: coefficients
            c2:
            c3:

        Returns: c1*x**c2

        """
        try:
            return c1 * pow(x, c2)
        except ValueError:
            return 0.0


class WaterIngressCosting(object):

    dic_costing = {1: 'type1', 2: 'type2'}

    def __init__(self, costing_name=None, **kwargs):
        """

        Args:
            costing_name: str
            **kwargs:
        """

        assert isinstance(costing_name, str)
        self.name = costing_name
        self.cost = None
        self.base_cost = None
        self.water_ingress = None
        self.formula_type = None
        self.coeff1 = None
        self.coeff2 = None
        self.coeff3 = None

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.assign_costing_function()

    def assign_costing_function(self):

        try:
            self.formula_type = int(self.formula_type)
        except ValueError:
            logging.error('Invalid formula_type: {}'.format(self.formula_type))
        else:
            try:
                assert self.formula_type in self.__class__.dic_costing
            except AssertionError:
                logging.error(
                    'Invalid formula_type: {}'.format(self.formula_type))
            else:
                self.cost = getattr(
                    Costing, self.__class__.dic_costing[self.formula_type])

    def compute_cost(self, damage_index):
        assert 0.0 <= damage_index <= 1.0
        return (self.cost(damage_index, self.coeff1, self.coeff2, self.coeff3) *
                self.base_cost)


def compute_water_ingress_given_damage(damage_index, wind_speed,
                                       water_ingress):
    """

    Args:
        damage_index:
        wind_speed:
        water_ingress: pd.DataFrame
    Returns:

    """
    assert 0.0 <= damage_index <= 1.0

    # Note that thresholds are upper values
    idx = water_ingress.index[(water_ingress.index < damage_index).sum()]
    return water_ingress.at[idx, 'wi'](wind_speed)
