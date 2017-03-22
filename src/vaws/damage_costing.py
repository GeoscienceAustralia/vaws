"""
    Damage Costing Module - costing profiles for a "type of damage"
        - loaded from the database
        - imported from house
        - referenced by damage module to cost damages
"""
import logging

from math import pow


class Costing(object):

    dic_costing = {1: 'func_type1', 2: 'func_type2'}

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

        default_attr = dict(surface_area=None,
                            envelope_repair_rate=None,
                            envelope_factor_formula_type=None,
                            envelope_coeff1=None,
                            envelope_coeff2=None,
                            envelope_coeff3=None,
                            internal_repair_rate=None,
                            internal_factor_formula_type=None,
                            internal_coeff1=None,
                            internal_coeff2=None,
                            internal_coeff3=None)
        default_attr.update(kwargs)
        for key, value in default_attr.iteritems():
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
                assert _value in Costing.dic_costing
            except AssertionError:
                logging.error('Invalid {}_factor_formula_type: {}'.format(
                    key, _value))
            else:
                setattr(self, '{}_repair'.format(key),
                        getattr(self, Costing.dic_costing[_value]))

    def calculate_cost(self, x):
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
    def func_type1(x, c1, c2, c3):
        """

        Args:
            x: damage ratio between 0 and 1
            c1: coefficients
            c2:
            c3:

        Returns: c1*x**2 + c2*x + c3

        """
        return c1 * pow(x, 2) + c2 * x + c3

    @staticmethod
    def func_type2(x, c1, c2, c3):
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

