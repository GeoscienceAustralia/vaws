"""
    Damage Costing Module - costing profiles for a "type of damage"
        - loaded from the database
        - imported from house
        - referenced by damage module to cost damages
"""

from vaws.model.constants import COSTING_FORMULA_TYPES


class Costing(object):

    def __init__(self, name=None, **kwargs):
        """

        Args:
            costing_name: str
            **kwargs:
        """

        assert isinstance(name, str)
        self.name = name

        self.surface_area = None

        self._envelope_repair = None
        self.envelope_repair_rate = None
        self.envelope_factor_formula_type = None
        self.envelope_coeff1 = None
        self.envelope_coeff2 = None
        self.envelope_coeff3 = None

        self._internal_repair = None
        self.internal_repair_rate = None
        self.internal_factor_formula_type = None
        self.internal_coeff1 = None
        self.internal_coeff2 = None
        self.internal_coeff3 = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def envelope_repair(self):
        msg = 'Invalid envelope factor formula type'
        if self._envelope_repair is None:
            assert self.envelope_factor_formula_type in COSTING_FORMULA_TYPES, msg
            self._envelope_repair = getattr(
                    self, f'type{self.envelope_factor_formula_type:.0f}')
        return self._envelope_repair

    @property
    def internal_repair(self):
        msg = 'Invalid internal factor formula type'
        if self._internal_repair is None:
            assert self.internal_factor_formula_type in COSTING_FORMULA_TYPES, msg
            self._internal_repair = getattr(
                    self, f'type{self.internal_factor_formula_type:.0f}')
        return self._internal_repair

    def compute_cost(self, x):
        if x:
            envelop_costing = self.envelope_repair(x=x,
                                                   c1=self.envelope_coeff1,
                                                   c2=self.envelope_coeff2,
                                                   c3=self.envelope_coeff3)
            internal_costing = self.internal_repair(x=x,
                                                    c1=self.internal_coeff1,
                                                    c2=self.internal_coeff2,
                                                    c3=self.internal_coeff3)
            return x * (self.surface_area * envelop_costing * self.envelope_repair_rate +
                        internal_costing * self.internal_repair_rate)
        else:
            return 0.0

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
        if x:
            return c1 * x ** c2
        else:
            return 0.0


class WaterIngressCosting(object):

    def __init__(self, name=None, **kwargs):
        """

        Args:
            costing_name: str
            **kwargs:
        """

        assert isinstance(name, str)
        self.name = name
        self._cost = None
        self.base_cost = None
        self.water_ingress = None
        self.formula_type = None
        self.coeff1 = None
        self.coeff2 = None
        self.coeff3 = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def cost(self):
        msg = 'Invalid formula type'
        if self._cost is None:
            assert self.formula_type in COSTING_FORMULA_TYPES, msg
            self._cost = getattr(Costing, f'type{self.formula_type}')
        return self._cost

    def compute_cost(self, x):
        if x:
            return self.base_cost * self.cost(x=x,
                                              c1=self.coeff1,
                                              c2=self.coeff2,
                                              c3=self.coeff3)
        else:
            return 0.0


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
