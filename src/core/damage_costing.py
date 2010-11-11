'''
    Damage Costing Module - costing profiles for a "type of damage"
        - loaded from the database
        - imported from house
        - referenced by damage module to cost damages
'''
import math
from sqlalchemy import create_engine, Table, Integer, String, Float, Column, MetaData, ForeignKey
from sqlalchemy.orm import relation, backref
import database

## --------------------------------------------------------------------------------------
class DamageFactoring(database.Base):
    __tablename__                   = 'damage_factorings'
    parent_id                       = Column(Integer, ForeignKey('connection_type_groups.id'), primary_key=True)
    factor_id                       = Column(Integer, ForeignKey('connection_type_groups.id'), primary_key=True)
    house_id                        = Column(Integer, ForeignKey('houses.id'))
    parent                          = relation("ConnectionTypeGroup", 
                                               primaryjoin="damage_factorings.c.parent_id==connection_type_groups.c.id")
    factor                          = relation("ConnectionTypeGroup", 
                                               primaryjoin="damage_factorings.c.factor_id==connection_type_groups.c.id")
    
## --------------------------------------------------------------------------------------
class DamageCosting(database.Base):
    __tablename__                   = 'damage_costings'
    id                              = Column(Integer, primary_key=True)
    costing_name                    = Column(String)
    area                            = Column(Float)
    envelope_factor_formula_type    = Column(Integer)
    envelope_repair_rate            = Column(Float)
    env_coeff_1                     = Column(Float)
    env_coeff_2                     = Column(Float)
    env_coeff_3                     = Column(Float)
    internal_factor_formula_type    = Column(Integer)
    internal_repair_rate            = Column(Float)
    int_coeff_1                     = Column(Float)
    int_coeff_2                     = Column(Float)
    int_coeff_3                     = Column(Float)
    house_id                        = Column(Integer, ForeignKey('houses.id'))

    def __repr__(self):
        return "<DamageCosting('%s', '%f m', '%d', '$ %f', '%f', '%f', '%f', '%d', '$ %f', '%f', '%f', '%f')>" % (
            self.costing_name,
            self.area,
            self.envelope_factor_formula_type,
            self.envelope_repair_rate,
            self.env_coeff_1,
            self.env_coeff_2,
            self.env_coeff_3,
            self.internal_factor_formula_type,
            self.internal_repair_rate,
            self.int_coeff_1,
            self.int_coeff_2,
            self.int_coeff_3)
        
    def env_func1(self, X):
        return (self.env_coeff_1 * math.pow(X, 2)) + (self.env_coeff_2 * X) + self.env_coeff_3

    def env_func2(self, X):
        return 0 if X == 0 else self.env_coeff_1 * math.pow(X, self.env_coeff_2)

    def lining_func1(self, X):
        return (self.int_coeff_1 * math.pow(X, 2)) + (self.int_coeff_2 * X) + self.int_coeff_3

    def lining_func2(self, X):
        return 0 if X == 0 else self.int_coeff_1 * math.pow(X, self.int_coeff_2)
    
    def calc_lining_repair_cost(self, X):
        f = self.lining_func1
        if self.internal_factor_formula_type == 2:
            f = self.lining_func2
        lrc = X * f(X) * self.internal_repair_rate
        return lrc

    def calc_envelope_repair_cost(self, X):
        f = self.env_func1
        if self.envelope_factor_formula_type == 2:
            f = self.env_func2
        erc = X * self.area * f(X) * self.envelope_repair_rate
        return erc
    
    def calculate_damage(self, perc):
        return self.calc_envelope_repair_cost(perc) + self.calc_lining_repair_cost(perc) 
    






