"""
    Connection Type Module - reference storage Connections_Types
        - loaded from database
        - imported from '../data/houses/subfolder'
"""
from numpy.random import lognormal
from stats import compute_logarithmic_mean_stddev, \
    compute_arithmetic_mean_stddev
from sqlalchemy import Integer, String, Float, Column, ForeignKey, orm
from sqlalchemy.orm import relation, backref
import database


class ConnectionTypeGroup(object):
    # __tablename__ = 'connection_type_groups'
    # id = Column(Integer, primary_key=True)
    # group_name = Column(String)
    # distribution_order = Column(Integer)
    # distribution_direction = Column(String)
    # trigger_collapse_at = Column(Float)
    # patch_distribution = Column(Integer)
    # set_zone_to_zero = Column(Integer)
    # water_ingress_order = Column(Integer)
    # costing_id = Column(Integer, ForeignKey('damage_costings.id'))
    # house_id = Column(Integer, ForeignKey('houses.id'))
    # costing = relation('DamageCosting', uselist=False,
    #                    backref=backref('conn_types'))
    #
    # def __str__(self):
    #     return "({})".format(self.group_name)

    def perc_damaged(self):
        damaged_perc = 0
        for ct in self.conn_type:
            damaged_perc += ct.perc_damaged()
        return damaged_perc


class ConnectionType(object):
    # __tablename__ = 'connection_types'
    # id = Column(Integer, primary_key=True)
    # connection_type = Column(String)
    # costing_area = Column(Float)
    #
    # # assigned with functions
    # strength_mean = Column(Float)
    # strength_std_dev = Column(Float)
    # deadload_mean = Column(Float)
    # deadload_std_dev = Column(Float)
    #
    # grouping_id = Column(Integer, ForeignKey('connection_type_groups.id'))
    # house_id = Column(Integer, ForeignKey('houses.id'))
    # group = relation(ConnectionTypeGroup, uselist=False,
    #                  backref=backref('conn_types'))
    #
    # def __init__(self, conn_type, costing_area, mu_strength, sd_strength,
    #              mu_deadload, sd_deadload):
    #     """
    #     Args:
    #         conn_type: connection type
    #         costing_area: costing area
    #         mu_strength: arithmetic mean of strength
    #         sd_strength: arithmetic std of strength
    #         mu_deadload:
    #         sd_deadload:
    #     """
    #     self.connection_type = conn_type
    #     self.costing_area = costing_area
    #
    #     self.strength_mean, self.strength_std_dev = \
    #         compute_logarithmic_mean_stddev(mu_strength, sd_strength)
    #
    #     self.deadload_mean, self.deadload_std_dev = \
    #         compute_logarithmic_mean_stddev(mu_deadload, sd_deadload)
    #
    # def __str__(self):
    #     return "({}/{})".format(self.group.group_name, self.connection_type)

    @orm.reconstructor
    def reset_results(self):
        self.result_num_damaged = 0

    def incr_damaged(self):
        self.result_num_damaged += 1

    def perc_damaged(self):
        try:
            return float(self.result_num_damaged) / len(self.connections_of_type)
        except ZeroDivisionError:
            return 0

    def sample_strength(self, mean_factor, cov_factor):
        """

        Args:
            mean_factor: factor adjusting arithmetic mean strength
            cov_factor: factor adjusting arithmetic cov

        Returns: sample of strength following log normal dist.

        """
        mu, std = compute_arithmetic_mean_stddev(self.strength_mean,
                                                 self.strength_std_dev)
        mu *= mean_factor
        std *= cov_factor

        mu_lnx, std_lnx = compute_logarithmic_mean_stddev(mu, std)

        try:
            return lognormal(mu_lnx, std_lnx)
        except ValueError:
            return 0.0

    def sample_deadload(self):
        """

        Returns: sample of dead load following log normal dist.
        Note that mean and/or std can be zero for some components
        """
        try:
            return lognormal(self.deadload_mean,
                             self.deadload_std_dev)
        except ValueError:
            return 0.0

