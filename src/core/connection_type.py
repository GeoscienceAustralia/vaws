"""
    Connection Type Module - reference storage Connections_Types
        - loaded from database
        - imported from '../data/houses/subfolder'
"""
import numpy as np
from stats import lognormal_mean, lognormal_stddev
from sqlalchemy import Integer, String, Float, Column, ForeignKey
from sqlalchemy.orm import relation, backref
import database
import stats


class ConnectionTypeGroup(database.Base):
    __tablename__ = 'connection_type_groups'
    id = Column(Integer, primary_key=True)
    group_name = Column(String)
    distribution_order = Column(Integer)
    distribution_direction = Column(String)
    trigger_collapse_at = Column(Float)
    patch_distribution = Column(Integer)
    set_zone_to_zero = Column(Integer)
    water_ingress_order = Column(Integer)
    costing_id = Column(Integer, ForeignKey('damage_costings.id'))
    house_id = Column(Integer, ForeignKey('houses.id'))
    costing = relation('DamageCosting', uselist=False,
                       backref=backref('conn_types'))

    def __str__(self):
        return "({})".format(self.group_name)

    def perc_damaged(self):
        damaged_perc = 0
        for ct in self.conn_type:
            damaged_perc += ct.perc_damaged()
        return damaged_perc


class ConnectionType(database.Base):
    __tablename__ = 'connection_types'
    id = Column(Integer, primary_key=True)
    connection_type = Column(String)
    costing_area = Column(Float)
    strength_mean = Column(Float)
    strength_std_dev = Column(Float)
    deadload_mean = Column(Float)
    deadload_std_dev = Column(Float)
    grouping_id = Column(Integer, ForeignKey('connection_type_groups.id'))
    house_id = Column(Integer, ForeignKey('houses.id'))
    group = relation(ConnectionTypeGroup, uselist=False,
                     backref=backref('conn_types'))

    def __init__(self, ct, ca, sm, ssd, dm, dsd):

        self.deadload_mean = 0
        self.deadload_std_dev = 0
        self.strength_mean = 0
        self.strength_std_dev = 0
        self.result_num_damaged = 0

        self.connection_type = ct
        self.costing_area = ca
        self.set_strength_params(sm, ssd)
        self.set_deadload_params(dm, dsd)
        # self.reset_results()

    def __str__(self):
        return "({}/{})".format(self.group.group_name, self.connection_type)

    def reset_results(self):
        self.result_num_damaged = 0

    def incr_damaged(self):
        self.result_num_damaged += 1

    def perc_damaged(self):
        num_conns_of_type = len(self.connections_of_type)
        if num_conns_of_type > 0:
            return float(self.result_num_damaged) / float(num_conns_of_type)
        else:
            return 0

    def set_strength_params(self, mean, stddev):
        if mean > 0 and stddev > 0:
            self.strength_mean = lognormal_mean(mean, stddev)
            self.strength_std_dev = lognormal_stddev(mean, stddev)

    def set_deadload_params(self, mean, stddev):
        self.deadload_mean = 0
        self.deadload_std_dev = 0
        if mean > 0 and stddev > 0:
            self.deadload_mean = lognormal_mean(mean, stddev)
            self.deadload_std_dev = lognormal_stddev(mean, stddev)

    def sample_strength(self, mean_factor, cov_factor):
        real_mean = stats.lognormal_underlying_mean(self.strength_mean,
                                                    self.strength_std_dev)
        real_stddev = stats.lognormal_underlying_stddev(self.strength_mean,
                                                        self.strength_std_dev)
        real_mean = real_mean * mean_factor
        real_stddev = real_stddev * cov_factor
        mean = lognormal_mean(real_mean, real_stddev)
        stddev = lognormal_stddev(real_mean, real_stddev)
        return np.random.lognormal(mean, stddev)

    def sample_deadload(self):
        if self.deadload_mean == 0 or self.deadload_std_dev == 0:
            return 0
        else:
            return np.random.lognormal(self.deadload_mean,
                                          self.deadload_std_dev)

#
# unit tests
# if __name__ == '__main__':
#    import unittest
#    import damage
#    import scenario
#    import np.random
#    import matplotlib.pyplot as plt
#
#    verb = False
#    damage.configure('../model.db', verbose=verb)
#    
#    class MyTestCase(unittest.TestCase):
#        def setUp(self):
#            self.s = scenario.loadFromCSV('../scenarios/carl1.csv')
#                    
#        def test_lognormal(self):
#            m = 70.0
#            stddev = 14.0
#            m2 = lognormal_mean(m, stddev)
#            stddev2 = lognormal_stddev(m, stddev)
#            self.assertAlmostEqual(stddev2, 0.1980422)
#            self.assertAlmostEqual(m2, 4.228884885)
#            
#        def test_pdfs(self):
#            for ct in self.s.house.conn_types:
#                if True or ct.connection_type == 'piers':
#                    x = []
#                    for i in xrange(50000):
#                        print 'ctype mean(%.3f), stddev(%.3f)' % (ct.strength_mean, ct.strength_std_dev)
#                        rv = np.random.lognormal(ct.strength_mean, ct.strength_std_dev)
#                        x.append(rv)
#                    n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
#                    plt.title(ct.connection_type)
#                    plt.grid(True)
#                    plt.show()
#
#    suite = unittest.TestLoader().loadTestsFromTestCase(MyTestCase)
#    unittest.TextTestRunner(verbosity=2).run(suite)
